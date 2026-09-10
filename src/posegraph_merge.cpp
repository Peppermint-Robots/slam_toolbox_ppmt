// Offline merge of two serialized slam_toolbox pose-graphs: base_map and updater_map.
//
// Loads base_map and updater_map - both assumed to already be expressed in the same coordinate
// frame, no alignment step performed - then replays both graphs' scans, in turn, through a
// brand-new third Mapper, `fused_mapper`. Neither base_map's nor updater_map's own loaded
// Mapper is ever mutated or saved; fused_mapper is what gets written, in the current working
// directory, as merged(.posegraph/.data/.png/.yaml/_graph.png).
//
// Both graphs are treated identically: base_map's scans are replayed first, then updater_map's,
// each through fused_mapper's own Mapper::Process()/ProcessAgainstNodesNearBy(), backed by a
// real CeresSolver - exactly as DecentralizedMultiRobotSlamToolbox::addExternalScan() does for
// live multi-robot merging (src/slam_toolbox_decentralized_multirobot.cpp on the ros2 branch).
// Neither graph's original edges/covariances are trusted as-is - both get rebuilt from scratch
// by fused_mapper's own scan matcher, and both get an equal chance to loop-close against
// whatever is already in fused_mapper when their turn comes. Overlapping regions loop-close
// instead of being drawn twice, and CorrectPoses() is forced once at the end regardless of
// whether a loop closure happened to trigger it, so the saved poses reflect a converged solve.
// fused_mapper's non-anchor scan poses CAN shift wherever real evidence ties the two graphs
// together (CeresSolver only pins the single first-ever-added node constant) - expected, not a
// bug: if this merged file is later loaded live by slam_toolbox, loadSerializedPoseGraph() would
// re-run the same kind of optimization anyway.
//
// This is an offline CLI tool: it needs a bare, unspun rclcpp::Node only to Configure() the
// Ceres solver (CeresSolver::Configure reads a handful of ROS parameters); nothing here ever
// spins. Like this fork's other offline pose-graph tools, every Mapper/Dataset (base_map's,
// updater_map's, and fused_mapper/fused_dataset) is heap-allocated and deliberately never freed -
// see finish() for why.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <yaml-cpp/yaml.h>

#include "karto_sdk/Mapper.h"
#include "solvers/ceres_solver.hpp"

namespace
{

/** Exit immediately, running no destructor.
 *
 * karto's Mapper/Dataset teardown is a path slam_toolbox itself never exercises - it keeps both
 * alive for the lifetime of the process - and this fork's own offline tools have observed it
 * abort with "stack smashing detected", after all real work had already completed. A
 * one-shot batch tool gains nothing from that teardown: the OS reclaims the memory either way,
 * and skipping it means a successful run cannot be turned into a failed exit code by a
 * destructor. _Exit does not flush, so flush first.
 */
[[noreturn]] void finish(int code)
{
  std::cout.flush();
  std::cerr.flush();
  std::fflush(nullptr);
  std::_Exit(code);
}

struct Stats
{
  size_t nodes_from_base_map = 0;
  size_t nodes_from_updater_map = 0;
  size_t base_map_merged = 0;
  size_t base_map_dropped = 0;
  size_t updater_map_merged = 0;
  size_t updater_map_dropped = 0;

  // Local scan-matcher confidence per merged scan, from both graphs (translational covariance
  // trace, i.e. cov(0,0)+cov(1,1) - smaller is a tighter/more confident match) and how far
  // CorrectPoses()'s global solve then moved things from that local-match placement.
  double cov_trace_min = std::numeric_limits<double>::infinity();
  double cov_trace_max = 0.0;
  double cov_trace_sum = 0.0;

  size_t base_map_scans_shifted = 0;
  double base_map_shift_max = 0.0;
  double base_map_shift_sum = 0.0;

  double updater_map_shift_max = 0.0;
  double updater_map_shift_sum = 0.0;
};

/** Write <stem>.png + <stem>.yaml: the occupancy grid built from every scan in `mapper`, in the
 * classification convention this fork's other offline pose-graph tools already use: free=254,
 * occupied=0, unknown=205, negate=0, default thresholds. Karto's OccupancyGrid stores row 0 at
 * the WORLD-MINIMUM y, but a PNG/cv::Mat row 0 is the TOP of the image (world-maximum y), so row
 * order is flipped on the way out.
 *
 * If `display_graph` is set, the fused pose graph is drawn on top of that same image before it's
 * written: every vertex as a small filled circle (labeled with its UniqueId) in one uniform
 * color. Edges get one of two uniform colors: green for an edge whose two endpoints came from
 * the same source graph, red for an edge that connects a base_map node to an updater_map node -
 * i.e. an edge that actually ties the two graphs together, found during replay (either a genuine
 * TryCloseLoop() loop closure, or the one ordinary "link to previous scan" edge AddEdges()
 * creates for updater_map's very first replayed scan - see replayGraph's docs). `updater_map_scans`
 * is the membership set used for that classification; pass an empty set to draw every edge green.
 */
bool saveMapImage(
  karto::Mapper * mapper, bool display_graph,
  const std::set<karto::LocalizedRangeScan *> & updater_map_scans, double resolution,
  const std::string & stem, std::string & err)
{
  const karto::LocalizedRangeScanVector scans = mapper->GetAllProcessedScans();
  std::unique_ptr<karto::OccupancyGrid> occ_grid(
    karto::OccupancyGrid::CreateFromScans(scans, resolution));
  if (!occ_grid) {
    err = "no scans to build a map from";
    return false;
  }

  const int width = occ_grid->GetWidth();
  const int height = occ_grid->GetHeight();
  const karto::Vector2<kt_double> offset = occ_grid->GetCoordinateConverter()->GetOffset();

  cv::Mat gray(height, width, CV_8UC1, cv::Scalar(205));
  for (int y = 0; y < height; ++y) {
    const int row = height - 1 - y;
    for (int x = 0; x < width; ++x) {
      const kt_int8u value = occ_grid->GetValue(karto::Vector2<kt_int32s>(x, y));
      uint8_t pixel = 205;
      if (value == karto::GridStates_Occupied) {
        pixel = 0;
      } else if (value == karto::GridStates_Free) {
        pixel = 254;
      }
      gray.at<uint8_t>(row, x) = pixel;
    }
  }

  cv::Mat image;
  if (!display_graph) {
    image = gray;
  } else {
    cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);

    auto toPixel = [&](const karto::Pose2 & pose) {
        const int gx = static_cast<int>(std::lround((pose.GetX() - offset.GetX()) / resolution));
        const int gy = static_cast<int>(std::lround((pose.GetY() - offset.GetY()) / resolution));
        return cv::Point(gx, height - 1 - gy);
      };

    const cv::Scalar intra_edge_color(0, 200, 0);  // green (BGR) - both endpoints, same source
    const cv::Scalar cross_edge_color(0, 0, 255);  // red - ties base_map to updater_map
    const cv::Scalar vertex_color(255, 128, 0);    // blue-ish (BGR), uniform for every vertex

    for (auto * edge : mapper->GetGraph()->GetEdges()) {
      if (edge == nullptr || edge->GetSource() == nullptr || edge->GetTarget() == nullptr) {
        continue;
      }
      auto * source = edge->GetSource()->GetObject();
      auto * target = edge->GetTarget()->GetObject();
      if (source == nullptr || target == nullptr) {
        continue;
      }
      const bool crosses =
        (updater_map_scans.count(source) != 0) != (updater_map_scans.count(target) != 0);
      cv::line(
        image, toPixel(source->GetCorrectedPose()), toPixel(target->GetCorrectedPose()),
        crosses ? cross_edge_color : intra_edge_color, crosses ? 1 : 1, cv::LINE_AA);
    }

    for (const auto & by_sensor : mapper->GetGraph()->GetVertices()) {
      for (const auto & entry : by_sensor.second) {
        if (entry.second == nullptr || entry.second->GetObject() == nullptr) {
          continue;
        }
        auto * scan = entry.second->GetObject();
        const cv::Point px = toPixel(scan->GetCorrectedPose());
        cv::circle(image, px, 1, vertex_color, cv::FILLED, cv::LINE_AA);
        // cv::putText(
        //   image, std::to_string(scan->GetUniqueId()), px + cv::Point(4, -4),
        //   cv::FONT_HERSHEY_SIMPLEX, 0.20, cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
      }
    }
  }

  const std::string png_path = stem + ".png";
  const std::string yaml_path = stem + ".yaml";
  if (!cv::imwrite(png_path, image)) {
    err = "failed to write '" + png_path + "'";
    return false;
  }

  const size_t slash = png_path.find_last_of('/');
  const std::string png_name = slash == std::string::npos ? png_path : png_path.substr(slash + 1);

  YAML::Emitter yaml;
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "image" << YAML::Value << png_name;
  yaml << YAML::Key << "resolution" << YAML::Value << resolution;
  yaml << YAML::Key << "origin" << YAML::Value << YAML::Flow <<
    std::vector<double>{offset.GetX(), offset.GetY(), 0.0};
  yaml << YAML::Key << "negate" << YAML::Value << 0;
  yaml << YAML::Key << "occupied_thresh" << YAML::Value << 0.65;
  yaml << YAML::Key << "free_thresh" << YAML::Value << 0.196;
  yaml << YAML::EndMap;

  std::ofstream out(yaml_path);
  if (!out) {
    err = "failed to write '" + yaml_path + "'";
    return false;
  }
  out << yaml.c_str() << "\n";
  return true;
}

/** Register every laser in a Dataset with the global SensorManager (LocalizedRangeScan resolves
 * its laser by name lookup through it). Names already present are skipped rather than
 * re-registered: both graphs almost certainly share the same physical sensor name, and
 * re-registering under override would silently repoint base_map's own already-loaded scans at
 * updater_map's copy of that laser's config.
 */
size_t registerLasers(karto::Dataset * dataset, std::set<std::string> & registered_names)
{
  size_t newly_registered = 0;
  for (auto * object : dataset->GetLasers()) {
    auto * laser = dynamic_cast<karto::LaserRangeFinder *>(object);
    if (laser == nullptr) {
      continue;
    }
    const std::string name = laser->GetName().ToString();
    if (registered_names.count(name) != 0) {
      continue;
    }
    karto::SensorManager::GetInstance()->RegisterSensor(laser, false);
    registered_names.insert(name);
    ++newly_registered;
  }
  return newly_registered;
}

/** Replay every scan of `source`'s graph into `fused` via `fused`'s own scan matcher/solver: the
 * first scan uses ProcessAgainstNodesNearBy() (an unconditional nearest-neighbour
 * relocalization - necessary because `fused`'s running-scan buffer has nothing of `source`'s in
 * it yet, so plain Process()'s reliance on running-buffer/"last scan" continuity would have
 * nothing to work with for that first scan); every scan after that uses plain sequential
 * Process(), which also runs TryCloseLoop() against everything already in `fused` (both earlier
 * scans from this same source and, on the second call, all of the other source). This mirrors
 * DecentralizedMultiRobotSlamToolbox::addExternalScan() (slam_toolbox_decentralized_multirobot
 * .cpp) - the same mechanism that feeds a peer robot's scans into the host's own live Mapper.
 *
 * `fused`'s MapperSensorManager assigns each scan a fresh StateId/UniqueId from its own counters
 * as a side effect of Process()/ProcessAgainstNodesNearBy(), so calling this once for base_map
 * and once for updater_map can never collide, regardless of either source's original ids. Process
 * ()/ProcessAgainstNodesNearBy() only touch `fused`'s own graph/sensor-manager structures, never
 * its Dataset, so every accepted scan is added to `fused_dataset` here by hand. `accepted_scans`
 * records every scan this call actually merged in (used later to tell the two sources apart for
 * the shift diagnostics); `pre_correct_pose` records each accepted scan's corrected pose right
 * after its local match, before the caller's later CorrectPoses() call, so the caller can report
 * how far the global solve then moved it. `merged_count`/`dropped_count` are the caller's
 * per-source counters to update (e.g. &stats.base_map_merged, &stats.base_map_dropped).
 * `first_accepted_uid`, if given, receives the fused-graph UniqueId of the first scan this call
 * successfully processed - the caller can use it to recognise the one edge
 * MapperGraph::AddEdges() creates as an ordinary "link to previous scan" for that first scan
 * (Mapper.cpp:1441-1449, since it's the first call into ProcessAgainstNodesNearBy() and the only
 * point where "previous StateId" crosses from one source's id range into the other's) - that one
 * edge is not a TryCloseLoop() loop closure even though it connects the two sources.
 */
void replayGraph(
  karto::Mapper * source, karto::Mapper * fused, karto::Dataset * fused_dataset, Stats & stats,
  size_t & merged_count, size_t & dropped_count,
  std::set<karto::LocalizedRangeScan *> & accepted_scans,
  std::map<int, karto::Pose2> & pre_correct_pose, int * first_accepted_uid = nullptr)
{
  std::vector<karto::LocalizedRangeScan *> scans;
  for (const auto & by_sensor : source->GetGraph()->GetVertices()) {
    for (const auto & entry : by_sensor.second) {
      if (entry.second != nullptr && entry.second->GetObject() != nullptr) {
        scans.push_back(entry.second->GetObject());
      }
    }
  }

  bool first = true;
  for (auto * scan : scans) {
    karto::Matrix3 covariance;
    covariance.SetToIdentity();
    const bool ok = first ?
      fused->ProcessAgainstNodesNearBy(scan, false, &covariance) :
      fused->Process(scan, &covariance);
    first = false;

    if (ok) {
      ++merged_count;
      const double trace = covariance(0, 0) + covariance(1, 1);
      stats.cov_trace_min = std::min(stats.cov_trace_min, trace);
      stats.cov_trace_max = std::max(stats.cov_trace_max, trace);
      stats.cov_trace_sum += trace;
      pre_correct_pose[scan->GetUniqueId()] = scan->GetCorrectedPose();
      fused_dataset->Add(scan);
      accepted_scans.insert(scan);
      if (first_accepted_uid != nullptr && *first_accepted_uid == -1) {
        *first_accepted_uid = scan->GetUniqueId();
      }
    } else {
      ++dropped_count;
      delete scan;
    }
  }
}

void usage(const char * argv0)
{
  std::cerr <<
    "Merge base_map and updater_map into a fused pose graph.\n\n"
    "Usage:\n  " << argv0 <<
    " --base_map <stem> --updater_map <stem> [--display-graph]\n\n"
    "Stems carry no extension: <stem>.posegraph and <stem>.data are both read.\n"
    "base_map and updater_map are assumed to already be expressed in the same\n"
    "  coordinate frame - this tool performs no alignment.\n"
    "base_map's scans are replayed into the fused graph first, then updater_map's -\n"
    "  both through the fused graph's own scan matcher/solver, so overlapping content\n"
    "  reconciles via loop closure instead of being drawn twice, and both graphs get an\n"
    "  equal chance to link into whatever is already there.\n\n"
    "--display-graph draws the fused graph's vertices/edges on top of the occupancy\n"
    "  grid before writing it out: every vertex in one uniform color, an edge within one\n"
    "  source graph in green, and an edge tying base_map to updater_map (a loop closure,\n"
    "  or the one ordinary link created for updater_map's first replayed scan) in red.\n"
    "  Cross-graph edges are also logged to stdout. Without this flag, the image is the\n"
    "  plain occupancy grid.\n\n"
    "Output is always written in the current working directory, as merged.posegraph/\n"
    "  .data (the fused graph) and merged.png/.yaml (its occupancy grid).\n";
}

}  // namespace

int main(int argc, char ** argv)
{

  /// Parse command-line arguments

  std::cout << "\n\n---\n\nStarting: parse command-line arguments" << std::endl;

  std::string base_map_stem;
  std::string updater_map_stem;
  bool display_graph = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = (i + 1) < argc;
    if (arg == "--base_map" && has_value) {
      base_map_stem = argv[++i];
    } else if (arg == "--updater_map" && has_value) {
      updater_map_stem = argv[++i];
    } else if (arg == "--display-graph") {
      display_graph = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "error: unrecognised argument '" << arg << "'\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (base_map_stem.empty() || updater_map_stem.empty()) {
    usage(argv[0]);
    return 2;
  }

  std::cout << "\nDone: parse command-line arguments (base_map=" << base_map_stem
            << ", updater_map=" << updater_map_stem << ")" << std::endl;

  /// Load base_map and updater_map

  std::cout << "\n\n---\n\nStarting: load base_map and updater_map" << std::endl;

  // Deliberately leaked - nothing may run ~Mapper / ~Dataset / UnregisterSensor. See finish().
  auto * base_mapper = new karto::Mapper();
  auto * base_dataset = new karto::Dataset();
  auto * updater_mapper = new karto::Mapper();
  auto * updater_dataset = new karto::Dataset();

  std::cout << "reading " << base_map_stem << ".posegraph / .data ..." << std::endl;
  std::cout << "reading " << updater_map_stem << ".posegraph / .data ..." << std::endl;
  try {
    base_mapper->LoadFromFile(base_map_stem + ".posegraph");
    base_dataset->LoadFromFile(base_map_stem + ".data");
    updater_mapper->LoadFromFile(updater_map_stem + ".posegraph");
    updater_dataset->LoadFromFile(updater_map_stem + ".data");
  } catch (const std::exception & e) {
    std::cerr << "error: failed to read pose graphs: " << e.what() << "\n";
    finish(1);
  }

  std::cout << "\nDone: load base_map and updater_map" << std::endl;

  /// Register lasers

  std::cout << "\n\n---\n\nStarting: register lasers" << std::endl;

  std::set<std::string> registered_names;
  const size_t base_map_lasers = registerLasers(base_dataset, registered_names);
  const size_t updater_map_lasers = registerLasers(updater_dataset, registered_names);
  if (base_map_lasers == 0) {
    std::cerr << "error: no LaserRangeFinder in " << base_map_stem << ".data\n";
    finish(1);
  }
  std::cout << "  " << base_map_lasers << " laser(s) from base_map, " << updater_map_lasers
            << " new laser(s) from updater_map" << std::endl;

  std::cout << "\nDone: register lasers" << std::endl;

  /// Count nodes in each input graph

  std::cout << "\n\n---\n\nStarting: count nodes in each input graph" << std::endl;

  Stats stats;
  for (const auto & by_sensor : base_mapper->GetGraph()->GetVertices()) {
    for (const auto & entry : by_sensor.second) {
      if (entry.second != nullptr && entry.second->GetObject() != nullptr) {
        ++stats.nodes_from_base_map;
      }
    }
  }
  for (const auto & by_sensor : updater_mapper->GetGraph()->GetVertices()) {
    for (const auto & entry : by_sensor.second) {
      if (entry.second != nullptr && entry.second->GetObject() != nullptr) {
        ++stats.nodes_from_updater_map;
      }
    }
  }
  std::cout << "base_map: " << stats.nodes_from_base_map << " nodes, updater_map: "
            << stats.nodes_from_updater_map << " nodes" << std::endl;

  std::cout << "\nDone: count nodes in each input graph" << std::endl;

  /// Create the fused pose graph

  // TODO (AdityaPatil): Currently both the graphs can be modified, we ideally should constrain the base posegraph

  std::cout << "\n\n---\n\nStarting: create the fused pose graph" << std::endl;

  // Neither base_map nor updater_map is mutated: base_mapper/updater_mapper stay exactly as
  // loaded. Everything from both graphs is combined into this fresh, independent third Mapper
  // instead.
  auto * fused_mapper = new karto::Mapper();
  auto * fused_dataset = new karto::Dataset();

  // A freshly-constructed Mapper doesn't allocate its graph/sensor-manager/scan-matcher until
  // Initialize() runs - normally triggered lazily by a Mapper's own first Process() call. The
  // sensor-registration loop right below needs GetMapperSensorManager() to already be non-null,
  // so call it explicitly here first, exactly as Process() would (same range threshold, taken
  // from the primary registered laser).
  auto * primary_laser = dynamic_cast<karto::LaserRangeFinder *>(base_dataset->GetLasers()[0]);
  fused_mapper->Initialize(primary_laser->GetRangeThreshold());

  std::cout << "\nDone: create the fused pose graph" << std::endl;

  /// Register sensor names on the fused graph

  std::cout << "\n\n---\n\nStarting: register sensor names on the fused graph" << std::endl;

  // Register every sensor name from BOTH graphs before either is replayed - updater_map may use
  // a name base_map never did, and it has to be known before its first Process() call.
  std::set<std::string> sensor_names_seen;
  for (auto * source : {base_mapper, updater_mapper}) {
    for (const auto & by_sensor : source->GetGraph()->GetVertices()) {
      for (const auto & entry : by_sensor.second) {
        if (entry.second == nullptr || entry.second->GetObject() == nullptr) {
          continue;
        }
        const karto::Name & name = entry.second->GetObject()->GetSensorName();
        if (sensor_names_seen.insert(name.ToString()).second) {
          fused_mapper->GetMapperSensorManager()->RegisterSensor(name);
        }
      }
    }
  }

  std::cout << "\nDone: register sensor names on the fused graph" << std::endl;

  /// Configure scan-matching / loop-closure parameters

  std::cout << "\n\n---\n\nStarting: configure scan-matching / loop-closure parameters" << std::endl;

  // Mapper's own scan-matching/loop-closure tuning otherwise defaults to karto's stock
  // library values, not necessarily this robot's actual production tuning. Apply the same
  // values the live node runs with, from
  // ppmt_robot_driver/config/slam_tb_online_async_params_default.yaml, so this offline replay
  // behaves like live mapping - with one deliberate exception, LoopMatchMinimumChainSize below.
  fused_mapper->setParamUseScanMatching(true);
  fused_mapper->setParamUseScanBarycenter(true);
  fused_mapper->setParamMinimumTimeInterval(0.5);
  fused_mapper->setParamMinimumTravelDistance(0.5);
  fused_mapper->setParamMinimumTravelHeading(0.5);
  fused_mapper->setParamScanBufferSize(10);
  fused_mapper->setParamScanBufferMaximumScanDistance(15.0);
  fused_mapper->setParamLinkMatchMinimumResponseFine(0.1);
  fused_mapper->setParamLinkScanMaximumDistance(1.5);
  fused_mapper->setParamLoopSearchMaximumDistance(3.0);
  fused_mapper->setParamDoLoopClosing(true);
  // Production value is 10, tuned for LIVE continuous mapping where a false-positive loop
  // closure is the main risk to guard against. Here, a scan's running-scan buffer only ever
  // holds a handful of scans from the OTHER graph (seeded once by ProcessAgainstNodesNearBy for
  // each graph's first scan, then evicted within ~scan_buffer_size replayed scans) - the ONLY
  // other path that can ever tie the two graphs together is TryCloseLoop's graph-wide search,
  // and that requires a *contiguous* run of this many nearby scans before it will even consider
  // a candidate (FindPossibleLoopClosure, Mapper.cpp:2001). If the two graphs only share a thin
  // border strip, one may never have 10 scans within LoopSearchMaximumDistance of any single
  // point on the other's path, so the production threshold structurally forbids any closure
  // regardless of match quality. Lowered here so a real but small overlap still has a chance;
  // the coarse/fine response and variance thresholds above remain the actual quality gate
  // against false positives.
  fused_mapper->setParamLoopMatchMinimumChainSize(3);
  fused_mapper->setParamLoopMatchMaximumVarianceCoarse(3.0);
  fused_mapper->setParamLoopMatchMinimumResponseCoarse(0.35);
  fused_mapper->setParamLoopMatchMinimumResponseFine(0.45);
  fused_mapper->setParamCorrelationSearchSpaceDimension(0.5);
  fused_mapper->setParamCorrelationSearchSpaceResolution(0.01);
  fused_mapper->setParamCorrelationSearchSpaceSmearDeviation(0.1);
  fused_mapper->setParamLoopSearchSpaceDimension(8.0);
  fused_mapper->setParamLoopSearchSpaceResolution(0.05);
  fused_mapper->setParamLoopSearchSpaceSmearDeviation(0.03);
  fused_mapper->setParamDistanceVariancePenalty(0.5);
  fused_mapper->setParamAngleVariancePenalty(1.0);
  fused_mapper->setParamFineSearchAngleOffset(0.00349);
  fused_mapper->setParamCoarseSearchAngleOffset(0.349);
  fused_mapper->setParamCoarseAngleResolution(0.0349);
  fused_mapper->setParamMinimumAnglePenalty(0.9);
  fused_mapper->setParamMinimumDistancePenalty(0.5);
  fused_mapper->setParamUseResponseExpansion(true);

  std::cout << "  loop closure params (production tuning): do_loop_closing="
            << (fused_mapper->getParamDoLoopClosing() ? "true" : "false")
            << " search_max_distance=" << fused_mapper->getParamLoopSearchMaximumDistance()
            << "m min_chain_size=" << fused_mapper->getParamLoopMatchMinimumChainSize()
            << " min_response_coarse=" << fused_mapper->getParamLoopMatchMinimumResponseCoarse()
            << " max_variance_coarse=" << fused_mapper->getParamLoopMatchMaximumVarianceCoarse()
            << " min_response_fine=" << fused_mapper->getParamLoopMatchMinimumResponseFine()
            << std::endl;

  std::cout << "\nDone: configure scan-matching / loop-closure parameters" << std::endl;

  /// Replay both graphs through the fused graph's own scan matcher/solver

  std::cout << "\n\n---\n\nStarting: replay both graphs through the fused graph's own scan matcher/solver"
            << std::endl;

  std::set<karto::LocalizedRangeScan *> base_map_scan_set;
  std::set<karto::LocalizedRangeScan *> updater_map_scan_set;
  std::map<int, karto::Pose2> pre_correct_pose;
  int updater_map_first_uid = -1;

  rclcpp::init(0, nullptr);
  {
    // ceres_*/mode are declared INSIDE CeresSolver::Configure() itself (with its own hardcoded
    // defaults) - declaring them again here would throw ParameterAlreadyDeclaredException, so
    // to use the production values instead of Configure()'s defaults they have to go in as
    // parameter overrides, which declare_parameter picks up in place of its own default.
    rclcpp::NodeOptions node_options;
    node_options.parameter_overrides(
    {
      rclcpp::Parameter("ceres_linear_solver", "SPARSE_NORMAL_CHOLESKY"),
      rclcpp::Parameter("ceres_preconditioner", "SCHUR_JACOBI"),
      rclcpp::Parameter("ceres_trust_strategy", "LEVENBERG_MARQUARDT"),
      rclcpp::Parameter("ceres_dogleg_type", "TRADITIONAL_DOGLEG"),
      rclcpp::Parameter("ceres_loss_function", "HuberLoss"),
      rclcpp::Parameter("mode", "mapping"),
    });
    auto node = std::make_shared<rclcpp::Node>("posegraph_merge", node_options);
    // debug_logging is read with get_parameter (not declared by Configure()), which throws on
    // an undeclared name, so it must be pre-declared here.
    node->declare_parameter("debug_logging", false);

    auto solver = std::make_unique<solver_plugins::CeresSolver>();
    solver->Configure(node);
    solver->Reset();
    // Set BEFORE either replay call: MapperGraph::AddVertex/LinkScans call
    // m_pScanOptimizer->AddNode()/AddConstraint() as a side effect of every Process()/
    // ProcessAgainstNodesNearBy() call, so the solver accumulates both graphs' structure
    // automatically - no separate seeding pass needed, unlike when only one side was spliced in.
    fused_mapper->SetScanSolver(solver.get());

    // Both graphs replayed the same way, in turn: base_map first, then updater_map. Neither is
    // trusted as a fixed "base" - both get scan-matched and both get an equal chance to
    // loop-close against whatever the other call already put into fused_mapper.
    replayGraph(
      base_mapper, fused_mapper, fused_dataset, stats, stats.base_map_merged,
      stats.base_map_dropped, base_map_scan_set, pre_correct_pose);
    replayGraph(
      updater_mapper, fused_mapper, fused_dataset, stats, stats.updater_map_merged,
      stats.updater_map_dropped, updater_map_scan_set, pre_correct_pose,
      &updater_map_first_uid);

    // Process()/ProcessAgainstNodesNearBy() only place each scan against its *local*
    // neighbourhood (the sequential/loop scan matchers); the global Ceres solve that
    // reconciles the whole graph only runs inside TryCloseLoop, and only when a candidate
    // loop closure clears its coarse+fine response/variance thresholds. If that never fires
    // during replay, AddNode/AddConstraint have built up structure that's never actually
    // solved or applied - the emitted map would then just reflect local placement, which is
    // exactly the kind of residual offset a misaligned pair of input frames would leave
    // behind. Force one global solve + pose-correction pass unconditionally, regardless of
    // whether any loop closure happened to fire along the way.
    fused_mapper->CorrectPoses();

    for (auto * scan : fused_mapper->GetAllProcessedScans()) {
      const auto it = pre_correct_pose.find(scan->GetUniqueId());
      if (it == pre_correct_pose.end()) {
        continue;
      }
      const double dx = scan->GetCorrectedPose().GetX() - it->second.GetX();
      const double dy = scan->GetCorrectedPose().GetY() - it->second.GetY();
      const double shift = std::sqrt(dx * dx + dy * dy);
      if (updater_map_scan_set.count(scan) != 0) {
        stats.updater_map_shift_max = std::max(stats.updater_map_shift_max, shift);
        stats.updater_map_shift_sum += shift;
      } else {
        if (shift > 1e-6) {
          ++stats.base_map_scans_shifted;
        }
        stats.base_map_shift_max = std::max(stats.base_map_shift_max, shift);
        stats.base_map_shift_sum += shift;
      }
    }

    // solver, node, sensor manager registrations are intentionally left alive - see finish().
    solver.release();
  }

  std::cout << "\nDone: replay both graphs through the fused graph's own scan matcher/solver"
            << std::endl;

  /// Log edges that tie base_map to updater_map

  if(display_graph)
  {
    std::cout << "\n\n---\n\nStarting: log edges that tie base_map to updater_map" << std::endl;
    
    // An edge with one endpoint in each source is either a genuine TryCloseLoop() loop closure or
    // (exactly once, for updater_map's very first replayed scan) the ordinary "link to previous
    // scan" edge AddEdges() creates because that scan's fused-graph StateId happens to follow
    // directly after base_map's last one - see replayGraph's docs. There's no per-edge "how was
    // this created" flag in karto to tell the two apart directly, so the one known non-loop-closure
    // edge is identified by checking whether it touches updater_map_first_uid; every other
    // cross-source edge is a real loop closure.
    size_t cross_edge_count = 0;
    for (auto * edge : fused_mapper->GetGraph()->GetEdges()) {
      if (edge == nullptr || edge->GetSource() == nullptr || edge->GetTarget() == nullptr) {
        continue;
      }
      auto * source = edge->GetSource()->GetObject();
      auto * target = edge->GetTarget()->GetObject();
      if (source == nullptr || target == nullptr) {
        continue;
      }
      const bool source_is_updater = updater_map_scan_set.count(source) != 0;
      const bool target_is_updater = updater_map_scan_set.count(target) != 0;
      if (source_is_updater == target_is_updater) {
        continue;  // both endpoints from the same source - not a cross-graph edge
      }
      ++cross_edge_count;
      const int base_uid = source_is_updater ? target->GetUniqueId() : source->GetUniqueId();
      const int updater_uid = source_is_updater ? source->GetUniqueId() : target->GetUniqueId();
      const bool is_known_seam_edge = updater_uid == updater_map_first_uid;
      std::cout << "  base_map node " << base_uid << " <-> updater_map node " << updater_uid
      << (is_known_seam_edge ?
        " (sequential link - updater_map's first replayed scan, not a loop closure)" :
        " (loop closure)") << std::endl;
      }
      if (cross_edge_count == 0) {
        std::cout << "  none - base_map and updater_map never linked" << std::endl;
      }
      
      std::cout << "\nDone: log edges that tie base_map to updater_map (" << cross_edge_count
      << " found)" << std::endl;
  }
      
  /// Print merge results

  std::cout << "\n\n---\n\nStarting: print merge results" << std::endl;

  const size_t total_merged = stats.base_map_merged + stats.updater_map_merged;

  std::cout << "\nresult\n"
            << "  base_map nodes merged/dropped    : " << stats.base_map_merged << " / "
            << stats.base_map_dropped << "\n"
            << "  updater_map nodes merged/dropped : " << stats.updater_map_merged << " / "
            << stats.updater_map_dropped << std::endl;

  if (total_merged > 0) {
    const double cov_trace_mean = stats.cov_trace_sum / static_cast<double>(total_merged);
    const double updater_map_shift_mean = stats.updater_map_merged > 0 ?
      stats.updater_map_shift_sum / static_cast<double>(stats.updater_map_merged) : 0.0;
    const double base_map_shift_mean = stats.base_map_merged > 0 ?
      stats.base_map_shift_sum / static_cast<double>(stats.base_map_merged) : 0.0;
    std::cout << "\nmerge diagnostics (local-match confidence, then how far the global solve\n"
                 "moved things from that local placement - large numbers mean the matches\n"
                 "were poor and the graphs are still misaligned)\n"
              << "  local match covariance trace (x+y variance, m^2), min/mean/max   : "
              << stats.cov_trace_min << " / " << cov_trace_mean << " / "
              << stats.cov_trace_max << "\n"
              << "  updater_map nodes shifted by CorrectPoses(), mean/max (m)        : "
              << updater_map_shift_mean << " / " << stats.updater_map_shift_max << "\n"
              << "  base_map nodes moved at all                                     : "
              << stats.base_map_scans_shifted << "\n"
              << "  base_map nodes shifted by CorrectPoses(), mean/max (m)           : "
              << base_map_shift_mean << " / " << stats.base_map_shift_max << std::endl;
  }

  std::cout << "\nDone: print merge results" << std::endl;

  const std::string out_stem = "merged";


  /// Saves the merged posegraph


  // try {
  //   fused_mapper->SaveToFile(out_stem + ".posegraph");
  //   fused_dataset->SaveToFile(out_stem + ".data");
  // } catch (const std::exception & e) {
  //   std::cerr << "error: failed to write merged pose graph: " << e.what() << "\n";
  //   finish(1);
  // }
  // std::cout << "wrote " << out_stem << ".posegraph / .data" << std::endl;


  /// Saves the 2D merged map

  std::cout << "\n\n---\n\nStarting: save the 2D merged map" << std::endl;

  std::string err;
  if (!saveMapImage(fused_mapper, display_graph, updater_map_scan_set, 0.05, out_stem, err)) {
    std::cerr << "error: failed to write merged map: " << err << "\n";
    finish(1);
  }
  std::cout << "wrote " << out_stem << ".png / .yaml"
            << (display_graph ? " (with graph overlay)" : "") << std::endl;

  std::cout << "\nDone: save the 2D merged map" << std::endl;

  finish(0);
}
