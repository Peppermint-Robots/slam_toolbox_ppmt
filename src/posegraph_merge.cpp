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
  size_t scans_from_base_map = 0;
  size_t scans_from_updater_map = 0;
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

/** Write <stem>.png + <stem>.yaml, in the classification convention this fork's other offline
 * pose-graph tools already use: free=254, occupied=0, unknown=205, negate=0, default thresholds.
 * Karto's OccupancyGrid stores row 0 at the WORLD-MINIMUM y, but a PNG/cv::Mat row 0 is the TOP
 * of the image (world-maximum y), so row order is flipped on the way out.
 */
bool saveMapImage(
  const karto::LocalizedRangeScanVector & scans, double resolution,
  const std::string & stem, std::string & err)
{
  std::unique_ptr<karto::OccupancyGrid> occ_grid(
    karto::OccupancyGrid::CreateFromScans(scans, resolution));
  if (!occ_grid) {
    err = "no scans to build a map from";
    return false;
  }

  const int width = occ_grid->GetWidth();
  const int height = occ_grid->GetHeight();
  const karto::Vector2<kt_double> offset = occ_grid->GetCoordinateConverter()->GetOffset();

  cv::Mat image(height, width, CV_8UC1, cv::Scalar(205));
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
      image.at<uint8_t>(row, x) = pixel;
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

/** Draw the fused pose graph on top of its own occupancy grid: every vertex and edge colored by
 * a JET gradient (blue = earliest, red = latest) keyed on the scan's UniqueId in fused_mapper -
 * i.e. the order it was replayed in, not which of base_map/updater_map it came from.
 * fused_mapper is visualized as a single graph, not as two merged sources; an edge's color is
 * its two endpoints' midpoint id. Each vertex is labeled with its own UniqueId, and each edge
 * with "sourceId-targetId", so a specific node/link can be picked out by eye. Reuses the exact
 * same world-to-pixel convention as saveMapImage so the overlay lines up with it pixel-for-pixel.
 */
bool saveGraphOverlay(karto::Mapper * mapper, double resolution, const std::string & stem, std::string & err)
{
  const karto::LocalizedRangeScanVector all_scans = mapper->GetAllProcessedScans();
  std::unique_ptr<karto::OccupancyGrid> occ_grid(
    karto::OccupancyGrid::CreateFromScans(all_scans, resolution));
  if (!occ_grid) {
    err = "no scans to build a graph overlay from";
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
  cv::cvtColor(gray, image, cv::COLOR_GRAY2BGR);

  auto toPixel = [&](const karto::Pose2 & pose) {
      const int gx = static_cast<int>(std::lround((pose.GetX() - offset.GetX()) / resolution));
      const int gy = static_cast<int>(std::lround((pose.GetY() - offset.GetY()) / resolution));
      return cv::Point(gx, height - 1 - gy);
    };

  // Order gradient: fused_mapper's own UniqueId is assigned sequentially by
  // MapperSensorManager::AddScan as each scan is replayed in (base_map's, then updater_map's) -
  // so it's exactly "insertion order into the fused graph". Map that to a 256-entry JET lookup
  // table (blue = earliest, red = latest) computed once, rather than calling applyColorMap per
  // pixel/shape.
  int min_uid = std::numeric_limits<int>::max();
  int max_uid = std::numeric_limits<int>::min();
  for (auto * scan : all_scans) {
    min_uid = std::min(min_uid, scan->GetUniqueId());
    max_uid = std::max(max_uid, scan->GetUniqueId());
  }
  const int uid_span = std::max(max_uid - min_uid, 1);

  cv::Mat gray_lut(256, 1, CV_8UC1);
  for (int i = 0; i < 256; ++i) {
    gray_lut.at<uint8_t>(i, 0) = static_cast<uint8_t>(i);
  }
  cv::Mat colormap;
  cv::applyColorMap(gray_lut, colormap, cv::COLORMAP_JET);
  auto colorForUniqueId = [&](int uid) {
      const double t = static_cast<double>(uid - min_uid) / static_cast<double>(uid_span);
      const int index = std::clamp(static_cast<int>(std::lround(t * 255.0)), 0, 255);
      return colormap.at<cv::Vec3b>(index, 0);
    };

  auto drawLabel = [&](const cv::Point & anchor, const std::string & text) {
      cv::putText(
        image, text, anchor + cv::Point(4, -4), cv::FONT_HERSHEY_SIMPLEX, 0.20,
        cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    };

  for (auto * edge : mapper->GetGraph()->GetEdges()) {
    if (edge == nullptr || edge->GetSource() == nullptr || edge->GetTarget() == nullptr) {
      continue;
    }
    auto * source = edge->GetSource()->GetObject();
    auto * target = edge->GetTarget()->GetObject();
    if (source == nullptr || target == nullptr) {
      continue;
    }
    const int mid_uid = (source->GetUniqueId() + target->GetUniqueId()) / 2;
    const cv::Vec3b color = colorForUniqueId(mid_uid);
    const cv::Point source_px = toPixel(source->GetCorrectedPose());
    const cv::Point target_px = toPixel(target->GetCorrectedPose());
    cv::line(
      image, source_px, target_px, cv::Scalar(color[0], color[1], color[2]), 1, cv::LINE_AA);
    // drawLabel(
    //   (source_px + target_px) / 2,
    //   std::to_string(source->GetUniqueId()) + "-" + std::to_string(target->GetUniqueId()));
  }

  for (const auto & by_sensor : mapper->GetGraph()->GetVertices()) {
    for (const auto & entry : by_sensor.second) {
      if (entry.second == nullptr || entry.second->GetObject() == nullptr) {
        continue;
      }
      auto * scan = entry.second->GetObject();
      const cv::Vec3b color = colorForUniqueId(scan->GetUniqueId());
      const cv::Point px = toPixel(scan->GetCorrectedPose());
      cv::circle(image, px, 1, cv::Scalar(color[0], color[1], color[2]), cv::FILLED, cv::LINE_AA);
      drawLabel(px, std::to_string(scan->GetUniqueId()));
    }
  }

  const std::string path = stem + "_graph.png";
  if (!cv::imwrite(path, image)) {
    err = "failed to write '" + path + "'";
    return false;
  }
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
 */
void replayGraph(
  karto::Mapper * source, karto::Mapper * fused, karto::Dataset * fused_dataset, Stats & stats,
  size_t & merged_count, size_t & dropped_count,
  std::set<karto::LocalizedRangeScan *> & accepted_scans,
  std::map<int, karto::Pose2> & pre_correct_pose)
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
    " --base_map <stem> --updater_map <stem>\n\n"
    "Stems carry no extension: <stem>.posegraph and <stem>.data are both read.\n"
    "base_map and updater_map are assumed to already be expressed in the same\n"
    "  coordinate frame - this tool performs no alignment.\n"
    "base_map's scans are replayed into the fused graph first, then updater_map's -\n"
    "  both through the fused graph's own scan matcher/solver, so overlapping content\n"
    "  reconciles via loop closure instead of being drawn twice, and both graphs get an\n"
    "  equal chance to link into whatever is already there.\n\n"
    "Output is always written in the current working directory, as merged.posegraph/\n"
    "  .data (the fused graph), merged.png/.yaml (its occupancy grid), and\n"
    "  merged_graph.png (its vertices/edges overlaid on that grid).\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string base_map_stem;
  std::string updater_map_stem;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = (i + 1) < argc;
    if (arg == "--base_map" && has_value) {
      base_map_stem = argv[++i];
    } else if (arg == "--updater_map" && has_value) {
      updater_map_stem = argv[++i];
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

  std::set<std::string> registered_names;
  const size_t base_map_lasers = registerLasers(base_dataset, registered_names);
  const size_t updater_map_lasers = registerLasers(updater_dataset, registered_names);
  if (base_map_lasers == 0) {
    std::cerr << "error: no LaserRangeFinder in " << base_map_stem << ".data\n";
    finish(1);
  }
  std::cout << "  " << base_map_lasers << " laser(s) from base_map, " << updater_map_lasers
            << " new laser(s) from updater_map" << std::endl;

  Stats stats;
  for (const auto & by_sensor : base_mapper->GetGraph()->GetVertices()) {
    for (const auto & entry : by_sensor.second) {
      if (entry.second != nullptr && entry.second->GetObject() != nullptr) {
        ++stats.scans_from_base_map;
      }
    }
  }
  for (const auto & by_sensor : updater_mapper->GetGraph()->GetVertices()) {
    for (const auto & entry : by_sensor.second) {
      if (entry.second != nullptr && entry.second->GetObject() != nullptr) {
        ++stats.scans_from_updater_map;
      }
    }
  }
  std::cout << "base_map: " << stats.scans_from_base_map << " scans, updater_map: "
            << stats.scans_from_updater_map << " scans" << std::endl;

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

  std::set<karto::LocalizedRangeScan *> base_map_scan_set;
  std::set<karto::LocalizedRangeScan *> updater_map_scan_set;
  std::map<int, karto::Pose2> pre_correct_pose;

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
      stats.updater_map_dropped, updater_map_scan_set, pre_correct_pose);

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

  const karto::LocalizedRangeScanVector final_scans = fused_mapper->GetAllProcessedScans();
  const size_t total_merged = stats.base_map_merged + stats.updater_map_merged;

  std::cout << "\nresult\n"
            << "  base_map scans merged/dropped    : " << stats.base_map_merged << " / "
            << stats.base_map_dropped << "\n"
            << "  updater_map scans merged/dropped : " << stats.updater_map_merged << " / "
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
              << "  updater_map scans shifted by CorrectPoses(), mean/max (m)        : "
              << updater_map_shift_mean << " / " << stats.updater_map_shift_max << "\n"
              << "  base_map scans moved at all                                     : "
              << stats.base_map_scans_shifted << "\n"
              << "  base_map scans shifted by CorrectPoses(), mean/max (m)           : "
              << base_map_shift_mean << " / " << stats.base_map_shift_max << std::endl;
  }

  const std::string out_stem = "merged";

  try {
    fused_mapper->SaveToFile(out_stem + ".posegraph");
    fused_dataset->SaveToFile(out_stem + ".data");
  } catch (const std::exception & e) {
    std::cerr << "error: failed to write merged pose graph: " << e.what() << "\n";
    finish(1);
  }
  std::cout << "wrote " << out_stem << ".posegraph / .data" << std::endl;

  std::string err;
  if (!saveMapImage(final_scans, 0.05, out_stem, err)) {
    std::cerr << "error: failed to write merged map: " << err << "\n";
    finish(1);
  }
  std::cout << "wrote " << out_stem << ".png / .yaml" << std::endl;

  if (!saveGraphOverlay(fused_mapper, 0.05, out_stem, err)) {
    std::cerr << "error: failed to write graph overlay: " << err << "\n";
    finish(1);
  }
  std::cout << "wrote " << out_stem << "_graph.png" << std::endl;

  finish(0);
}
