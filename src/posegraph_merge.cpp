// Offline merge of two serialized slam_toolbox pose-graphs into one occupancy grid.
//
// Loads posegraph1 (base) and posegraph2 (source), applies a manually-supplied rigid transform
// to every posegraph2 scan to align it into posegraph1's frame, then rasterizes the combined
// scans into <in1-stem>_merged_aligned.pgm/.yaml or <in1-stem>_merged_unaligned.pgm/.yaml,
// depending on --align. This tool never re-serializes a merged pose graph -
// the occupancy grid it emits is a one-shot snapshot, not a persisted graph.
//
// Two mechanisms, chosen by --align:
//
//  - default: posegraph1 is untouched and no solver is ever created. Every (transformed)
//    posegraph2 scan is simply concatenated with posegraph1's own scans before rasterizing -
//    the same thing merge_maps_kinematic.cpp's interactive tool already does, minus the ROS
//    node/marker-drag machinery. Content that both graphs cover independently will be drawn
//    twice (no reconciliation).
//
//  - --align: posegraph2's (transformed) scans are replayed one at a time through posegraph1's
//    own Mapper::Process()/ProcessAgainstNodesNearBy(), backed by a real CeresSolver, exactly as
//    DecentralizedMultiRobotSlamToolbox::addExternalScan() does for live multi-robot merging
//    (src/slam_toolbox_decentralized_multirobot.cpp on the ros2 branch). Overlapping regions
//    loop-close via the real scan matcher instead of being drawn twice. This is an in-memory
//    solve only - nothing is written back to posegraph1's files - but posegraph1's own
//    non-anchor scan poses CAN shift in the emitted image wherever real evidence ties them to
//    posegraph2's content (CeresSolver only pins the single first-ever-added node constant).
//    That is expected, not a bug: if this merged file were later loaded live by slam_toolbox,
//    loadSerializedPoseGraph() would re-run the same kind of full optimization anyway.
//
// This is an offline CLI tool: --align needs a bare, unspun rclcpp::Node only to Configure() the
// Ceres solver (CeresSolver::Configure reads a handful of ROS parameters); nothing here ever
// spins. Like this fork's other offline pose-graph tools, Mapper/Dataset are heap-allocated and
// deliberately never freed - see finish() for why.

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
#include <sstream>
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
 * abort with "stack smashing detected" on exit, after all real work had already completed. A
 * one-shot batch tool gains nothing from that teardown: the OS reclaims the memory either way,
 * we never serialize mapper1/dataset1 back out, and skipping it means a successful run cannot be
 * turned into a failed exit code by a destructor. _Exit does not flush, so flush first.
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
  size_t scans_from_graph2 = 0;
  size_t scans_merged = 0;
  size_t scans_dropped = 0;

  // --align only: local scan-matcher confidence per merged scan (translational covariance
  // trace, i.e. cov(0,0)+cov(1,1) - smaller is a tighter/more confident match) and how far
  // CorrectPoses()'s global solve then moved things from that local-match placement.
  double cov_trace_min = std::numeric_limits<double>::infinity();
  double cov_trace_max = 0.0;
  double cov_trace_sum = 0.0;

  size_t graph1_scans_shifted = 0;
  double graph1_shift_max = 0.0;
  double graph1_shift_sum = 0.0;

  double graph2_shift_max = 0.0;
  double graph2_shift_sum = 0.0;
};

/** Rigidly transform one scan's pose/geometry fields in place (barycenter, bounding box, point
 * readings, corrected pose, odometric pose), mirroring merge_maps_kinematic.cpp's transformScan
 * - ported to use karto::Transform directly instead of tf2, since this tool has no other need
 * for ROS types. Odometric and corrected pose both get the same correction so their relative
 * relationship (real, internally-consistent odometry from the original posegraph2 recording) is
 * preserved exactly; only their shared global placement changes. That preserved relationship is
 * what lets --align mode's sequential Mapper::Process() calls (scan 2..N of a run) trust
 * odometric continuity between consecutive replayed scans.
 */
void transformScan(karto::LocalizedRangeScan * scan, karto::Transform & correction)
{
  auto transformPoint = [&correction](const karto::Vector2<kt_double> & p) {
      karto::Pose2 transformed = correction.TransformPose(karto::Pose2(p.GetX(), p.GetY(), 0.0));
      return karto::Vector2<kt_double>(transformed.GetX(), transformed.GetY());
    };

  karto::Pose2 barycenter = scan->GetBarycenterPose();
  karto::Pose2 barycenter_corr = correction.TransformPose(barycenter);
  scan->SetBarycenterPose(barycenter_corr);

  const karto::BoundingBox2 & bbox = scan->GetBoundingBox();
  const karto::Vector2<kt_double> min_corr = transformPoint(bbox.GetMinimum());
  const karto::Vector2<kt_double> max_corr = transformPoint(bbox.GetMaximum());
  const karto::Vector2<kt_double> min_right_corr = transformPoint(
    karto::Vector2<kt_double>(bbox.GetMaximum().GetX(), bbox.GetMinimum().GetY()));
  const karto::Vector2<kt_double> max_left_corr = transformPoint(
    karto::Vector2<kt_double>(bbox.GetMinimum().GetX(), bbox.GetMaximum().GetY()));
  karto::BoundingBox2 transformed_bbox;
  transformed_bbox.Add(min_corr);
  transformed_bbox.Add(max_corr);
  transformed_bbox.Add(min_right_corr);
  transformed_bbox.Add(max_left_corr);
  scan->SetBoundingBox(transformed_bbox);

  karto::PointVectorDouble points = scan->GetPointReadings();
  for (auto & point : points) {
    const karto::Vector2<kt_double> corrected = transformPoint(point);
    point.SetX(corrected.GetX());
    point.SetY(corrected.GetY());
  }
  scan->SetPointReadings(points);

  scan->SetCorrectedPose(correction.TransformPose(scan->GetCorrectedPose()));
  scan->SetOdometricPose(correction.TransformPose(scan->GetOdometricPose()));

  kt_bool dirty = true;
  scan->SetIsDirty(dirty);
  scan->GetPointReadings(false);
}

bool parseTransform(const std::string & spec, double & x, double & y, double & theta)
{
  std::stringstream ss(spec);
  std::string token;
  std::vector<double> v;
  while (std::getline(ss, token, ',')) {
    try {
      v.push_back(std::stod(token));
    } catch (const std::exception &) {
      return false;
    }
  }
  if (v.size() != 3) {
    return false;
  }
  x = v[0];
  y = v[1];
  theta = v[2];
  return true;
}

/** Write <stem>.pgm + <stem>.yaml, in the classification convention this fork's other offline
 * pose-graph tools already use: free=254, occupied=0, unknown=205, negate=0, default thresholds.
 * Karto's OccupancyGrid stores row 0 at the WORLD-MINIMUM y, but a PGM/cv::Mat row 0 is the TOP
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

  const std::string pgm_path = stem + ".pgm";
  const std::string yaml_path = stem + ".yaml";
  if (!cv::imwrite(pgm_path, image)) {
    err = "failed to write '" + pgm_path + "'";
    return false;
  }

  const size_t slash = pgm_path.find_last_of('/');
  const std::string pgm_name = slash == std::string::npos ? pgm_path : pgm_path.substr(slash + 1);

  YAML::Emitter yaml;
  yaml << YAML::BeginMap;
  yaml << YAML::Key << "image" << YAML::Value << pgm_name;
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
 * re-registering under override would silently repoint posegraph1's own already-loaded scans at
 * posegraph2's copy of that laser's config.
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

void usage(const char * argv0)
{
  std::cerr <<
    "Merge posegraph2 into posegraph1 and emit one occupancy grid.\n\n"
    "Usage:\n  " << argv0 <<
    " --in1 <stem> --in2 <stem> --transform x,y,theta [--align]\n\n"
    "Stems carry no extension: <stem>.posegraph and <stem>.data are both read.\n"
    "--transform is the rigid transform (metres/radians) that aligns posegraph2 into\n"
    "  posegraph1's frame - this tool does not compute alignment itself.\n"
    "--align replays posegraph2's (transformed) scans through posegraph1's own scan\n"
    "  matcher/solver instead of just concatenating scans, so overlapping content\n"
    "  reconciles via loop closure instead of being drawn twice. Without it, posegraph1\n"
    "  is never touched and no solver is created.\n\n"
    "Output is always written next to --in1, as <in1-stem>_merged_aligned.pgm/.yaml when\n"
    "  --align is set, or <in1-stem>_merged_unaligned.pgm/.yaml otherwise. This tool\n"
    "never re-serializes a merged pose graph - the occupancy grid is a one-shot snapshot.\n";
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string in1_stem;
  std::string in2_stem;
  std::string transform_spec;
  bool align = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = (i + 1) < argc;
    if (arg == "--in1" && has_value) {
      in1_stem = argv[++i];
    } else if (arg == "--in2" && has_value) {
      in2_stem = argv[++i];
    } else if (arg == "--transform" && has_value) {
      transform_spec = argv[++i];
    } else if (arg == "--align") {
      align = true;
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "error: unrecognised argument '" << arg << "'\n";
      usage(argv[0]);
      return 2;
    }
  }

  double tx = 0.0, ty = 0.0, ttheta = 0.0;
  if (in1_stem.empty() || in2_stem.empty() || !parseTransform(transform_spec, tx, ty, ttheta)) {
    usage(argv[0]);
    return 2;
  }

  // Deliberately leaked - nothing may run ~Mapper / ~Dataset / UnregisterSensor. See finish().
  auto * mapper1 = new karto::Mapper();
  auto * dataset1 = new karto::Dataset();
  auto * mapper2 = new karto::Mapper();
  auto * dataset2 = new karto::Dataset();

  std::cout << "reading " << in1_stem << ".posegraph / .data ..." << std::endl;
  std::cout << "reading " << in2_stem << ".posegraph / .data ..." << std::endl;
  try {
    mapper1->LoadFromFile(in1_stem + ".posegraph");
    dataset1->LoadFromFile(in1_stem + ".data");
    mapper2->LoadFromFile(in2_stem + ".posegraph");
    dataset2->LoadFromFile(in2_stem + ".data");
  } catch (const std::exception & e) {
    std::cerr << "error: failed to read pose graphs: " << e.what() << "\n";
    finish(1);
  }

  std::set<std::string> registered_names;
  const size_t lasers1 = registerLasers(dataset1, registered_names);
  const size_t lasers2 = registerLasers(dataset2, registered_names);
  if (lasers1 == 0) {
    std::cerr << "error: no LaserRangeFinder in " << in1_stem << ".data\n";
    finish(1);
  }
  std::cout << "  " << lasers1 << " laser(s) from posegraph1, " << lasers2
            << " new laser(s) from posegraph2" << std::endl;

  karto::Pose2 transform_pose(tx, ty, ttheta);
  karto::Transform correction(transform_pose);

  karto::LocalizedRangeScanVector scans2 = mapper2->GetAllProcessedScans();
  std::sort(
    scans2.begin(), scans2.end(),
    [](karto::LocalizedRangeScan * a, karto::LocalizedRangeScan * b) {
      const std::string name_a = a->GetSensorName().ToString();
      const std::string name_b = b->GetSensorName().ToString();
      if (name_a != name_b) {
        return name_a < name_b;
      }
      return a->GetStateId() < b->GetStateId();
    });

  for (auto * scan : scans2) {
    transformScan(scan, correction);
  }

  Stats stats;
  stats.scans_from_graph2 = scans2.size();
  std::cout << "posegraph2: " << stats.scans_from_graph2
            << " scans, transform (" << tx << ", " << ty << ", " << ttheta << ")" << std::endl;

  karto::LocalizedRangeScanVector final_scans;

  if (!align) {
    std::cout << "mode: concatenate (no solver, posegraph1 untouched)" << std::endl;
    final_scans = mapper1->GetAllProcessedScans();
    final_scans.insert(final_scans.end(), scans2.begin(), scans2.end());
    stats.scans_merged = scans2.size();
  } else {
    std::cout << "mode: align (replaying through posegraph1's Mapper::Process)" << std::endl;

    rclcpp::init(0, nullptr);
    {
      auto node = std::make_shared<rclcpp::Node>("posegraph_merge");
      // Configure() declares every ceres_*/mode parameter itself (with sane defaults) via
      // declare_parameter - only debug_logging is read with get_parameter, which throws on an
      // undeclared name, so it must be pre-declared here.
      node->declare_parameter("debug_logging", true);

      auto solver = std::make_unique<solver_plugins::CeresSolver>();
      solver->Configure(node);

      solver->Reset();
      for (const auto & by_sensor : mapper1->GetGraph()->GetVertices()) {
        for (const auto & entry : by_sensor.second) {
          if (entry.second != nullptr) {
            solver->AddNode(entry.second);
          }
        }
      }
      for (auto * edge : mapper1->GetGraph()->GetEdges()) {
        if (edge != nullptr) {
          solver->AddConstraint(edge);
        }
      }
      mapper1->SetScanSolver(solver.get());

      std::set<std::string> sensor_names_seen;
      for (auto * scan : scans2) {
        const std::string name = scan->GetSensorName().ToString();
        if (sensor_names_seen.insert(name).second) {
          mapper1->GetMapperSensorManager()->RegisterSensor(scan->GetSensorName());
        }
      }

      // Snapshot posegraph1's own corrected poses before touching anything, and each
      // posegraph2 scan's corrected pose right after its *local* scan-matcher placement (i.e.
      // before the global solve below), so we can report how far CorrectPoses() moves each
      // group - that's the actual signal for "did the align do anything."
      std::map<int, karto::Pose2> pre_correct_pose;
      for (auto * scan : mapper1->GetAllProcessedScans()) {
        pre_correct_pose[scan->GetUniqueId()] = scan->GetCorrectedPose();
      }

      bool first = true;
      for (auto * scan : scans2) {
        karto::Matrix3 covariance;
        covariance.SetToIdentity();
        const bool ok = first ?
          mapper1->ProcessAgainstNodesNearBy(scan, false, &covariance) :
          mapper1->Process(scan, &covariance);
        first = false;

        if (ok) {
          ++stats.scans_merged;
          const double trace = covariance(0, 0) + covariance(1, 1);
          stats.cov_trace_min = std::min(stats.cov_trace_min, trace);
          stats.cov_trace_max = std::max(stats.cov_trace_max, trace);
          stats.cov_trace_sum += trace;
          pre_correct_pose[scan->GetUniqueId()] = scan->GetCorrectedPose();
        } else {
          ++stats.scans_dropped;
          delete scan;
        }
      }

      // Process()/ProcessAgainstNodesNearBy() only place each scan against its *local*
      // neighbourhood (the sequential/loop scan matchers); the global Ceres solve that
      // reconciles the whole graph only runs inside TryCloseLoop, and only when a candidate
      // loop closure clears its coarse+fine response/variance thresholds. If that never fires
      // during replay, AddNode/AddConstraint have built up structure that's never actually
      // solved or applied - the emitted map would then just reflect local placement, which is
      // exactly the kind of residual offset an imprecise --transform would leave behind. Force
      // one global solve + pose-correction pass unconditionally, regardless of whether any
      // loop closure happened to fire along the way.
      mapper1->CorrectPoses();

      const std::set<karto::LocalizedRangeScan *> graph2_scan_set(scans2.begin(), scans2.end());
      for (auto * scan : mapper1->GetAllProcessedScans()) {
        const auto it = pre_correct_pose.find(scan->GetUniqueId());
        if (it == pre_correct_pose.end()) {
          continue;
        }
        const double dx = scan->GetCorrectedPose().GetX() - it->second.GetX();
        const double dy = scan->GetCorrectedPose().GetY() - it->second.GetY();
        const double shift = std::sqrt(dx * dx + dy * dy);
        if (graph2_scan_set.count(scan) != 0) {
          stats.graph2_shift_max = std::max(stats.graph2_shift_max, shift);
          stats.graph2_shift_sum += shift;
        } else {
          if (shift > 1e-6) {
            ++stats.graph1_scans_shifted;
          }
          stats.graph1_shift_max = std::max(stats.graph1_shift_max, shift);
          stats.graph1_shift_sum += shift;
        }
      }

      // solver, node, sensor manager registrations are intentionally left alive - see finish().
      solver.release();
    }

    final_scans = mapper1->GetAllProcessedScans();
  }

  std::cout << "\nresult\n"
            << "  scans from posegraph2 : " << stats.scans_from_graph2 << "\n"
            << "  scans merged          : " << stats.scans_merged << "\n"
            << "  scans dropped         : " << stats.scans_dropped << std::endl;

  if (align && stats.scans_merged > 0) {
    const double cov_trace_mean = stats.cov_trace_sum / static_cast<double>(stats.scans_merged);
    const double graph2_shift_mean =
      stats.graph2_shift_sum / static_cast<double>(stats.scans_merged);
    const size_t graph1_scan_count = final_scans.size() - stats.scans_merged;
    const double graph1_shift_mean = graph1_scan_count > 0 ?
      stats.graph1_shift_sum / static_cast<double>(graph1_scan_count) : 0.0;
    std::cout << "\nalign diagnostics (local-match confidence, then how far the global solve\n"
                 "moved things from that local placement - large numbers mean the manual\n"
                 "--transform and/or matches were poor and the graphs are still misaligned)\n"
              << "  local match covariance trace (x+y variance, m^2), min/mean/max : "
              << stats.cov_trace_min << " / " << cov_trace_mean << " / "
              << stats.cov_trace_max << "\n"
              << "  posegraph2 scans shifted by CorrectPoses(), mean/max (m)       : "
              << graph2_shift_mean << " / " << stats.graph2_shift_max << "\n"
              << "  posegraph1 scans moved at all                                  : "
              << stats.graph1_scans_shifted << "\n"
              << "  posegraph1 scans shifted by CorrectPoses(), mean/max (m)       : "
              << graph1_shift_mean << " / " << stats.graph1_shift_max << std::endl;
  }

  const std::string out_stem = in1_stem + "_merged" + (align ? "_aligned" : "_unaligned");
  std::string err;
  if (!saveMapImage(final_scans, 0.05, out_stem, err)) {
    std::cerr << "error: failed to write merged map: " << err << "\n";
    finish(1);
  }
  std::cout << "wrote " << out_stem << ".pgm / .yaml" << std::endl;

  finish(0);
}
