/*
 * Copyright (c) 2018, Simbe Robotics, Inc.
 *
 * THE WORK (AS DEFINED BELOW) IS PROVIDED UNDER THE TERMS OF THIS CREATIVE
 * COMMONS PUBLIC LICENSE ("CCPL" OR "LICENSE"). THE WORK IS PROTECTED BY
 * COPYRIGHT AND/OR OTHER APPLICABLE LAW. ANY USE OF THE WORK OTHER THAN AS
 * AUTHORIZED UNDER THIS LICENSE OR COPYRIGHT LAW IS PROHIBITED.
 *
 * BY EXERCISING ANY RIGHTS TO THE WORK PROVIDED HERE, YOU ACCEPT AND AGREE TO
 * BE BOUND BY THE TERMS OF THIS LICENSE. THE LICENSOR GRANTS YOU THE RIGHTS
 * CONTAINED HERE IN CONSIDERATION OF YOUR ACCEPTANCE OF SUCH TERMS AND
 * CONDITIONS.
 *
 */

/*
 * Offline pose-graph rasterizer.
 *
 * Renders a serialized pose graph (<name>.posegraph + <name>.data) straight to
 * an occupancy-grid image + yaml at an arbitrary resolution, without going
 * through the /map topic.
 *
 * This exists because the live map is rebuilt from scratch every
 * map_update_interval: OccupancyGrid::CreateFromScans() re-raytraces the entire
 * pose graph, so its cost is O(cells) + O(scans x beams x range/resolution) and
 * it is paid over and over while mapping. At a fine resolution that is far too
 * slow to publish continuously (and the /map publisher is transient_local +
 * reliable, so every publish also ships the whole grid).
 *
 * So keep the live map coarse for operator feedback and run this once, at save
 * time, to get the fine-resolution artifact. Nothing here touches DDS, so the
 * grid never has to cross a transport.
 */

#include <sys/resource.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "karto_sdk/Mapper.h"
#include "nav2_map_server/map_io.hpp"
#include "nav2_map_server/map_mode.hpp"
#include "rclcpp/rclcpp.hpp"
#include "slam_toolbox/serialization.hpp"
#include "slam_toolbox/toolbox_types.hpp"
#include "slam_toolbox/visualization_utils.hpp"

namespace
{

struct Options
{
  std::string posegraph;
  std::string output;
  double resolution = 0.01;
  double occupied_thresh = 0.65;
  double free_thresh = 0.1;
  std::string image_format = "png";
  std::string mode = "trinary";
  // Guardrail against a stray outlier scan blowing the bounding box up: karto
  // holds 9 bytes/cell while rastering (int8 occupancy + two uint32 counter
  // grids), so 150M cells is already ~1.35GB of working set.
  // Max map size that can be rasterized at 1cm resolution: ~122 × 122 m, ~15,000 m²
  long long max_cells = 754000000LL;
  // Boost.Serialization recurses through the graph in LoadFromFile(), which can
  // overflow the default 8MB stack on a large map. Mirrors stack_size_to_use.
  long long stack_size = 40000000LL;
};

const char * kUsage =
  "Usage: map_rasterizer --posegraph <basename> --output <basename> [options]\n"
  "\n"
  "  Renders a serialized slam_toolbox pose graph to <output>.<fmt> + <output>.yaml.\n"
  "\n"
  "Required:\n"
  "  --posegraph <basename>   Pose graph basename, WITHOUT the .posegraph/.data\n"
  "                           suffix (both files must sit next to each other).\n"
  "  --output <basename>      Output basename, without extension.\n"
  "\n"
  "Options:\n"
  "  --resolution <m>         Metres per cell (default 0.01).\n"
  "  --occupied-thresh <0-1>  Occupancy threshold (default 0.65).\n"
  "  --free-thresh <0-1>      Free threshold (default 0.1).\n"
  "  --format <png|pgm|bmp>   Image format (default png).\n"
  "  --mode <trinary|scale|raw>  Map mode (default trinary).\n"
  "  --max-cells <n>          Refuse grids larger than this (default 150000000).\n"
  "  --stack-size <bytes>     RLIMIT_STACK for deserialization (default 40000000).\n"
  "  --help                   Print this message.\n";

bool takeValue(int argc, char ** argv, int & i, const rclcpp::Logger & logger, std::string & out)
{
  if (i + 1 >= argc) {
    RCLCPP_ERROR(logger, "map_rasterizer: %s requires a value.", argv[i]);
    return false;
  }
  out = argv[++i];
  return true;
}

bool takeDouble(int argc, char ** argv, int & i, const rclcpp::Logger & logger, double & out)
{
  std::string raw;
  if (!takeValue(argc, argv, i, logger, raw)) {
    return false;
  }
  try {
    out = std::stod(raw);
  } catch (const std::exception &) {
    RCLCPP_ERROR(logger, "map_rasterizer: could not parse '%s' as a number.", raw.c_str());
    return false;
  }
  return true;
}

bool takeLongLong(int argc, char ** argv, int & i, const rclcpp::Logger & logger, long long & out)
{
  std::string raw;
  if (!takeValue(argc, argv, i, logger, raw)) {
    return false;
  }
  try {
    out = std::stoll(raw);
  } catch (const std::exception &) {
    RCLCPP_ERROR(logger, "map_rasterizer: could not parse '%s' as an integer.", raw.c_str());
    return false;
  }
  return true;
}

// Returns false on a usage error, true otherwise. Sets `help` when the caller
// should print usage and exit successfully.
bool parseArgs(int argc, char ** argv, const rclcpp::Logger & logger, Options & opts, bool & help)
{
  help = false;

  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      help = true;
      return true;
    } else if (arg == "--posegraph") {
      if (!takeValue(argc, argv, i, logger, opts.posegraph)) {
        return false;
      }
    } else if (arg == "--output") {
      if (!takeValue(argc, argv, i, logger, opts.output)) {
        return false;
      }
    } else if (arg == "--resolution") {
      if (!takeDouble(argc, argv, i, logger, opts.resolution)) {
        return false;
      }
    } else if (arg == "--occupied-thresh") {
      if (!takeDouble(argc, argv, i, logger, opts.occupied_thresh)) {
        return false;
      }
    } else if (arg == "--free-thresh") {
      if (!takeDouble(argc, argv, i, logger, opts.free_thresh)) {
        return false;
      }
    } else if (arg == "--format") {
      if (!takeValue(argc, argv, i, logger, opts.image_format)) {
        return false;
      }
    } else if (arg == "--mode") {
      if (!takeValue(argc, argv, i, logger, opts.mode)) {
        return false;
      }
    } else if (arg == "--max-cells") {
      if (!takeLongLong(argc, argv, i, logger, opts.max_cells)) {
        return false;
      }
    } else if (arg == "--stack-size") {
      if (!takeLongLong(argc, argv, i, logger, opts.stack_size)) {
        return false;
      }
    } else {
      RCLCPP_ERROR(logger, "map_rasterizer: unrecognised argument '%s'.", arg.c_str());
      return false;
    }
  }

  if (opts.posegraph.empty()) {
    RCLCPP_ERROR(logger, "map_rasterizer: --posegraph is required.");
    return false;
  }
  if (opts.output.empty()) {
    RCLCPP_ERROR(logger, "map_rasterizer: --output is required.");
    return false;
  }
  if (opts.resolution <= 0.0) {
    RCLCPP_ERROR(logger, "map_rasterizer: --resolution must be > 0 (got %f).", opts.resolution);
    return false;
  }

  return true;
}

// Replica of the protected karto::OccupancyGrid::ComputeDimensions(). We need
// the grid size *before* allocating it so an absurd bounding box can be
// rejected with a clear message rather than a bad_alloc or an OOM kill, but the
// real one is not reachable from outside the class. Cheap either way: it only
// walks each scan's bounding box.
void computeDimensions(
  const karto::LocalizedRangeScanVector & scans, double resolution, kt_int32s & width,
  kt_int32s & height)
{
  karto::BoundingBox2 bounding_box;
  for (const auto * scan : scans) {
    if (scan == nullptr) {
      continue;
    }
    bounding_box.Add(scan->GetBoundingBox());
  }

  const double scale = 1.0 / resolution;
  const karto::Size2<kt_double> size = bounding_box.GetSize();

  width = static_cast<kt_int32s>(karto::math::Round(size.GetWidth() * scale));
  height = static_cast<kt_int32s>(karto::math::Round(size.GetHeight() * scale));
}

// Boost.Serialization recurses through the pose graph while deserializing, so
// give the main thread room before LoadFromFile() runs. Only the main thread
// benefits (glibc snapshots RLIMIT_STACK for pthread defaults at startup), but
// that is where we deserialize, so this is effective here.
void raiseStackLimit(long long stack_size, const rclcpp::Logger & logger)
{
  struct rlimit stack_limit;
  if (getrlimit(RLIMIT_STACK, &stack_limit) != 0) {
    RCLCPP_WARN(logger, "map_rasterizer: could not read RLIMIT_STACK; leaving it alone.");
    return;
  }
  if (
    stack_limit.rlim_cur != RLIM_INFINITY &&
    static_cast<long long>(stack_limit.rlim_cur) < stack_size) {
    stack_limit.rlim_cur = static_cast<rlim_t>(stack_size);
    if (setrlimit(RLIMIT_STACK, &stack_limit) != 0) {
      RCLCPP_WARN(logger, "map_rasterizer: could not raise RLIMIT_STACK to %lld.", stack_size);
    } else {
      RCLCPP_INFO(logger, "map_rasterizer: raised stack limit to %lld bytes.", stack_size);
    }
  }
}

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  // Create node and logger
  auto node = std::make_shared<rclcpp::Node>("map_rasterizer");
  const auto logger = node->get_logger();

  // Get args vector
  const std::vector<std::string> args = rclcpp::remove_ros_arguments(argc, argv);
  std::vector<char *> argv_clean;
  argv_clean.reserve(args.size());
  for (const auto & arg : args) {
    argv_clean.push_back(const_cast<char *>(arg.c_str()));
  }

  // Parse args
  Options opts;
  bool help = false;
  if (!parseArgs(static_cast<int>(argv_clean.size()), argv_clean.data(), logger, opts, help)) {
    RCLCPP_ERROR(logger, "%s", kUsage);
    rclcpp::shutdown();
    return 1;
  }
  if (help) {
    RCLCPP_INFO(logger, "%s", kUsage);
    rclcpp::shutdown();
    return 0;
  }

  // Get map mode
  nav2_map_server::MapMode mode;
  try {
    mode = nav2_map_server::map_mode_from_string(opts.mode);
  } catch (const std::invalid_argument &) {
    RCLCPP_ERROR(
      logger,
      "map_rasterizer: '%s' is not a valid map mode "
      "(expected trinary, scale or raw).",
      opts.mode.c_str());
    rclcpp::shutdown();
    return 1;
  }

  // Raise the stack limit before deserializing the pose graph.
  raiseStackLimit(opts.stack_size, logger);

  // ---- Deserialize the pose graph -----------------------------------------

  // mapper: To get graph structure, vertices, edges, poses from .posegraph
  // dataset: To get sensor definition and the scans from .data
  auto mapper = std::make_unique<karto::Mapper>();
  auto dataset = std::make_unique<karto::Dataset>();

  if (!serialization::read(opts.posegraph, *mapper, *dataset, node)) {
    RCLCPP_ERROR(
      logger,
      "map_rasterizer: failed to read pose graph '%s' "
      "(expected %s.posegraph and %s.data).",
      opts.posegraph.c_str(), opts.posegraph.c_str(), opts.posegraph.c_str());
    rclcpp::shutdown();
    return 1;
  }

  if (dataset->GetLasers().empty()) {
    RCLCPP_ERROR(logger, "map_rasterizer: pose graph has no laser objects, cannot raster.");
    rclcpp::shutdown();
    return 1;
  }

  // Scans resolve their sensor through the global SensorManager, so the
  // deserialized laser has to be registered before AddScan() can read its range
  // thresholds. Same step loadSerializedPoseGraph() does for live mapping in
  // slam_toolbox_common.cpp.
  auto * laser = dynamic_cast<karto::LaserRangeFinder *>(dataset->GetLasers()[0]);
  auto * sensor = dynamic_cast<karto::Sensor *>(laser);
  if (!sensor) {
    RCLCPP_ERROR(logger, "map_rasterizer: invalid sensor pointer in dataset.");
    rclcpp::shutdown();
    return 1;
  }
  try {
    karto::SensorManager::GetInstance()->RegisterSensor(sensor, true);
  } catch (const karto::Exception & e) {
    RCLCPP_ERROR(
      logger, "map_rasterizer: could not register sensor: %s", e.GetErrorMessage().c_str());
    rclcpp::shutdown();
    return 1;
  }

  const karto::LocalizedRangeScanVector scans = mapper->GetAllProcessedScans();
  if (scans.empty()) {
    RCLCPP_ERROR(logger, "map_rasterizer: pose graph contains no processed scans.");
    rclcpp::shutdown();
    return 1;
  }

  // ---- Size check before we commit to the allocation ----------------------
  kt_int32s width = 0, height = 0;
  try {
    computeDimensions(scans, opts.resolution, width, height);
  } catch (const karto::Exception & e) {
    RCLCPP_ERROR(
      logger, "map_rasterizer: could not compute map dimensions: %s", e.GetErrorMessage().c_str());
    rclcpp::shutdown();
    return 1;
  }

  const long long cells = static_cast<long long>(width) * static_cast<long long>(height);
  RCLCPP_INFO(
    logger,
    "map_rasterizer: %zu scans -> %d x %d cells at %.4f m/cell (%.1f x %.1f m), "
    "~%lld MB working set.",
    scans.size(), static_cast<int>(width), static_cast<int>(height), opts.resolution,
    width * opts.resolution, height * opts.resolution, (cells * 9LL) / (1024LL * 1024LL));

  if (cells > opts.max_cells) {
    RCLCPP_ERROR(
      logger,
      "map_rasterizer: refusing to raster %lld cells (limit %lld). "
      "Rastering holds ~9 bytes/cell, so this needs ~%lld MB. "
      "Use a coarser --resolution or raise --max-cells if the machine has the RAM.",
      cells, opts.max_cells, (cells * 9LL) / (1024LL * 1024LL));
    rclcpp::shutdown();
    return 1;
  }

  // ---- Raster -------------------------------------------------------------
  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.header.stamp = node->now();
  map.info.resolution = opts.resolution;
  map.info.origin.orientation.w = 1.0;

  const auto raster_started = std::chrono::steady_clock::now();
  {
    karto::OccupancyGrid * occ_grid = nullptr;
    try {
      occ_grid = karto::OccupancyGrid::CreateFromScans(scans, opts.resolution);
    } catch (const karto::Exception & e) {
      RCLCPP_ERROR(
        logger, "map_rasterizer: failed to build the grid: %s", e.GetErrorMessage().c_str());
      rclcpp::shutdown();
      return 1;
    } catch (const std::bad_alloc &) {
      RCLCPP_ERROR(
        logger,
        "map_rasterizer: out of memory building a %lld cell grid. "
        "Use a coarser --resolution.",
        cells);
      rclcpp::shutdown();
      return 1;
    }

    if (!occ_grid) {
      RCLCPP_ERROR(logger, "map_rasterizer: CreateFromScans returned no grid.");
      rclcpp::shutdown();
      return 1;
    }

    vis_utils::toNavMap(occ_grid, map);

    // Free karto's three grids before handing off to the image writer, which
    // allocates its own full-size buffer. Keeps peak RSS to one copy, not two.
    delete occ_grid;
  }
  const auto raster_secs =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - raster_started).count();
  RCLCPP_INFO(logger, "map_rasterizer: rastered in %.1f s.", raster_secs);

  // ---- Write image + yaml -------------------------------------------------
  nav2_map_server::SaveParameters save_parameters;
  save_parameters.map_file_name = opts.output;
  save_parameters.image_format = opts.image_format;
  save_parameters.free_thresh = opts.free_thresh;
  save_parameters.occupied_thresh = opts.occupied_thresh;
  save_parameters.mode = mode;

  const auto write_started = std::chrono::steady_clock::now();

  bool saved = false;
  try {
    saved = nav2_map_server::saveMapToFile(map, save_parameters);
  } catch (const std::exception & e) {
    RCLCPP_ERROR(logger, "map_rasterizer: failed to write map: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  if (!saved) {
    RCLCPP_ERROR(logger, "map_rasterizer: failed to write '%s'.", opts.output.c_str());
    rclcpp::shutdown();
    return 1;
  }

  const auto write_secs =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - write_started).count();

  RCLCPP_INFO(
    logger, "map_rasterizer: wrote %s.%s + %s.yaml in %.1f s (%.1f s total).", opts.output.c_str(),
    opts.image_format.c_str(), opts.output.c_str(), write_secs, raster_secs + write_secs);

  rclcpp::shutdown();
  return 0;
}
