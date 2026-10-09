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
 * Renders a saved pose graph (<name>.posegraph + <name>.data) to <name>.png +
 * <name>.yaml at any resolution.
 *
 * The live map is raytraced from the whole pose graph on every
 * map_update_interval and published on "/map" topic, so it has to stay coarse.
 * This tool does the fine resolution render once, at save time.
 *
 * The image is written directly with libpng, one row at a time, rather than via
 * nav2_map_server::saveMapToFile, which sets pixels one by one through
 * GraphicsMagick and is ~18x slower on large maps (--legacy-writer keeps this option).
 */

// System
#include <sys/resource.h>

// C++
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

// Third-party
#include <png.h>

// ROS2
#include <tf2/utils.h>

#include "rclcpp/rclcpp.hpp"

// Nav2
#include "nav2_map_server/map_io.hpp"
#include "nav2_map_server/map_mode.hpp"

// SLAM Toolbox
#include "karto_sdk/Mapper.h"
#include "slam_toolbox/serialization.hpp"
#include "slam_toolbox/toolbox_types.hpp"
#include "slam_toolbox/visualization_utils.hpp"

namespace
{

using karto_map_mode = nav2_map_server::MapMode;

struct Options
{
  std::string posegraph;
  std::string output;
  double resolution = 0.01;
  double occupied_thresh = 0.65;
  double free_thresh = 0.1;
  std::string image_format = "png";
  std::string mode = "trinary";
  // Boost.Serialization recurses through the graph in LoadFromFile(), which can
  // overflow the default 8MB stack on a large map. Mirrors stack_size_to_use from slam_tb params.
  long long stack_size = 40000000LL;
  bool legacy_writer = false;
};

// Occupancy value -> 8-bit grey, pixel-identical to nav2's per-mode switch (verified
// against libmap_io for every int8 cell value, all modes and formats).
// Follows the same logic as tryWriteMapToFile() in nav2_map_server/map_io.cpp:
// https://github.com/Peppermint-Robots/navigation2/blob/humble_main_ppmt/nav2_map_server/src/map_io.cpp
inline uint8_t cellToGrey(
  int8_t cell, karto_map_mode mode, int free_thresh_int, int occupied_thresh_int)
{
  switch (mode) {
    case nav2_map_server::MapMode::Trinary:
      if (cell < 0 || 100 < cell) {
        return 205;
      } else if (cell <= free_thresh_int) {
        return 254;
      } else if (occupied_thresh_int <= cell) {
        return 0;
      }
      return 205;
    case nav2_map_server::MapMode::Scale:
      if (cell < 0 || 100 < cell) {
        return 127;  // nav2's ColorGray{0.5}; the PNG also marks it transparent
      }
      // Truncating, as GraphicsMagick does: cell 1 -> 252, not 253.
      return static_cast<uint8_t>((100.0 - cell) / 100.0 * 255.0);
    case nav2_map_server::MapMode::Raw:
    default:
      if (cell < 0 || 100 < cell) {
        return 255;
      }
      return static_cast<uint8_t>(cell);
  }
}

// Fill Row `y` of the output image.
inline void fillRow(
  const nav_msgs::msg::OccupancyGrid & map, size_t y, karto_map_mode mode, double free_thresh,
  double occupied_thresh, bool alpha, std::vector<uint8_t> & row)
{
  const int free_thresh_int = std::rint(free_thresh * 100.0);
  const int occupied_thresh_int = std::rint(occupied_thresh * 100.0);
  const size_t w = map.info.width;
  const int8_t * src = &map.data[w * (map.info.height - y - 1)];
  for (size_t x = 0; x < w; x++) {
    const uint8_t grey = cellToGrey(src[x], mode, free_thresh_int, occupied_thresh_int);
    if (alpha) {
      row[2 * x] = grey;
      row[2 * x + 1] = (src[x] < 0 || 100 < src[x]) ? 0 : 255;
    } else {
      row[x] = grey;
    }
  }
}

bool writePng(
  const std::string & file, const nav_msgs::msg::OccupancyGrid & map, karto_map_mode mode,
  double free_thresh, double occupied_thresh, const rclcpp::Logger & logger)
{
  FILE * fp = fopen(file.c_str(), "wb");
  if (fp == nullptr) {
    RCLCPP_ERROR(logger, "map_rasterizer: cannot open '%s' for writing.", file.c_str());
    return false;
  }
  png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  png_infop info = png ? png_create_info_struct(png) : nullptr;
  if (png == nullptr || info == nullptr) {
    if (png != nullptr) {
      png_destroy_write_struct(&png, info ? &info : nullptr);
    }
    fclose(fp);
    RCLCPP_ERROR(logger, "map_rasterizer: libpng allocation failed.");
    return false;
  }
  const bool alpha = (mode == nav2_map_server::MapMode::Scale);
  // Declared before setjmp: a longjmp out of png_write_row must not skip its destructor.
  std::vector<uint8_t> row(map.info.width * (alpha ? 2 : 1));
  // NOTE: (Ulkesh, 09-10-2026): libpng reports errors by longjmp'ing back here. Refer this for more
  // details: https://github.com/pnggroup/libpng/blob/libpng16/example.c#L204
  if (setjmp(png_jmpbuf(png))) {
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    std::remove(file.c_str());
    RCLCPP_ERROR(logger, "map_rasterizer: libpng failed writing '%s'.", file.c_str());
    return false;
  }

  png_init_io(png, fp);
  png_set_IHDR(
    png, info, map.info.width, map.info.height, 8,
    alpha ? PNG_COLOR_TYPE_GRAY_ALPHA : PNG_COLOR_TYPE_GRAY, PNG_INTERLACE_NONE,
    PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
  png_write_info(png, info);

  for (size_t y = 0; y < map.info.height; y++) {
    fillRow(map, y, mode, free_thresh, occupied_thresh, alpha, row);
    png_write_row(png, row.data());
  }
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
  return true;
}

bool writePgm(
  const std::string & file, const nav_msgs::msg::OccupancyGrid & map, karto_map_mode mode,
  double free_thresh, double occupied_thresh, const rclcpp::Logger & logger)
{
  FILE * fp = fopen(file.c_str(), "wb");
  if (fp == nullptr) {
    RCLCPP_ERROR(logger, "map_rasterizer: cannot open '%s' for writing.", file.c_str());
    return false;
  }
  fprintf(fp, "P5\n%u %u\n255\n", map.info.width, map.info.height);
  std::vector<uint8_t> row(map.info.width);
  for (size_t y = 0; y < map.info.height; y++) {
    fillRow(map, y, mode, free_thresh, occupied_thresh, false, row);
    if (fwrite(row.data(), 1, row.size(), fp) != row.size()) {
      fclose(fp);
      RCLCPP_ERROR(logger, "map_rasterizer: short write on '%s'.", file.c_str());
      return false;
    }
  }
  fclose(fp);
  return true;
}

// Same output as nav2_map_server::saveMapToFile's YAML
bool writeYaml(
  const std::string & file, const std::string & image_name,
  const nav_msgs::msg::OccupancyGrid & map, karto_map_mode mode, double free_thresh,
  double occupied_thresh, const rclcpp::Logger & logger)
{
  const double yaw = tf2::getYaw(map.info.origin.orientation);

  std::ostringstream o;
  o << std::setprecision(3);
  o << "image: " << image_name << "\n";
  o << "mode: " << nav2_map_server::map_mode_to_string(mode) << "\n";
  o << "resolution: " << map.info.resolution << "\n";
  o << "origin: [" << map.info.origin.position.x << ", " << map.info.origin.position.y << ", "
    << yaw << "]\n";
  o << "negate: 0\n";
  o << "occupied_thresh: " << occupied_thresh << "\n";
  o << "free_thresh: " << free_thresh;

  std::ofstream out(file);
  if (!out) {
    RCLCPP_ERROR(logger, "map_rasterizer: cannot open '%s' for writing.", file.c_str());
    return false;
  }
  out << o.str();
  return out.good();
}

bool saveMapDirect(
  const nav_msgs::msg::OccupancyGrid & map, const std::string & basename,
  const std::string & image_format, karto_map_mode mode, double free_thresh, double occupied_thresh,
  const rclcpp::Logger & logger)
{
  const std::string image_file = basename + "." + image_format;
  bool ok = false;
  if (image_format == "png") {
    ok = writePng(image_file, map, mode, free_thresh, occupied_thresh, logger);
  } else if (image_format == "pgm") {
    ok = writePgm(image_file, map, mode, free_thresh, occupied_thresh, logger);
  } else {
    RCLCPP_ERROR(
      logger, "map_rasterizer: --format %s has no direct writer; use --legacy-writer.",
      image_format.c_str());
    return false;
  }
  if (!ok) {
    RCLCPP_ERROR(
      logger, "map_rasterizer: failed to write the %s image '%s'; not writing the yaml.",
      image_format.c_str(), image_file.c_str());
    return false;
  }

  const size_t slash = image_file.find_last_of("/\\");
  const std::string image_name =
    (slash == std::string::npos) ? image_file : image_file.substr(slash + 1);
  return writeYaml(basename + ".yaml", image_name, map, mode, free_thresh, occupied_thresh, logger);
}

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
  "  --stack-size <bytes>     RLIMIT_STACK for deserialization (default 40000000).\n"
  "  --legacy-writer          Write via nav2_map_server instead of the direct writer\n"
  "                           (much slower; for output comparison only).\n"
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
    } else if (arg == "--stack-size") {
      if (!takeLongLong(argc, argv, i, logger, opts.stack_size)) {
        return false;
      }
    } else if (arg == "--legacy-writer") {
      opts.legacy_writer = true;
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
        "map_rasterizer: out of memory building the grid from %zu scans at %.4f m/cell. "
        "Use a coarser --resolution.",
        scans.size(), opts.resolution);
      rclcpp::shutdown();
      return 1;
    }

    if (!occ_grid) {
      RCLCPP_ERROR(logger, "map_rasterizer: CreateFromScans returned no grid.");
      rclcpp::shutdown();
      return 1;
    }

    vis_utils::toNavMap(occ_grid, map);

    // Free up memory before writing the image
    delete occ_grid;
  }
  const auto raster_secs =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - raster_started).count();
  const long long cells =
    static_cast<long long>(map.info.width) * static_cast<long long>(map.info.height);
  RCLCPP_INFO(
    logger,
    "map_rasterizer: rastered %zu scans -> %u x %u cells at %.4f m/cell (%.1f x %.1f m) in "
    "%.1f s.",
    scans.size(), map.info.width, map.info.height, opts.resolution,
    map.info.width * opts.resolution, map.info.height * opts.resolution, raster_secs);

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
    saved = opts.legacy_writer ? nav2_map_server::saveMapToFile(map, save_parameters)
                               : saveMapDirect(
                                   map, opts.output, opts.image_format, mode, opts.free_thresh,
                                   opts.occupied_thresh, logger);
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
