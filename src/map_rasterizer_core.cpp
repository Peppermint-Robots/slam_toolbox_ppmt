/*
 * Copyright (c) 2026, Peppermint Robotics
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

// Rendering core shared by the map_rasterizer CLI and map_rasterizer_server. See
// slam_toolbox/map_rasterizer.hpp.

#include "slam_toolbox/map_rasterizer.hpp"

// System
#include <malloc.h>

// C++
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <new>
#include <sstream>
#include <string>
#include <vector>

// Third-party
#include <png.h>

// ROS2
#include <tf2/utils.h>

// Nav2
#include "nav2_map_server/map_io.hpp"
#include "nav2_map_server/map_mode.hpp"

// SLAM Toolbox
#include "karto_sdk/Mapper.h"
#include "slam_toolbox/serialization.hpp"
#include "slam_toolbox/toolbox_types.hpp"
#include "slam_toolbox/visualization_utils.hpp"

namespace map_rasterizer
{

namespace
{

using karto_map_mode = nav2_map_server::MapMode;

// Reports `done` of `total`, if anyone is listening.
inline void report(const ProgressCallback & progress, Phase phase, uint32_t done, uint32_t total)
{
  if (progress) {
    progress(phase, done, total);
  }
}

// Occupancy value -> 8-bit grey, pixel-identical to nav2's per-mode switch (verified
// against libmap_io for every int8 cell value, all modes and formats).
// Follows the same logic as tryWriteMapToFile() in nav2_map_server/map_io.cpp:
// https://github.com/ros-navigation/navigation2/blob/humble_main/nav2_map_server/src/map_io.cpp
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
  double free_thresh, double occupied_thresh, const ProgressCallback & progress,
  const rclcpp::Logger & logger)
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
    report(progress, Phase::WRITING, y + 1, map.info.height);
  }
  png_write_end(png, nullptr);
  png_destroy_write_struct(&png, &info);
  fclose(fp);
  return true;
}

bool writePgm(
  const std::string & file, const nav_msgs::msg::OccupancyGrid & map, karto_map_mode mode,
  double free_thresh, double occupied_thresh, const ProgressCallback & progress,
  const rclcpp::Logger & logger)
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
    report(progress, Phase::WRITING, y + 1, map.info.height);
  }
  fclose(fp);
  return true;
}

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
  const ProgressCallback & progress, const rclcpp::Logger & logger)
{
  const std::string image_file = basename + "." + image_format;
  bool ok = false;
  if (image_format == "png") {
    ok = writePng(image_file, map, mode, free_thresh, occupied_thresh, progress, logger);
  } else if (image_format == "pgm") {
    ok = writePgm(image_file, map, mode, free_thresh, occupied_thresh, progress, logger);
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

// A karto occupancy grid that reports progress after each scan it adds.
// CreateFromScans() below is a copy of karto's (Karto.h).
class ReportingOccupancyGrid : public karto::OccupancyGrid
{
public:
  static std::unique_ptr<ReportingOccupancyGrid> create(
    const karto::LocalizedRangeScanVector & scans, double resolution,
    const ProgressCallback & progress)
  {
    if (scans.empty()) {
      return nullptr;
    }
    kt_int32s width = 0;
    kt_int32s height = 0;
    karto::Vector2<kt_double> offset;
    ComputeDimensions(scans, resolution, width, height, offset);
    std::unique_ptr<ReportingOccupancyGrid> grid(
      new ReportingOccupancyGrid(width, height, offset, resolution, progress));
    grid->CreateFromScans(scans);
    return grid;
  }

private:
  ReportingOccupancyGrid(
    kt_int32s width, kt_int32s height, const karto::Vector2<kt_double> & offset,
    kt_double resolution, const ProgressCallback & progress)
  : karto::OccupancyGrid(width, height, offset, resolution), progress_(progress)
  {
  }

  void CreateFromScans(const karto::LocalizedRangeScanVector & rScans) override
  {
    m_pCellPassCnt->Resize(GetWidth(), GetHeight());
    m_pCellPassCnt->GetCoordinateConverter()->SetOffset(GetCoordinateConverter()->GetOffset());

    m_pCellHitsCnt->Resize(GetWidth(), GetHeight());
    m_pCellHitsCnt->GetCoordinateConverter()->SetOffset(GetCoordinateConverter()->GetOffset());

    const auto total = static_cast<uint32_t>(rScans.size());
    uint32_t done = 0;
    for (karto::LocalizedRangeScan * scan : rScans) {
      if (scan != nullptr) {
        AddScan(scan);
      }
      report(progress_, Phase::RASTERING, ++done, total);
    }

    Update();
  }

  ProgressCallback progress_;
};

// Removes the staging directory on every exit path.
class StagingDir
{
public:
  explicit StagingDir(std::filesystem::path path) : path_(std::move(path)) {}
  ~StagingDir()
  {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  StagingDir(const StagingDir &) = delete;
  StagingDir & operator=(const StagingDir &) = delete;
  const std::filesystem::path & path() const { return path_; }

private:
  std::filesystem::path path_;
};

// Owns the deserialized Mapper and Dataset, and frees them in a safe order.
struct LoadedPoseGraph
{
  std::unique_ptr<karto::Mapper> mapper = std::make_unique<karto::Mapper>();
  std::unique_ptr<karto::Dataset> dataset = std::make_unique<karto::Dataset>();

  LoadedPoseGraph() = default;
  // One owner only: a copy would free the same Mapper, Dataset and scans a second time.
  LoadedPoseGraph(const LoadedPoseGraph &) = delete;
  LoadedPoseGraph & operator=(const LoadedPoseGraph &) = delete;

  ~LoadedPoseGraph()
  {
    const karto::LocalizedRangeScanVector mapper_scans = mapper->GetAllProcessedScans();
    mapper.reset();
    for (karto::LocalizedRangeScan * scan : mapper_scans) {
      delete scan;
    }
    dataset.reset();
  }
};

RenderResult fail(
  RenderResult result, ResultCode code, const std::string & message, const rclcpp::Logger & logger)
{
  RCLCPP_ERROR(logger, "map_rasterizer: %s", message.c_str());
  result.code = code;
  result.message = message;
  return result;
}

double secondsSince(std::chrono::steady_clock::time_point start)
{
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// The whole job, in four steps:
//   1. validate the request
//   2. load .posegraph + .data and register their lasers with karto
//   3. raster every scan into an occupancy grid at the requested resolution
//   4. write the image + yaml into a staging directory, then rename both into place
// Each step returns a failed RenderResult on error.
RenderResult render(
  const RenderRequest & request, const rclcpp::Node::SharedPtr & node,
  const ProgressCallback & progress)
{
  const auto logger = node->get_logger();
  RenderResult result;

  // ---- Validate Request -----------------------------------------
  const std::string invalid = validate(request);
  if (!invalid.empty()) {
    return fail(result, ResultCode::INVALID_REQUEST, invalid, logger);
  }
  const karto_map_mode mode = nav2_map_server::map_mode_from_string(request.mode);

  // ---- Load pose graph -----------------------------------------
  report(progress, Phase::LOADING, 0, 1);
  const auto deserialize_started = std::chrono::steady_clock::now();

  for (const char * extension : {".posegraph", ".data"}) {
    const std::string file = request.basename + extension;
    if (!std::filesystem::exists(file)) {
      return fail(result, ResultCode::LOAD_FAILED, "'" + file + "' does not exist.", logger);
    }
  }

  // mapper: To get graph structure, vertices, edges, poses from .posegraph
  // dataset: To get sensor definition and the scans from .data
  LoadedPoseGraph graph;
  karto::Mapper & mapper = *graph.mapper;
  karto::Dataset & dataset = *graph.dataset;

  // serialization::read only catches Boost archive errors. A .data that does not
  // match its .posegraph can instead read a garbage length and throw bad_alloc,
  // which must fail this render, not take the server down.
  std::string load_error;
  bool loaded = false;
  try {
    loaded = serialization::read(request.basename, mapper, dataset, node);
  } catch (const std::bad_alloc &) {
    load_error = "ran out of memory reading '" + request.basename +
                 "' -- the .data is likely corrupt or from a different save than the "
                 ".posegraph.";
  } catch (const karto::Exception & e) {
    load_error = "failed to read pose graph '" + request.basename + "': " + e.GetErrorMessage();
  } catch (const std::exception & e) {
    load_error = "failed to read pose graph '" + request.basename + "': " + e.what();
  }
  if (load_error.empty() && !loaded) {
    load_error = "failed to read pose graph '" + request.basename + "' (expected " +
                 request.basename + ".posegraph and " + request.basename + ".data).";
  }
  if (!load_error.empty()) {
    return fail(result, ResultCode::LOAD_FAILED, load_error, logger);
  }

  if (dataset.GetLasers().empty()) {
    return fail(
      result, ResultCode::LOAD_FAILED, "pose graph has no laser objects, cannot raster.", logger);
  }

  // Register the lidar with karto: scans look up their range limits through it.
  // Same step loadSerializedPoseGraph() does for live mapping in slam_toolbox_common.cpp.
  auto * sensor = dynamic_cast<karto::Sensor *>(dataset.GetLasers()[0]);
  if (!sensor) {
    return fail(result, ResultCode::LOAD_FAILED, "invalid sensor pointer in dataset.", logger);
  }
  try {
    karto::SensorManager::GetInstance()->RegisterSensor(sensor, true);
  } catch (const karto::Exception & e) {
    return fail(
      result, ResultCode::LOAD_FAILED, "could not register sensor: " + e.GetErrorMessage(), logger);
  }

  const karto::LocalizedRangeScanVector scans = mapper.GetAllProcessedScans();
  if (scans.empty()) {
    return fail(result, ResultCode::LOAD_FAILED, "pose graph contains no processed scans.", logger);
  }
  result.scans = static_cast<uint32_t>(scans.size());
  result.deserialize_s = secondsSince(deserialize_started);
  report(progress, Phase::LOADING, 1, 1);

  // ---- Raster -------------------------------------------------------------
  const auto raster_started = std::chrono::steady_clock::now();

  nav_msgs::msg::OccupancyGrid map;
  map.header.frame_id = "map";
  map.header.stamp = node->now();
  map.info.resolution = request.resolution;
  map.info.origin.orientation.w = 1.0;

  {
    std::unique_ptr<ReportingOccupancyGrid> occ_grid;
    try {
      occ_grid = ReportingOccupancyGrid::create(scans, request.resolution, progress);
    } catch (const karto::Exception & e) {
      return fail(
        result, ResultCode::RASTER_FAILED, "failed to build the grid: " + e.GetErrorMessage(),
        logger);
    } catch (const std::bad_alloc &) {
      std::ostringstream msg;
      msg << "out of memory building the grid from " << scans.size() << " scans at "
          << request.resolution << " m/cell. Use a coarser resolution.";
      return fail(result, ResultCode::OUT_OF_MEMORY, msg.str(), logger);
    }

    if (!occ_grid) {
      return fail(result, ResultCode::RASTER_FAILED, "CreateFromScans returned no grid.", logger);
    }

    vis_utils::toNavMap(occ_grid.get(), map);
  }

  result.raster_s = secondsSince(raster_started);
  result.width = map.info.width;
  result.height = map.info.height;
  RCLCPP_INFO(
    logger,
    "map_rasterizer: rastered %zu scans -> %u x %u cells at %.4f m/cell (%.1f x %.1f m) in "
    "%.1f s.",
    scans.size(), map.info.width, map.info.height, request.resolution,
    map.info.width * request.resolution, map.info.height * request.resolution, result.raster_s);

  // ---- Write image + yaml -------------------------------------------------
  // Into a staging directory beside the output, then renamed into place. Beside it,
  // so the renames stay on one filesystem and are atomic; under the final basename,
  // so the yaml's image: key needs no rewriting.
  namespace fs = std::filesystem;
  const fs::path output(request.basename);
  const fs::path output_dir = output.parent_path().empty() ? fs::path(".") : output.parent_path();
  std::string staging_template =
    (output_dir / (".render_" + output.filename().string() + "_XXXXXX")).string();
  if (mkdtemp(staging_template.data()) == nullptr) {
    return fail(
      result, ResultCode::WRITE_FAILED,
      "cannot create a staging directory in '" + output_dir.string() + "': " + std::strerror(errno),
      logger);
  }
  const StagingDir staging(staging_template);
  const std::string staged_basename = (staging.path() / output.filename()).string();

  const auto write_started = std::chrono::steady_clock::now();

  bool saved = false;
  try {
    if (request.legacy_writer) {
      nav2_map_server::SaveParameters save_parameters;
      save_parameters.map_file_name = staged_basename;
      save_parameters.image_format = request.image_format;
      save_parameters.free_thresh = request.free_thresh;
      save_parameters.occupied_thresh = request.occupied_thresh;
      save_parameters.mode = mode;
      // nav2's writer has no progress of its own: report its start and end.
      report(progress, Phase::WRITING, 0, 1);
      saved = nav2_map_server::saveMapToFile(map, save_parameters);
      report(progress, Phase::WRITING, 1, 1);
    } else {
      saved = saveMapDirect(
        map, staged_basename, request.image_format, mode, request.free_thresh,
        request.occupied_thresh, progress, logger);
    }
  } catch (const std::exception & e) {
    return fail(
      result, ResultCode::WRITE_FAILED, std::string("failed to write map: ") + e.what(), logger);
  }
  if (!saved) {
    return fail(
      result, ResultCode::WRITE_FAILED, "failed to write '" + request.basename + "'.", logger);
  }

  // Move the image and yaml out of staged directory.
  for (const std::string & extension : {request.image_format, std::string("yaml")}) {
    std::error_code ec;
    fs::rename(staged_basename + "." + extension, request.basename + "." + extension, ec);
    if (ec) {
      return fail(
        result, ResultCode::WRITE_FAILED,
        "could not move the rendered " + extension + " into place at '" + request.basename + "." +
          extension + "': " + ec.message(),
        logger);
    }
  }
  result.write_s = secondsSince(write_started);

  RCLCPP_INFO(
    logger, "map_rasterizer: wrote %s.%s + %s.yaml in %.1f s (%.1f s total).",
    request.basename.c_str(), request.image_format.c_str(), request.basename.c_str(), result.write_s,
    result.raster_s + result.write_s);

  result.code = ResultCode::SUCCESS;
  return result;
}

}  // namespace

std::string validate(const RenderRequest & request)
{
  if (request.basename.empty()) {
    return "basename is required.";
  }
  if (!(request.resolution > 0.0) || !std::isfinite(request.resolution)) {
    return "resolution must be > 0.";
  }
  if (
    !(request.free_thresh >= 0.0 && request.free_thresh <= 1.0) ||
    !(request.occupied_thresh >= 0.0 && request.occupied_thresh <= 1.0)) {
    return "free_thresh and occupied_thresh must be within 0-1.";
  }

  if (!request.legacy_writer && request.image_format != "png" && request.image_format != "pgm") {
    return "format '" + request.image_format + "' has no direct writer (png or pgm).";
  }
  try {
    nav2_map_server::map_mode_from_string(request.mode);
  } catch (const std::invalid_argument &) {
    return "'" + request.mode + "' is not a valid map mode (expected trinary, scale or raw).";
  }
  return "";
}

RenderResult renderPosegraph(
  const RenderRequest & request, const rclcpp::Node::SharedPtr & node,
  const ProgressCallback & progress)
{
  RenderResult result;
  try {
    result = render(request, node, progress);
  } catch (const std::bad_alloc &) {
    result = fail(result, ResultCode::OUT_OF_MEMORY, "out of memory.", node->get_logger());
  } catch (const karto::Exception & e) {
    result = fail(result, ResultCode::RASTER_FAILED, e.GetErrorMessage(), node->get_logger());
  } catch (const std::exception & e) {
    result = fail(result, ResultCode::RASTER_FAILED, e.what(), node->get_logger());
  }
  // Return freed memory to the OS, so the server does not hold it between renders.
  malloc_trim(0);
  return result;
}

}  // namespace map_rasterizer
