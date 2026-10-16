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

/*
 * Offline pose-graph rasterizer: renders a saved pose graph (<name>.posegraph +
 * <name>.data) to <name>.png + <name>.yaml at any resolution, at save time.
 *
 * The image is written directly with libpng, ~18x faster than
 * nav2_map_server::saveMapToFile on large maps (legacy_writer keeps that option).
 *
 * Used by the map_rasterizer CLI and the map_rasterizer_server node.
 */

#ifndef SLAM_TOOLBOX__MAP_RASTERIZER_HPP_
#define SLAM_TOOLBOX__MAP_RASTERIZER_HPP_

#include <cstdint>
#include <functional>
#include <string>

#include "rclcpp/rclcpp.hpp"

namespace map_rasterizer
{

// Values match the RESULT_* constants in srv/RasterizePosegraph.srv.
enum class ResultCode : uint8_t {
  SUCCESS = 0,
  BUSY = 1,
  INVALID_REQUEST = 2,
  LOAD_FAILED = 3,
  OUT_OF_MEMORY = 4,
  RASTER_FAILED = 5,
  WRITE_FAILED = 6,
};

// Stages of a render reported to a ProgressCallback. Values match the PHASE_* constants in
// msg/RasterizeProgress.msg.
enum class Phase : uint8_t {
  LOADING = 0,
  RASTERING = 1,
  WRITING = 2,
};

// Called as a render progresses: `done` of `total` scans while rastering, rows while writing.
// Loading has no progress of its own: 0 of 1 at start, 1 of 1 when done. Called on the
// rendering thread, often (once per scan or row), so keep it cheap.
using ProgressCallback = std::function<void(Phase phase, uint32_t done, uint32_t total)>;

// Boost.Serialization recurses through the graph in LoadFromFile(), which can
// overflow the default 8MB stack on a large map. Mirrors stack_size_to_use from slam_tb params.
constexpr long long kDefaultStackSize = 40000000LL;

struct RenderRequest
{
  // Reads <basename>.posegraph + .data; writes <basename>.<format> + .yaml next to them.
  std::string basename;
  double resolution = 0.01;
  double occupied_thresh = 0.65;
  double free_thresh = 0.1;
  std::string image_format = "png";
  std::string mode = "trinary";
  // nav2_map_server::saveMapToFile instead of the direct writer.
  bool legacy_writer = false;
};

struct RenderResult
{
  ResultCode code = ResultCode::RASTER_FAILED;
  std::string message;
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t scans = 0;
  double deserialize_s = 0.0;
  double raster_s = 0.0;
  double write_s = 0.0;
};

// Empty if `request` is usable, otherwise what is wrong with it.
std::string validate(const RenderRequest & request);

// Renders request.basename's pose graph to request.basename.<format> + .yaml.
RenderResult renderPosegraph(
  const RenderRequest & request, const rclcpp::Node::SharedPtr & node,
  const ProgressCallback & progress = {});

}  // namespace map_rasterizer

#endif  // SLAM_TOOLBOX__MAP_RASTERIZER_HPP_
