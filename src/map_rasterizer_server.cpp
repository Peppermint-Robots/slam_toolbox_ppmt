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
 * map_rasterizer_server: serves map_rasterizer/rasterize, which renders a saved pose
 * graph to an image + yaml.
 */

// C++
#include <atomic>
#include <functional>
#include <memory>
#include <string>

// Third-party
#include <boost/thread/thread.hpp>

// ROS2
#include "rclcpp/rclcpp.hpp"

// SLAM Toolbox
#include "slam_toolbox/map_rasterizer.hpp"
#include "slam_toolbox_ppmt/srv/rasterize_posegraph.hpp"

namespace
{

using RasterizePosegraph = slam_toolbox_ppmt::srv::RasterizePosegraph;

class MapRasterizerServer
{
public:
  explicit MapRasterizerServer(const rclcpp::Node::SharedPtr & node) : node_(node)
  {
    stack_size_ =
      node_->declare_parameter<int64_t>("stack_size", map_rasterizer::kDefaultStackSize);

    // Reentrant + MultiThreadedExecutor: a request arriving mid-render is answered
    // BUSY instead of queueing.
    callback_group_ = node_->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    service_ = node_->create_service<RasterizePosegraph>(
      "~/rasterize",
      std::bind(&MapRasterizerServer::handle, this, std::placeholders::_1, std::placeholders::_2),
      rmw_qos_profile_services_default, callback_group_);

    RCLCPP_INFO(node_->get_logger(), "map_rasterizer: serving %s.", service_->get_service_name());
  }

private:
  void handle(
    const std::shared_ptr<RasterizePosegraph::Request> request,
    std::shared_ptr<RasterizePosegraph::Response> response)
  {
    if (busy_.exchange(true)) {
      response->result = RasterizePosegraph::Response::RESULT_BUSY;
      response->message = "a render is already running.";
      RCLCPP_WARN(
        node_->get_logger(), "map_rasterizer: rejected '%s': %s", request->posegraph.c_str(),
        response->message.c_str());
      return;
    }
    // Clears busy_ however handle() exits, so an exception cannot leave it stuck at true.
    struct ClearBusy
    {
      std::atomic<bool> & busy;
      ~ClearBusy() { busy = false; }
    } clear_busy{busy_};

    map_rasterizer::RenderRequest render_request;
    render_request.posegraph = request->posegraph;
    render_request.output = request->output;
    render_request.resolution = request->resolution;
    render_request.image_format = request->format;
    render_request.free_thresh = request->free_thresh;
    render_request.occupied_thresh = request->occupied_thresh;

    RCLCPP_INFO(
      node_->get_logger(), "map_rasterizer: rendering '%s' -> '%s' at %.4f m/cell.",
      request->posegraph.c_str(), request->output.c_str(), request->resolution);

    // Deserializing recurses through the pose graph. Executor threads get glibc's
    // default 8 MB stack, which a large graph overflows, and RLIMIT_STACK cannot help
    // here: glibc reads it once, at startup, for every thread it creates later. So the
    // render gets a thread of its own, with the stack size set explicitly.
    map_rasterizer::RenderResult result;
    boost::thread::attributes attributes;
    attributes.set_stack_size(static_cast<std::size_t>(stack_size_));
    boost::thread worker(
      attributes, [&]() { result = map_rasterizer::renderPosegraph(render_request, node_); });
    worker.join();

    response->result = static_cast<uint8_t>(result.code);
    response->message = result.message;
    response->width = result.width;
    response->height = result.height;
    response->scans = result.scans;
    response->deserialize_s = result.deserialize_s;
    response->raster_s = result.raster_s;
    response->write_s = result.write_s;
  }

  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::Service<RasterizePosegraph>::SharedPtr service_;
  std::atomic<bool> busy_{false};
  int64_t stack_size_ = map_rasterizer::kDefaultStackSize;
};

}  // namespace

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);

  auto node = std::make_shared<rclcpp::Node>("map_rasterizer");
  MapRasterizerServer server(node);

  // Two threads: one for a render in progress, one to answer BUSY meanwhile.
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}
