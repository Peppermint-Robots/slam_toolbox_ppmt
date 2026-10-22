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
 * map_rasterizer CLI: renders a saved pose graph to an image + yaml once and exits.
 * For manual and batch use; the mapping stack calls map_rasterizer_server instead.
 * The rendering itself lives in slam_toolbox/map_rasterizer.hpp.
 */

// System
#include <sys/resource.h>

// C++
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// ROS2
#include "rclcpp/rclcpp.hpp"

// SLAM Toolbox
#include "slam_toolbox/map_rasterizer.hpp"

namespace
{

struct Options
{
  map_rasterizer::RenderRequest request;
  long long stack_size = map_rasterizer::kDefaultStackSize;
};

const char * kUsage =
  "Usage: map_rasterizer --basename <basename> [options]\n"
  "\n"
  "  Renders a serialized slam_toolbox pose graph to <basename>.<fmt> + <basename>.yaml.\n"
  "\n"
  "Required:\n"
  "  --basename <basename>    Pose graph basename, WITHOUT the .posegraph/.data\n"
  "                           suffix (both files must sit next to each other).\n"
  "                           The map is written next to them, under the same name.\n"
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
    } else if (arg == "--basename") {
      if (!takeValue(argc, argv, i, logger, opts.request.basename)) {
        return false;
      }
    } else if (arg == "--resolution") {
      if (!takeDouble(argc, argv, i, logger, opts.request.resolution)) {
        return false;
      }
    } else if (arg == "--occupied-thresh") {
      if (!takeDouble(argc, argv, i, logger, opts.request.occupied_thresh)) {
        return false;
      }
    } else if (arg == "--free-thresh") {
      if (!takeDouble(argc, argv, i, logger, opts.request.free_thresh)) {
        return false;
      }
    } else if (arg == "--format") {
      if (!takeValue(argc, argv, i, logger, opts.request.image_format)) {
        return false;
      }
    } else if (arg == "--mode") {
      if (!takeValue(argc, argv, i, logger, opts.request.mode)) {
        return false;
      }
    } else if (arg == "--stack-size") {
      if (!takeLongLong(argc, argv, i, logger, opts.stack_size)) {
        return false;
      }
    } else if (arg == "--legacy-writer") {
      opts.request.legacy_writer = true;
    } else {
      RCLCPP_ERROR(logger, "map_rasterizer: unrecognised argument '%s'.", arg.c_str());
      return false;
    }
  }

  if (opts.request.basename.empty()) {
    RCLCPP_ERROR(logger, "map_rasterizer: --basename is required.");
    return false;
  }
  if (opts.request.resolution <= 0.0) {
    RCLCPP_ERROR(
      logger, "map_rasterizer: --resolution must be > 0 (got %f).", opts.request.resolution);
    return false;
  }

  return true;
}

// Loading a pose graph recurses deeply, so raise the main thread's stack limit first.
// This only helps the main thread, which is where renderPosegraph() is called.
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

  // Raise the stack limit before deserializing the pose graph.
  raiseStackLimit(opts.stack_size, logger);

  const map_rasterizer::RenderResult result = map_rasterizer::renderPosegraph(opts.request, node);

  rclcpp::shutdown();
  return result.code == map_rasterizer::ResultCode::SUCCESS ? 0 : 1;
}
