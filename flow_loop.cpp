#include "frame_queue.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>

namespace {

struct CommandLineArguments
{
  size_t queueSize;
  size_t skipIn;
  int skipOut;
  int frameInterval;

  CommandLineArguments(int argc, char** argv)
  {
    const char* keys = R"(
{help h usage ? |    | print this message }
{queue_size     | 30 | number of frames to queue }
{skip_in        |  1 | number of queue frames to skip from input }
{skip_out       |  3 | number of queue frames to skip during output (can be negative) }
{frame_interval | 33 | 1/fps for output video }
)";
    auto parser = cv::CommandLineParser(argc, argv, keys);
    if(parser.has("help")) {
      parser.printMessage();
      exit(EXIT_SUCCESS);
    }
    if(!parser.check()) {
      parser.printErrors();
      exit(EXIT_FAILURE);
    }

    queueSize = parser.get<decltype(queueSize)>("queue_size");
    skipIn = parser.get<decltype(skipIn)>("skip_in");
    skipOut = parser.get<decltype(skipOut)>("skip_out");
    frameInterval = parser.get<decltype(frameInterval)>("frame_interval");

    if(skipIn < 1) {
      throw std::invalid_argument("'skip_in' must be >0");
    }
    if(frameInterval < 0) {
      throw std::invalid_argument("'frame_interval' must be >=0");
    }
  }
};

void calcOpticalFlow(const cv::Mat& prev, const cv::Mat& next, cv::Mat& flow)
{
  constexpr double pyr_scale = 0.5;
  constexpr int levels = 3;
  constexpr int winsize = 15;
  constexpr int iterations = 3;
  constexpr int poly_n = 5;
  constexpr double poly_sigma = 1.2;
  constexpr int flags = 0;
  cv::calcOpticalFlowFarneback(
    prev,
    next,
    flow,
    pyr_scale,
    levels,
    winsize,
    iterations,
    poly_n,
    poly_sigma,
    flags);
}

void flowToMap(cv::Mat& mat)
{
  mat.forEach<cv::Point2f>(
    [](cv::Point2f& px, const int position[]) {
      px = cv::Point2f(
        px.x + position[1],
        px.y + position[0]);
    });
}

} // unnamed namespace

int main(int argc, char** argv)
try {
  // parse command line arguments
  CommandLineArguments cmd(argc, argv);
  auto q = FrameQueue(cmd.queueSize, cmd.skipIn, cmd.skipOut);

  auto capture = cv::VideoCapture(0);
  if(!capture.isOpened()) {
    throw std::runtime_error("failed to open video capture");
  }

  // grab first camera frame
  cv::Mat captured;
  capture >> captured;

  cv::Mat current;
  cv::cvtColor(captured, current, cv::COLOR_BGR2GRAY);

  auto previous = cv::Mat(captured.size(), captured.type());
  auto flow = cv::Mat(captured.size(), CV_32FC2);
  cv::Mat remapped;
  for(;;) {
    // grab next camera frame
    current.copyTo(previous);
    capture >> captured;
    cv::cvtColor(captured, current, cv::COLOR_BGR2GRAY);

    // present frame to queue
    q.enqueueMaybe(
      [&]() -> cv::Mat {
        // calculate optical flow beteen the two frames
        calcOpticalFlow(previous, current, flow);
        flowToMap(flow);

        return flow.clone();
      });

    // get a queued (flow map) frame
    auto&& queued = q.get();

    // remap captured with queued flow map
    cv::remap(captured, remapped, queued, cv::Mat(), cv::INTER_CUBIC);

    // display
    cv::imshow("Display window", remapped);
    if(cv::waitKey(cmd.frameInterval) == 'q') {
      exit(EXIT_SUCCESS);
    }
  }

  return EXIT_SUCCESS;
} catch(const std::exception& e) {
  std::cout << e.what();
}
