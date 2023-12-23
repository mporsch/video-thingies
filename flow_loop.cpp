#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/videoio.hpp>

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

struct CommandLineArguments
{
  size_t queueSize;
  size_t skipIn;
  size_t skipOut;
  int frameInterval;

  CommandLineArguments(int argc, char** argv)
  {
    const char* keys = R"(
{help h usage ? |    | print this message }
{queue_size     | 30 | number of frames to queue }
{skip_in        |  1 | number of queue frames to skip from input }
{skip_out       |  3 | number of queue frames to skip during output }
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
    if(skipOut < 1) {
      throw std::invalid_argument("'skip_out' must be >0");
    }
    if(frameInterval < 0) {
      throw std::invalid_argument("'frame_interval' must be >=0");
    }
  }
};

using Frame = cv::Mat;

struct FrameQueue
{
  // maximum number of frames to keep
  size_t maxSize;

  // skip count used to skip some frames entirely
  size_t skipIn;
  size_t skipOut;

  // input frame index used to skip some frames entirely
  size_t idxIn;

  // output frame index used to iterate queue frames for display
  size_t idxOut;

  // underlying frame storage
  std::vector<Frame> storage;

  FrameQueue(
      size_t maxSize,
      size_t skipIn,
      size_t skipOut)
    : maxSize(maxSize)
    , skipIn(skipIn)
    , skipOut(skipOut)
    , idxIn(0)
    , idxOut(0)
  {
    storage.reserve(this->maxSize);
  }

  void enqueueMaybe(const Frame& f)
  {
    if((idxIn++ % skipIn) != 0) {
      return;
    }
    if(storage.size() >= maxSize) {
      storage.erase(storage.begin());
      --idxOut; // rollover is considered in get()
    }
    storage.push_back(f);
  }

  const Frame& get()
  {
    if(idxOut >= storage.size()) {
      idxOut %= storage.size();
    }
#ifdef DEBUG_QUEUE
    std::cout << "getting " << idxOut  << " (of " << storage.size() << ")" << std::endl;
#endif // DEBUG_QUEUE
    auto&& f = storage.at(idxOut);
    idxOut += skipOut;
    return f;
  }
};

void calcOpticalFlow(cv::Mat prev, cv::Mat next, cv::Mat flow)
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

void flowToMap(cv::Mat src, cv::Mat dst)
{
  for(int y = 0; y < src.rows; ++y) {
    for(int x = 0; x < src.cols; ++x) {
      auto f = src.at<cv::Point2f>(y, x);
      dst.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
    }
  }
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
  for(;;) {
    // grab next camera frame
    current.copyTo(previous);
    capture >> captured;
    cv::cvtColor(captured, current, cv::COLOR_BGR2GRAY);

    // calculate optical flow beteen the two frames
    calcOpticalFlow(previous, current, flow);
    flowToMap(flow, flow);

    // present frame to queue
    q.enqueueMaybe(flow);

    // get a queued (flow map) frame
    auto&& queued = q.get();

    // remap captured with queued flow map
    cv::remap(captured, captured, queued, cv::Mat(), cv::INTER_CUBIC);

    // display
    cv::imshow("Display window", captured);
    if(cv::waitKey(cmd.frameInterval) == 'q') {
      exit(EXIT_SUCCESS);
    }
  }

  return EXIT_SUCCESS;
} catch(const std::exception& e) {
  std::cout << e.what();
}