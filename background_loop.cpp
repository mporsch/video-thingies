#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/videoio.hpp"

#include <cassert>
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
  int morphSize;
  int frameInterval;

  CommandLineArguments(int argc, char** argv)
  {
    const char* keys = R"(
{help h usage ? |    | print this message }
{queue_size     | 30 | number of frames to queue }
{skip_in        |  1 | number of queue frames to skip from input }
{skip_out       |  3 | number of queue frames to skip during output }
{morph_size     |  5 | size of mophological close }
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
    morphSize = parser.get<decltype(morphSize)>("morph_size");
    frameInterval = parser.get<decltype(frameInterval)>("frame_interval");

    if(skipIn < 1) {
      throw std::invalid_argument("'skip_in' must be >0");
    }
    if(skipOut < 1) {
      throw std::invalid_argument("'skip_out' must be >0");
    }
    if(morphSize <= 0) {
      throw std::invalid_argument("'morph_size' must be >0");
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

  // underlying frame storage
  std::vector<Frame> storage;

  // input frame index used to skip some frames entirely
  size_t idxIn = 0;

  // output frame index used to iterate queue frames for display
  size_t idxOut = 0;

  void enqueueMaybe(const Frame& f)
  {
    if((idxIn++ % skipIn) != 0) {
      return;
    }
    if(storage.size() >= maxSize) {
      storage.erase(storage.begin());
      --idxOut; // rollover is considered in get()
    }
    storage.push_back(f.clone());
  }

  const Frame& get()
  {
    assert(!storage.empty());
    auto idx = idxOut % storage.size();
#ifdef DEBUG_QUEUE
    std::cout << "getting " << idx  << " (of " << storage.size() << ")" << std::endl;
#endif // DEBUG_QUEUE
    idxOut += skipOut;
    return storage.at(idx);
  }
};

} // unnamed namespace

int main(int argc, char** argv)
try {
  // parse command line arguments
  CommandLineArguments cmd(argc, argv);
  auto q = FrameQueue{cmd.queueSize, cmd.skipIn, cmd.skipOut};

  auto capture = cv::VideoCapture(0);
  if(!capture.isOpened()) {
    throw std::runtime_error("failed to open video capture");
  }

  constexpr int history = 500;
  constexpr bool detectShadows = false; //true;
#ifdef USE_BACKGROUND_KNN
  constexpr double dist2Threshold = 400.0;
  auto backSub = cv::createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows);
#else // USE_BACKGROUND_KNN
  constexpr double varThreshold = 16;
  auto backSub = cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);
#endif // USE_BACKGROUND_KNN

  cv::Mat current;
  cv::Mat foreground;
  const auto morphKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(cmd.morphSize, cmd.morphSize));
  for(;;) {
    // grab next camera frame
    capture >> current;

    // present frame to queue
    q.enqueueMaybe(current);

    // determine foreground mask
    backSub->apply(current, foreground);
    cv::morphologyEx(
      foreground,
      foreground,
      cv::MORPH_CLOSE,
      morphKernel);

    // get a queued (background) frame
    auto queued = q.get().clone();

    // paint current foreground over background
#ifdef DEBUG_BACKGROUND
    queued.setTo(cv::Scalar::all(0));
#endif // DEBUG_BACKGROUND
    current.copyTo(queued, foreground);

    // display
    cv::imshow("Display window", queued);
    if(cv::waitKey(cmd.frameInterval) == 'q') {
      exit(EXIT_SUCCESS);
    }
  }

  return EXIT_SUCCESS;
} catch(const std::exception& e) {
  std::cout << e.what();
}
