#include "frame_queue.h"

#include <opencv2/bgsegm.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
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
  int morphSize;
  int frameInterval;

  CommandLineArguments(int argc, char** argv)
  {
    const char* keys = R"(
{help h usage ? |    | print this message }
{queue_size     | 30 | number of frames to queue }
{skip_in        |  1 | number of queue frames to skip from input }
{skip_out       |  3 | number of queue frames to skip during output (can be negative) }
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
    if(morphSize <= 0) {
      throw std::invalid_argument("'morph_size' must be >0");
    }
    if(frameInterval < 0) {
      throw std::invalid_argument("'frame_interval' must be >=0");
    }
  }
};

cv::Ptr<cv::BackgroundSubtractor> createBackgroundSubtractor()
{
#if BACKGROUND_SUBTRACTOR == BACKGROUND_SUBTRACTOR_MOG2

  constexpr int history = 500;
  constexpr double varThreshold = 16;
  constexpr bool detectShadows = true;
  return cv::createBackgroundSubtractorMOG2(history, varThreshold, detectShadows);

#elif BACKGROUND_SUBTRACTOR == BACKGROUND_SUBTRACTOR_KNN

  constexpr int history = 500;
  constexpr double dist2Threshold = 400.0;
  constexpr bool detectShadows = true;
  return cv::createBackgroundSubtractorKNN(history, dist2Threshold, detectShadows);

#elif BACKGROUND_SUBTRACTOR == BACKGROUND_SUBTRACTOR_GMG

  constexpr int initializationFrames = 30; //120;
  constexpr double decisionThreshold = 0.8;
  return cv::bgsegm::createBackgroundSubtractorGMG(initializationFrames, decisionThreshold);

#elif BACKGROUND_SUBTRACTOR == BACKGROUND_SUBTRACTOR_MOG

  constexpr int history = 200;
  constexpr int nmixtures = 5;
  constexpr double backgroundRatio = 0.7;
  constexpr double noiseSigma = 0;
  return cv::bgsegm::createBackgroundSubtractorMOG(history, nmixtures, backgroundRatio, noiseSigma);

#elif BACKGROUND_SUBTRACTOR == BACKGROUND_SUBTRACTOR_CNT

  constexpr int minPixelStability = 15;
  constexpr bool useHistory = true;
  constexpr int maxPixelStability = 15 * 60;
  constexpr bool isParallel = true;
  return cv::bgsegm::createBackgroundSubtractorCNT(minPixelStability, useHistory, maxPixelStability, isParallel);

#elif BACKGROUND_SUBTRACTOR == BACKGROUND_SUBTRACTOR_GSOC

  constexpr int mc = cv::bgsegm::LSBP_CAMERA_MOTION_COMPENSATION_NONE;
  constexpr int nSamples = 20;
  constexpr float replaceRate = 0.003f;
  constexpr float propagationRate = 0.01f;
  constexpr int hitsThreshold = 32;
  constexpr float alpha = 0.01f;
  constexpr float beta = 0.0022f;
  constexpr float blinkingSupressionDecay = 0.1f;
  constexpr float blinkingSupressionMultiplier = 0.1f;
  constexpr float noiseRemovalThresholdFacBG = 0.0004f;
  constexpr float noiseRemovalThresholdFacFG = 0.0008f;
  return cv::bgsegm::createBackgroundSubtractorGSOC(
    mc,
    nSamples,
    replaceRate,
    propagationRate,
    hitsThreshold,
    alpha,
    beta,
    blinkingSupressionDecay,
    blinkingSupressionMultiplier,
    noiseRemovalThresholdFacBG,
    noiseRemovalThresholdFacFG);

#elif BACKGROUND_SUBTRACTOR == BACKGROUND_SUBTRACTOR_LSBP

  constexpr int mc = cv::bgsegm::LSBP_CAMERA_MOTION_COMPENSATION_NONE;
  constexpr int nSamples = 20;
  constexpr int LSBPRadius = 16;
  constexpr float Tlower = 2.0f;
  constexpr float Tupper = 32.0f;
  constexpr float Tinc = 1.0f;
  constexpr float Tdec = 0.05f;
  constexpr float Rscale = 10.0f;
  constexpr float Rincdec = 0.005f;
  constexpr float noiseRemovalThresholdFacBG = 0.0004f;
  constexpr float noiseRemovalThresholdFacFG = 0.0008f;
  constexpr int LSBPthreshold = 8;
  constexpr int minCount = 2;
  return cv::bgsegm::createBackgroundSubtractorLSBP(
    mc,
    nSamples,
    LSBPRadius,
    Tlower,
    Tupper,
    Tinc,
    Tdec,
    Rscale,
    Rincdec,
    noiseRemovalThresholdFacBG,
    noiseRemovalThresholdFacFG,
    LSBPthreshold,
    minCount);

#else
  #error invalid BACKGROUND_SUBTRACTOR
#endif
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

  auto backSub = createBackgroundSubtractor();

  cv::Mat current;
  cv::Mat foreground;
  const auto morphKernel = cv::getStructuringElement(
    cv::MORPH_ELLIPSE,
    cv::Size(cmd.morphSize, cmd.morphSize));
  for(;;) {
    // grab next camera frame
    capture >> current;

    // present frame to queue
    q.enqueueMaybe(
      [&current]() -> cv::Mat {
        return current.clone();
      });

    // determine foreground mask
    backSub->apply(current, foreground);

    // postprocess foreground mask
    static auto anchor = cv::Point(-1, -1);
    constexpr int iterations = 1;
    constexpr int borderType = cv::BORDER_CONSTANT;
    static auto&& borderValue = cv::morphologyDefaultBorderValue();
    cv::morphologyEx(
      foreground,
      foreground,
      cv::MORPH_CLOSE,
      morphKernel,
      anchor,
      iterations,
      borderType,
      borderValue);

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
