#ifndef FRAME_QUEUE_H
#define FRAME_QUEUE_H

#include <opencv2/core/mat.hpp>

#include <iostream>
#include <vector>

struct FrameQueue
{
  using Frame = cv::Mat;

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

  template<typename Fn>
  void enqueueMaybe(Fn fn)
  {
    if((idxIn++ % skipIn) != 0) {
      return;
    }
    if(storage.size() >= maxSize) {
      storage.erase(storage.begin());
      --idxOut; // rollover is considered in get()
    }
    storage.push_back(fn());
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

#endif // FRAME_QUEUE_H
