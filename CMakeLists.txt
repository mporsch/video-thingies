cmake_minimum_required(VERSION 3.5.0)

project(video-thingies CXX)

find_package(OpenCV REQUIRED)

set(BACKGROUND_SUBTRACTORS "MOG2;KNN;GMG;MOG;CNT;GSOC;LSBP")
set(BACKGROUND_SUBTRACTOR "KNN" CACHE STRING "background subtractor implementation to use")
set_property(CACHE BACKGROUND_SUBTRACTOR PROPERTY STRINGS ${BACKGROUND_SUBTRACTORS})
list(FIND BACKGROUND_SUBTRACTORS ${BACKGROUND_SUBTRACTOR} BACKGROUND_SUBTRACTOR_INDEX)
if(BACKGROUND_SUBTRACTOR_INDEX EQUAL -1)
  message(FATAL_ERROR "BACKGROUND_SUBTRACTOR must be one of ${BACKGROUND_SUBTRACTORS}")
endif()

option(DEBUG_QUEUE "" FALSE)
option(DEBUG_BACKGROUND "" FALSE)

add_executable(background_loop background_loop.cpp)
target_compile_features(background_loop PRIVATE cxx_std_11)
target_compile_definitions(background_loop PRIVATE
  BACKGROUND_SUBTRACTOR_MOG2=0
  BACKGROUND_SUBTRACTOR_KNN=1
  BACKGROUND_SUBTRACTOR_GMG=2
  BACKGROUND_SUBTRACTOR_MOG=3
  BACKGROUND_SUBTRACTOR_CNT=4
  BACKGROUND_SUBTRACTOR_GSOC=5
  BACKGROUND_SUBTRACTOR_LSBP=6
  BACKGROUND_SUBTRACTOR=${BACKGROUND_SUBTRACTOR_INDEX}
  $<$<BOOL:${DEBUG_QUEUE}>:DEBUG_QUEUE>
  $<$<BOOL:${DEBUG_BACKGROUND}>:DEBUG_BACKGROUND>
)
target_include_directories(background_loop PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(background_loop PRIVATE ${OpenCV_LIBS})

add_executable(flow_loop flow_loop.cpp)
target_compile_features(flow_loop PRIVATE cxx_std_11)
target_compile_definitions(flow_loop PRIVATE
  $<$<BOOL:${DEBUG_QUEUE}>:DEBUG_QUEUE>
)
target_include_directories(flow_loop PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(flow_loop PRIVATE ${OpenCV_LIBS})
