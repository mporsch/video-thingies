cmake_minimum_required(VERSION 3.5.0)

project(video-thingies CXX)

find_package(OpenCV REQUIRED)

option(USE_BACKGROUND_KNN "" TRUE)
option(DEBUG_QUEUE "" FALSE)
option(DEBUG_BACKGROUND "" FALSE)

add_executable(background_loop background_loop.cpp)
target_compile_features(background_loop PRIVATE cxx_std_17)
target_compile_definitions(background_loop PRIVATE
  $<$<BOOL:${USE_BACKGROUND_KNN}>:USE_BACKGROUND_KNN>
  $<$<BOOL:${DEBUG_QUEUE}>:DEBUG_QUEUE>
  $<$<BOOL:${DEBUG_BACKGROUND}>:DEBUG_BACKGROUND>
)
target_include_directories(background_loop PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(background_loop PRIVATE ${OpenCV_LIBS})
