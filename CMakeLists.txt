cmake_minimum_required(VERSION 3.10)
project(find_circle CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# CUDA 由 Python OpenCV 按需调用，CPU 光流复用 video 模块的金字塔。
find_package(OpenCV REQUIRED COMPONENTS core imgproc video)

set(EDLIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ED_Lib)
add_library(edcircles SHARED
  edcircles/edcircles_c.cpp
  ${EDLIB_DIR}/ED.cpp
  ${EDLIB_DIR}/EDPF.cpp
  ${EDLIB_DIR}/EDLines.cpp
  ${EDLIB_DIR}/EDCircles.cpp
  ${EDLIB_DIR}/EDColor.cpp
  ${EDLIB_DIR}/NFA.cpp)
target_include_directories(edcircles PRIVATE ${EDLIB_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(edcircles PRIVATE ${OpenCV_LIBS})
set_target_properties(edcircles PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ringdet/_lib
  CXX_VISIBILITY_PRESET hidden)
