cmake_minimum_required(VERSION 3.24.2)
project(KalmanFilter)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SOURCE_FILES KalmanFilter.cpp KalmanMain.cpp Reportable.cpp)
add_executable(KalmanMain ${SOURCE_FILES} Reportable.cpp Reportable.hpp)
