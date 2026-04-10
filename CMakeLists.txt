# Minimum CMake version required to process this file
cmake_minimum_required(VERSION 3.22.1)

# Project name and version — VERSION keyword must be uppercase
project(navio VERSION 1.0)

# Release build with O2 optimisation — essential for real-time performance.
# O2 enables vectorisation and inlining without the aggressive transformations
# of O3 that can occasionally produce unexpected behaviour in numeric code.
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# C++17 standard — required for structured bindings, std::clamp, if constexpr
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # Enforce strict standard, disable GNU extensions

# Dependencies
find_package(Eigen3    REQUIRED)  # Linear algebra and SE(3) transforms
find_package(OpenCV    REQUIRED)  # Feature detection, PnP, image I/O
find_package(yaml-cpp  REQUIRED)  # Camera parameter loading from YAML
find_package(Ceres     REQUIRED)  # Nonlinear least squares for Bundle Adjustment

# Executable target — all source files listed explicitly.
# CMake only recompiles files that have changed (incremental build).
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/camera.cpp
    src/feature_manager.cpp
    src/frame.cpp
    src/trajectory.cpp
    src/motion_estimator.cpp
    src/visualiser.cpp
    src/landmark_map.cpp
    src/bundle_adjuster.cpp
)

# Include paths — PRIVATE means only this target sees these directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# Link libraries — Eigen3::Eigen carries its own include paths automatically
target_link_libraries(${PROJECT_NAME} PRIVATE
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp
    Ceres::ceres
)