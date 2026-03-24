# Navio

RGB-D Visual Odometry pipeline in C++ using Eigen and OpenCV.

## Overview
Navio estimates camera ego-motion from sequential RGB-D frames using 
feature-based motion estimation. Evaluated on the TUM RGB-D benchmark dataset.

## Dependencies
- CMake 3.22+
- Eigen3
- OpenCV 4.x

## Build
```bash
mkdir build && cd build
cmake ..
make
```

## Status
Work in progress — built as a learning project for RGB-D VO fundamentals
prior to PhD research in robotics, 3D reconstruction, and surgical guidance.
