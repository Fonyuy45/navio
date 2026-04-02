#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace navio {

struct Observation {
    int frame_id;
    Eigen::Vector2d pixel_uv;
};

struct Landmark {
    int id;
    Eigen::Vector3d position_3d;  // optimised by BA
    cv::Mat descriptor {};
    std::vector<cv::Mat> all_descriptors {};
    std::vector<Observation> observations;
};

} // namespace navio