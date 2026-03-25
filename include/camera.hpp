#pragma once

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace navio {

class Camera {
public:
    // Updated constructor to include distortion coefficients (defaulted to 0)
    Camera(double fx, double fy, double cx, double cy, double depth_scale,
           double k1 = 0.0, double k2 = 0.0, double p1 = 0.0, double p2 = 0.0, double k3 = 0.0);

    // 3D to 2D
    Eigen::Vector2d project(const Eigen::Vector3d& point) const;

    // 2D to 3D
    Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth) const;

    // Getters
    cv::Mat getIntrinsicMatrix() const;  
    cv::Mat getDistCoeffs() const;

private:
    double fx_{};
    double fy_{};
    double cx_{};
    double cy_{};
    double depth_scale_{};

    // Distortion parameters
    double k1_{}, k2_{}, p1_{}, p2_{}, k3_{};
};

} // namespace navio