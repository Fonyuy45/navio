#pragma once

#include <Eigen/Core>

namespace navio {

class Camera {
public:
    Camera(double fx, double fy, double cx, double cy, double depth_scale);

    // 3D to 2D
    Eigen::Vector2d project(const Eigen::Vector3d& point) const;

    // 2D to 3D
    Eigen::Vector3d unproject(const Eigen::Vector2d& pixel, double depth) const;

private:
    double fx_{};
    double fy_{};
    double cx_{};
    double cy_{};
    double depth_scale_{};
};

}