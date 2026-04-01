#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace navio {

struct Observation {
    int frame_id;
    Eigen::Vector2d pixel_uv;
};

struct Landmark {
    int id;
    Eigen::Vector3d position_3d;  // optimised by BA
    std::vector<Observation> observations;
};

} // namespace navio