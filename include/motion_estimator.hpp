#pragma once

#include "camera.hpp"
#include "feature_manager.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace navio {

class MotionEstimator {
public:
    MotionEstimator();

    Eigen::Isometry3d estimateRelativePose(
        const std::vector<Match3D2D>& correspondences,
        const Camera& camera) const;
};

} // namespace navio