#pragma once
#include "camera.hpp"
#include "landmark_map.hpp"
#include <Eigen/Dense>
#include <unordered_map>
#include <vector>

namespace navio {

/**
 * @brief Local Bundle Adjuster using Ceres Solver.
 *
 * Optimises a sliding window of camera poses and their associated 3D
 * landmarks jointly, minimising the sum of squared reprojection errors
 * under a Huber robust kernel.
 *
 * Pose representation
 * -------------------
 * Frame poses are stored externally as Eigen::Matrix4d (Twc,
 * camera-to-world). Inside Ceres, each pose is converted to a 6-element
 * flat array {r0,r1,r2,t0,t1,t2} representing Tcw (world-to-camera) in
 * angle-axis + translation form. 
 *
 * Gauge freedom
 * -------------
 * The first frame in local_window_frames is held constant (parameter
 * block set to Constant). Without this, the system is under-constrained
 * and Ceres will drift the entire map rigidly while minimising nothing.
 */
class BundleAdjuster {
public:

    BundleAdjuster() = default;

    /**
     * @brief Runs local bundle adjustment over a window of keyframes.
     *
     * Mutates frame_poses (optimised Twc) and map (optimised positions)
     * in-place. Both are valid and self-consistent after the call.
     *
     * @param local_window_frames  Ordered list of frame IDs to optimise.
     * First frame is used as the gauge anchor.
     * @param frame_poses          Map of frame_id -> Twc (camera-to-world).
     * Modified in-place with optimised poses.
     * @param map                  Global landmark map. Landmark positions
     * are updated via updatePosition().
     * @param camera               Camera intrinsics for reprojection error.
     */
    void optimize(
        const std::vector<int>&                          local_window_frames,
        std::unordered_map<int, Eigen::Matrix4d>&        frame_poses, // FIX: Standardized to Matrix4d
        LandmarkMap&                                     map,
        const Camera&                                    camera);
};

} // namespace navio