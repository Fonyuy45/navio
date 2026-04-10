#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace navio {

/**
 * @brief Stores and manages the sequence of camera poses over time.
 *
 * Trajectory supports two fundamentally different pose accumulation modes
 * corresponding to the two pipeline versions:
 *
 * V1 — Relative accumulation (updatePose)
 * ----------------------------------------
 * The MotionEstimator produces frame-to-frame relative transforms. Each new
 * pose is computed by multiplying the previous global pose by the relative
 * transform: T_global_new = T_global_prev * T_relative. The trajectory is
 * built as a chain where each pose depends on all previous estimates.
 *
 * V2 — Absolute storage (addAbsolutePose)
 * ----------------------------------------
 * solvePnPRansac against the global landmark map produces direct absolute
 * poses — where the camera is in world coordinates right now, independently
 * of any previous frame. These are stored directly with no accumulation.
 * After Bundle Adjustment refines a pose, updatePoseAt corrects it in place.
 *
 * Memory alignment
 * ----------------
 * Eigen::Isometry3d contains fixed-size matrices that require 16-byte memory
 * alignment for SIMD operations. std::vector uses Eigen::aligned_allocator
 * to guarantee this alignment — without it, operations on the stored poses
 * may crash or produce wrong results on some platforms.
 */
class Trajectory {
public:
    /**
     * @brief Constructs a Trajectory initialised at the world origin.
     *
     * The identity pose is inserted as the starting point — the camera
     * begins at position (0,0,0) with no rotation relative to the world.
     */
    Trajectory();

    /**
     * @brief Appends a new pose by composing it with the current last pose.
     *
     * Used in V1 relative odometry. The relative transform is left-multiplied
     * by the current global pose to produce the new global pose:
     *   T_new = T_current * T_relative
     *
     * @param pose  Relative transform from the previous frame to the current.
     */
    void updatePose(const Eigen::Isometry3d& pose);

    /**
     * @brief Returns the most recently stored pose.
     */
    Eigen::Isometry3d getCurrentPose() const;

    /**
     * @brief Returns a const reference to the full pose sequence.
     *
     * Used by Visualiser to draw the trajectory and by the evaluation
     * pipeline to export poses in TUM format.
     */
    const std::vector<Eigen::Isometry3d,
                       Eigen::aligned_allocator<Eigen::Isometry3d>>&
    getTrajectory() const;

    /**
     * @brief Overwrites an existing pose at a specific index.
     *
     * Used in V2 after Bundle Adjustment refines a keyframe pose. The
     * optimised pose from frame_poses is written back into the trajectory
     * at the correct position.
     *
     * @param index  Zero-based index into the pose list.
     * @param pose   The refined absolute pose to store.
     */
    void updatePoseAt(int index, const Eigen::Isometry3d& pose);

    /**
     * @brief Appends a direct absolute pose without accumulation as was done in v1.
     *
     * Used in V2 where solvePnPRansac produces globally referenced poses
     * that do not need to be composed with previous estimates.
     *
     * @param pose  Absolute camera-to-world transform (Twc).
     */
    void addAbsolutePose(const Eigen::Isometry3d& pose);

    /**
     * @brief Convenience overload accepting Eigen::Matrix4d.
     *
     * Converts the raw 4x4 matrix to Isometry3d before storing. Provided
     * because frame_poses in main.cpp uses Matrix4d and forcing the caller
     * to convert manually would add noise to the call site.
     *
     * @param pose  Absolute camera-to-world transform as a 4x4 matrix.
     */
    void addAbsolutePose(const Eigen::Matrix4d& pose);

private:
    std::vector<Eigen::Isometry3d,
                Eigen::aligned_allocator<Eigen::Isometry3d>> pose_list_;
};

} // namespace navio