#include "bundle_adjuster.hpp"
#include "reprojection_error.hpp"
#include <ceres/ceres.h>
#include <iostream>

namespace navio {

// =============================================================================
// Internal helpers
// =============================================================================

/**
 * @brief Converts an Eigen::Matrix4d Twc into a flat 6-element Ceres pose.
 *
 * Ceres needs {r0,r1,r2,t0,t1,t2} in Tcw (world-to-camera) convention.
 */
static void matrixToCeresPose(const Eigen::Matrix4d& T_wc,
                              std::vector<double>&   out)
{
    // Invert to world-to-camera — this is what ReprojectionError expects.
    const Eigen::Matrix4d T_cw = T_wc.inverse();

    // Extract rotation matrix and translation vector
    const Eigen::AngleAxisd aa(T_cw.block<3,3>(0,0));
    const Eigen::Vector3d   r = aa.axis() * aa.angle();
    const Eigen::Vector3d   t = T_cw.block<3,1>(0,3);

    out = {r.x(), r.y(), r.z(), t.x(), t.y(), t.z()};
}

/**
 * @brief Converts an optimised 6-element Ceres pose back to Eigen::Matrix4d.
 */
static Eigen::Matrix4d ceresPoseToMatrix(const std::vector<double>& pose_array)
{
    const Eigen::Vector3d r_vec(pose_array[0], pose_array[1], pose_array[2]);
    const Eigen::Vector3d t_vec(pose_array[3], pose_array[4], pose_array[5]);

    const double angle = r_vec.norm();

    Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();

    if (angle < 1e-10) {
        // Near-zero rotation — identity rotation, avoid division by zero.
        T_cw.block<3,3>(0,0) = Eigen::Matrix3d::Identity();
    } else {
        T_cw.block<3,3>(0,0) = Eigen::AngleAxisd(angle, r_vec / angle).toRotationMatrix();
    }

    T_cw.block<3,1>(0,3) = t_vec;

    // Invert back to camera-to-world for storage in frame_poses.
    return T_cw.inverse();
}

// =============================================================================
// BundleAdjuster::optimize
// =============================================================================

void BundleAdjuster::optimize(
    const std::vector<int>&                     local_window_frames,
    std::unordered_map<int, Eigen::Matrix4d>&   frame_poses, // FIX: Matrix4d
    LandmarkMap&                                map,
    const Camera&                               camera)
{
    if (local_window_frames.size() < 2) return;

    ceres::Problem problem;

    // -------------------------------------------------------------------------
    // Step 1: Convert all window poses into flat Ceres arrays (Tcw convention)
    // -------------------------------------------------------------------------
    std::unordered_map<int, std::vector<double>> ceres_poses;
    ceres_poses.reserve(local_window_frames.size());

    for (int frame_id : local_window_frames) {
        auto it = frame_poses.find(frame_id);
        if (it == frame_poses.end()) continue;

        // FIX: Using the new Matrix4d converter
        matrixToCeresPose(it->second, ceres_poses[frame_id]);
    }

    // -------------------------------------------------------------------------
    // Step 2: Extract camera intrinsics once
    // -------------------------------------------------------------------------
    const cv::Mat  K  = camera.getIntrinsicMatrix();
    const double   fx = K.at<double>(0, 0);
    const double   fy = K.at<double>(1, 1);
    const double   cx = K.at<double>(0, 2);
    const double   cy = K.at<double>(1, 2);

    // -------------------------------------------------------------------------
    // Step 3: Build the factor graph
    // -------------------------------------------------------------------------
    std::vector<Landmark> active_landmarks = map.getLandmarksInFrames(local_window_frames);

    std::vector<Eigen::Vector3d> optimised_positions;
    optimised_positions.reserve(active_landmarks.size());

    for (Landmark& lm : active_landmarks) {
        optimised_positions.push_back(lm.position_3d);
        double* point_ptr = optimised_positions.back().data();

        for (const Observation& obs : lm.observations) {
            // Only add a residual if this observation's frame is in the window.
            auto pose_it = ceres_poses.find(obs.frame_id);
            if (pose_it == ceres_poses.end()) continue;

            // FIX: Use pixel_undist instead of pixel_uv
            ceres::CostFunction* cost = ReprojectionError::Create(
                obs.pixel_undist.x(), obs.pixel_undist.y(),
                fx, fy, cx, cy);

            problem.AddResidualBlock(
                cost,
                new ceres::HuberLoss(1.0),
                pose_it->second.data(),   // camera_pose[6]
                point_ptr);               // point_3d[3]
        }
    }

    // -------------------------------------------------------------------------
    // Step 4: Fix the first frame — gauge anchor
    // -------------------------------------------------------------------------
    const int anchor_id = local_window_frames.front();
    auto anchor_it = ceres_poses.find(anchor_id);

    if (anchor_it != ceres_poses.end() &&
        problem.HasParameterBlock(anchor_it->second.data()))
    {
        problem.SetParameterBlockConstant(anchor_it->second.data());
    }

    // -------------------------------------------------------------------------
    // Step 5: Solve
    // -------------------------------------------------------------------------
    ceres::Solver::Options options;
    options.linear_solver_type         = ceres::DENSE_SCHUR;
    options.max_num_iterations         = 5;

    // Toggle to true if you want to see Ceres output in the terminal
    constexpr bool DEBUG_BA = false;
    options.minimizer_progress_to_stdout = DEBUG_BA;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if constexpr (DEBUG_BA) {
        std::cout << summary.BriefReport() << "\n";
    }

    // -------------------------------------------------------------------------
    // Step 6: Write optimised poses back to frame_poses
    // -------------------------------------------------------------------------
    for (int frame_id : local_window_frames) {
        auto it = ceres_poses.find(frame_id);
        if (it == ceres_poses.end()) continue;

        // FIX: Using the new Matrix4d converter
        frame_poses[frame_id] = ceresPoseToMatrix(it->second);
    }

    // -------------------------------------------------------------------------
    // Step 7: Write optimised landmark positions back to the map
    // -------------------------------------------------------------------------
    for (std::size_t i = 0; i < active_landmarks.size(); ++i) {
        map.updatePosition(active_landmarks[i].id, optimised_positions[i]);
    }
}

} // namespace navio