#include "camera.hpp"
#include "feature_manager.hpp"
#include "frame.hpp"
#include "trajectory.hpp"
#include "visualiser.hpp"
#include "landmark_map.hpp"
#include "bundle_adjuster.hpp"

#include <iomanip>
#include <deque>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <unordered_map>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>
#include <iostream>

/**
 * @brief Navio V2 — RGB-D Visual Odometry with Local Bundle Adjustment
 *
 * Pipeline overview
 * -----------------
 * 1. Load camera intrinsics from YAML configuration.
 * 2. Seed the landmark map from frame 0 using the V2 decoupled architecture.
 * 3. For each subsequent frame:
 *    a. Extract ORB features.
 *    b. Match read-only against the local BA window landmarks.
 *    c. Solve absolute pose with solvePnPRansac.
 *    d. On keyframe: write RANSAC inliers to map, run Local BA.
 * 4. Save trajectory in TUM format for evaluation.
 *
 * Key architectural decisions
 * ---------------------------
 * - frame_poses uses Eigen::Matrix4d (Twc) as the global pose standard.
 *   BundleAdjuster handles the internal Twc->Tcw conversion for Ceres.
 *
 * - Map writes (updateMap) happen ONLY on keyframes after RANSAC confirmation.
 *   This prevents outlier observations from poisoning landmark quality.
 *
 * - Matching is restricted to the local BA window via getLandmarkIdsInFrames.
 *   This keeps descriptor matching O(k) rather than O(N) as the map grows.
 *
 * - The sliding window is capped at 10 keyframes. BA runs in bounded time
 *   regardless of sequence length — essential for real-time operation.
 *

 */
int main() {
    try {

        // =====================================================================
        // 1. Load Camera Calibration
        // =====================================================================
        // Camera parameters are stored in YAML rather than hardcoded to allow
        // the same binary to run on different sensors without recompilation.

        YAML::Node config = YAML::LoadFile("../config/camera_params.yaml");

        const double fx          = config["fx"].as<double>();
        const double fy          = config["fy"].as<double>();
        const double cx          = config["cx"].as<double>();
        const double cy          = config["cy"].as<double>();
        const double depth_scale = config["depth_scale"].as<double>();
        const double k1          = config["k1"].as<double>();
        const double k2          = config["k2"].as<double>();
        const double p1          = config["p1"].as<double>();
        const double p2          = config["p2"].as<double>();
        const double k3          = config["k3"].as<double>();

        navio::Camera camera(fx, fy, cx, cy, depth_scale, k1, k2, p1, p2, k3);

        // =====================================================================
        // 2. Initialise V2 Core Objects
        // =====================================================================

        navio::FeatureManager featureManager;
        navio::LandmarkMap    map;
        navio::BundleAdjuster bundle_adjuster;
        navio::Trajectory     trajectory;
        navio::Visualiser     visualiser;

        // frame_poses stores Twc (camera-to-world) as Matrix4d.
        // Using Matrix4d rather than Isometry3d here because BundleAdjuster
        // works with raw matrix math internally — no semantic benefit to
        // Isometry3d at the BA boundary.
        std::unordered_map<int, Eigen::Matrix4d> frame_poses;

        // Sliding window of keyframe IDs for Local BA.
        // deque allows O(1) front removal when the window slides forward.
        std::deque<int> current_ba_window;

        // Keyframe selection thresholds.
        // A new keyframe is selected when the camera moves more than
        // TRANSLATION_THRESH metres OR rotates more than ROTATION_THRESH
        // radians from the last keyframe. These values balance map density
        // against BA computation cost.
        const double    TRANSLATION_THRESH{0.1};  // 10 cm
        const double    ROTATION_THRESH{0.087};   // ~5 degrees in radians
        Eigen::Matrix4d last_keyframe_pose = Eigen::Matrix4d::Identity();

        // =====================================================================
        // 3. Load TUM RGB-D Dataset
        // =====================================================================
        // The associated.txt file provides timestamp-synchronised RGB-depth
        // pairs. Without association, RGB and depth timestamps would not
        // align — causing depth values to correspond to the wrong RGB frame.

        std::fstream assoc_file("../rgbd_dataset_freiburg1_xyz/associated.txt");
        if (!assoc_file.is_open()) {
            throw std::runtime_error("Cannot open associated.txt");
        }

        std::vector<std::string> rgb_paths;
        std::vector<std::string> depth_paths;
        std::vector<double>      timestamps;
        std::string              line;

        while (std::getline(assoc_file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string ts_rgb, rgb_file, ts_depth, depth_file;
            if (ss >> ts_rgb >> rgb_file >> ts_depth >> depth_file) {
                rgb_paths.push_back(
                    "../rgbd_dataset_freiburg1_xyz/" + rgb_file);
                depth_paths.push_back(
                    "../rgbd_dataset_freiburg1_xyz/" + depth_file);
                timestamps.push_back(std::stod(ts_rgb));
            }
        }
        std::cout << "Loaded " << rgb_paths.size() << " frame pairs.\n";

        // =====================================================================
        // 4. Bootstrap Frame 0 — Seed the Landmark Map
        // =====================================================================
        // Frame 0 is defined as the world origin (Identity pose).
        // updateMap is called with empty matches and empty inliers so that
        // Write Phase 1 (observations) is skipped and only Write Phase 2
        // (new landmark seeding) runs. This cleanly populates the map with
        // the first set of 3D landmarks in world coordinates.

        cv::Mat rgb0   = cv::imread(rgb_paths[0],   cv::IMREAD_COLOR);
        cv::Mat depth0 = cv::imread(depth_paths[0], cv::IMREAD_ANYDEPTH);
        navio::Frame frame0(0, timestamps[0], rgb0, depth0);

        frame_poses[0] = Eigen::Matrix4d::Identity();
        current_ba_window.push_back(0);

        std::vector<cv::KeyPoint> kp0;
        cv::Mat desc0;
        featureManager.extractFeatures(frame0, kp0, desc0);
        featureManager.updateMap(frame0, kp0, desc0, {},
                                 cv::Mat(), Eigen::Matrix4d::Identity(),
                                 camera, map);

        std::cout << "Map seeded with " << map.size() << " landmarks.\n";

        // =====================================================================
        // 5. Main Odometry Loop
        // =====================================================================

        const auto start_time = std::chrono::high_resolution_clock::now();
        int total_frames_processed{0};

        for (std::size_t i = 1; i < rgb_paths.size(); ++i) {

            ++total_frames_processed;

            cv::Mat rgb_curr   = cv::imread(rgb_paths[i],   cv::IMREAD_COLOR);
            cv::Mat depth_curr = cv::imread(depth_paths[i], cv::IMREAD_ANYDEPTH);
            if (rgb_curr.empty() || depth_curr.empty()) continue;

            navio::Frame frame_curr(
                static_cast<int>(i), timestamps[i], rgb_curr, depth_curr);

            // -----------------------------------------------------------------
            // A. Feature Extraction and Local Map Matching (Read-Only)
            //
            // Matching is restricted to landmarks in the current BA window.
            // getLandmarkIdsInFrames uses the reverse index for O(k) lookup.
            // This prevents the O(N) degradation that caused cascade failure
            // in the naive global matching approach.
            // -----------------------------------------------------------------
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            featureManager.extractFeatures(frame_curr, keypoints, descriptors);

            const std::vector<int> window_vec(
                current_ba_window.begin(), current_ba_window.end());
            const std::vector<int> active_ids =
                map.getLandmarkIdsInFrames(window_vec);

            auto matches = featureManager.matchLocalMap(
                descriptors, keypoints, map, active_ids, camera);

            if (matches.size() < 15) {
                std::cerr << "[Warning] Tracking lost at frame " << i
                          << ". Insufficient local matches.\n";
                continue;
            }

            // -----------------------------------------------------------------
            // B. Build PnP Correspondences
            //
            // 3D positions come from the landmark map (world frame).
            // 2D pixels come from matchLocalMap (already undistorted).
            // solvePnPRansac expects distortion coefficients — passing them
            // here even though pixels are already undistorted is safe because
            // OpenCV handles the zero-distortion case correctly.
            // -----------------------------------------------------------------
            std::vector<cv::Point3f> object_points;
            std::vector<cv::Point2f> image_points;

            for (const auto& match : matches) {
                try {
                    const navio::Landmark& lm =
                        map.getLandmark(match.landmark_id);
                    object_points.emplace_back(
                        lm.position_3d.x(),
                        lm.position_3d.y(),
                        lm.position_3d.z());
                    image_points.emplace_back(
                        match.pixel_undist.x(),
                        match.pixel_undist.y());
                } catch (...) { continue; }
            }

            // -----------------------------------------------------------------
            // C. Absolute Pose Estimation with RANSAC Guards
            //
            // solvePnPRansac recovers the camera pose directly in world
            // coordinates by matching 3D world points against 2D observations.
            // This is absolute localisation — not relative to the previous
            // frame — which is why drift correction via BA is effective.
            //
            // Two guards protect against bad pose estimates:
            // Guard 1 — RANSAC inlier count: reject if fewer than 15 points
            //   agree with the estimated pose. Low inlier count indicates
            //   the map does not explain the current frame reliably.
            // Guard 2 — Physics check: reject poses that imply the camera
            //   moved more than 1 metre since the last keyframe. At 30fps
            //   this would require superhuman speed — the estimate is wrong.
            // -----------------------------------------------------------------
            cv::Mat rvec, tvec, inliers;
            bool pnp_success = cv::solvePnPRansac(
                object_points, image_points,
                camera.getIntrinsicMatrix(), camera.getDistCoeffs(),
                rvec, tvec, false, 100, 2.5, 0.9,
                inliers, cv::SOLVEPNP_ITERATIVE);

            if (!pnp_success || inliers.empty() || inliers.total() < 15) {
                std::cerr << "[Warning] PnP failed at frame " << i
                          << ". Skipping.\n";
                continue;
            }

            // Convert OpenCV rotation vector and translation to Eigen Matrix4d
            cv::Mat R_cv;
            cv::Rodrigues(rvec, R_cv);
            Eigen::Matrix3d R_eigen;
            Eigen::Vector3d t_eigen;
            cv::cv2eigen(R_cv,  R_eigen);
            cv::cv2eigen(tvec,  t_eigen);

            // Build Tcw then invert to Twc — solvePnP returns world-to-camera
            Eigen::Matrix4d T_cw = Eigen::Matrix4d::Identity();
            T_cw.block<3,3>(0,0) = R_eigen;
            T_cw.block<3,1>(0,3) = t_eigen;
            const Eigen::Matrix4d T_wc = T_cw.inverse();

            // Physics guard — compare against last confirmed keyframe pose
            const Eigen::Matrix4d test_motion =
                last_keyframe_pose.inverse() * T_wc;
            if (test_motion.block<3,1>(0,3).norm() > 1.0) {
                std::cerr << "[Warning] Physics violation at frame "
                          << i << ". Skipping.\n";
                continue;
            }

            // Pose is geometrically verified — safe to store
            frame_poses[i] = T_wc;
            trajectory.addAbsolutePose(T_wc);

            // -----------------------------------------------------------------
            // D. Keyframe Selection, Map Writing and Local BA
            //
            // A frame becomes a keyframe when sufficient motion has occurred
            // since the last keyframe. Only keyframes:
            //   - Write observations and new landmarks to the map (updateMap)
            //   - Enter the sliding BA window
            //   - Trigger a Local BA run
            //
            // Non-keyframes contribute only to the trajectory, not the map.
            // This keeps the map sparse and the BA problem well-conditioned.
            // -----------------------------------------------------------------
            const double translation_dist =
                test_motion.block<3,1>(0,3).norm();
            const double trace =
                test_motion.block<3,3>(0,0).trace();
            const double rotation_angle =
                std::acos(std::clamp((trace - 1.0) / 2.0, -1.0, 1.0));

            const bool is_keyframe =
                translation_dist > TRANSLATION_THRESH ||
                rotation_angle   > ROTATION_THRESH;

            if (is_keyframe) {

                // Write: inlier observations + new landmarks using confirmed pose
                featureManager.updateMap(
                    frame_curr, keypoints, descriptors,
                    matches, inliers, T_wc, camera, map);

                current_ba_window.push_back(static_cast<int>(i));
                last_keyframe_pose = T_wc;

                // Slide the window — remove oldest keyframe when full
                if (current_ba_window.size() > 10) {
                    current_ba_window.pop_front();
                }

                // Run Local Bundle Adjustment over the current window
                const std::vector<int> active_window(
                    current_ba_window.begin(), current_ba_window.end());
                bundle_adjuster.optimize(
                    active_window, frame_poses, map, camera);

                // Write the BA-refined pose back into the trajectory
                trajectory.updatePoseAt(
                    static_cast<int>(trajectory.getTrajectory().size() - 1),
                    Eigen::Isometry3d(frame_poses[i]));
            }

            // Map hygiene — cull stale landmarks every 20 frames
            // Prevents unbounded map growth while preserving well-established points
            if (i % 20 == 0) {
                map.cullStaleLandmarks(static_cast<int>(i), 30);
            }

            // -----------------------------------------------------------------
            // E. Visualisation
            // -----------------------------------------------------------------
            cv::Mat traj_image =
                visualiser.drawTrajectory(trajectory.getTrajectory());
            cv::imshow("Navio V2 - Local BA", traj_image);

            if (i % 10 == 0) {
                std::cout << "Frame " << i << "/" << rgb_paths.size()
                          << " | Map: " << map.size()
                          << " | Inliers: " << inliers.total() << "\n";
            }

            if (cv::waitKey(1) == 'q') break;
        }

        // =====================================================================
        // Performance Report
        // =====================================================================

        const auto end_time = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> execution_time =
            end_time - start_time;
        const double total_seconds = execution_time.count();
        const double fps = total_frames_processed / total_seconds;

        std::cout << "\n==================================================\n";
        std::cout << " NAVIO V2 PERFORMANCE REPORT\n";
        std::cout << "==================================================\n";
        std::cout << "Total frames processed : " << total_frames_processed << "\n";
        std::cout << "Total execution time   : " << total_seconds << " s\n";
        std::cout << "Average FPS            : " << fps << " fps\n";
        std::cout << "==================================================\n";

        // =====================================================================
        // 6. Save Trajectory in TUM Format
        // =====================================================================
        // Format: timestamp tx ty tz qx qy qz qw
        // Only frames with a confirmed pose are written — skipped frames
        // (tracking lost or physics violation) are omitted.

        std::ofstream traj_file("../results_v2/estimated_trajectory.txt");
        for (std::size_t i = 0; i < timestamps.size(); ++i) {
            auto it = frame_poses.find(static_cast<int>(i));
            if (it == frame_poses.end()) continue;

            const Eigen::Matrix4d&   Twc = it->second;
            const Eigen::Vector3d    t   = Twc.block<3,1>(0,3);
            const Eigen::Quaterniond q(Twc.block<3,3>(0,0));

            traj_file << std::fixed << std::setprecision(6)
                      << timestamps[i] << " "
                      << t.x() << " " << t.y() << " " << t.z() << " "
                      << q.x() << " " << q.y() << " " << q.z() << " "
                      << q.w() << "\n";
        }
        traj_file.close();
        std::cout << "Trajectory saved to results_v2/estimated_trajectory.txt\n";

    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}