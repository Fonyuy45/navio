#include "motion_estimator.hpp"
#include "feature_manager.hpp"
#include <iostream>

namespace navio {
    

MotionEstimator::MotionEstimator() {}

/**
 * @brief Estimates the rigid body transformation (pose) between two frames.
 * * Takes 3D points from the previous frame (Frame A) and their matched 2D pixel 
 * coordinates in the current frame (Frame B), and computes the camera's motion 
 * using Perspective-n-Point (PnP) inside a RANSAC scheme.
 * * @param correspondences A vector of Match3D2D pairs connecting Frame A and Frame B.
 * @param camera The Camera object containing intrinsic and distortion parameters.
 * @return Eigen::Isometry3d The Camera-to-World transformation matrix. 
 * Returns Identity if motion estimation fails or is deemed physically impossible.
 */
Eigen::Isometry3d MotionEstimator::estimateRelativePose(const std::vector<Match3D2D>& correspondences, const Camera& camera) const {

    constexpr bool DEBUG_MOTION = false;

    // 1. Minimum Points Check
    // PnP mathematically requires at least 4 points. 
    if (correspondences.size() < 4) {
        return Eigen::Isometry3d::Identity();
    }

    // 2. Unpack Correspondences into OpenCV Formats
    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> points2D;
    
    // reserve() prevents memory reallocation during the loop, speeding up the code slightly
    points3D.reserve(correspondences.size());
    points2D.reserve(correspondences.size());

    for (const auto& match : correspondences) {
        points3D.push_back(cv::Point3f(
            static_cast<float>(match.point3d_A.x()),
            static_cast<float>(match.point3d_A.y()),
            static_cast<float>(match.point3d_A.z())));

        points2D.push_back(cv::Point2f(
            static_cast<float>(match.point2d_B.x()),
            static_cast<float>(match.point2d_B.y())));
    }

    // 3. Retrieve Camera Intrinsics
    cv::Mat K = camera.getIntrinsicMatrix();  
    cv::Mat distCoeffs = camera.getDistCoeffs();

    // 4. Solve PnP with RANSAC
    cv::Mat rvec, tvec;
    std::vector<int> inliers; // Will hold the indices of the points that actually fit the model
    
    // Parameters: 3D pts, 2D pts, Intrinsics, Distortion, rvec out, tvec out, 
    // useExtrinsicGuess (false), iterations (100), reprojectionError (3.0 pixels), confidence (0.99), inliers out
    bool success = cv::solvePnPRansac(points3D, points2D, K, distCoeffs, rvec, tvec, false, 100, 3.0, 0.99, inliers);

    // --- SAFEGUARD A: Inlier Validation ---
    // Even if RANSAC "succeeds", we reject it if less than 15 points agreed with the math.
    // This protects against motion blur or extreme feature mismatching.

    if (!success || inliers.size() < 15) {
        if constexpr (DEBUG_MOTION) {
            std::cerr << "[Odometry Warning] RANSAC failed or insufficient inliers ("
                      << inliers.size() << ").\n";
        }
        return Eigen::Isometry3d::Identity();
    }

    // --- SAFEGUARD B: Physical Sanity Check ---
    // OpenCV's rotation vector norm is the rotation angle in radians.
    double rotation_angle = cv::norm(rvec); 
    double translation_dist = cv::norm(tvec);

    // If the camera rotates > 0.2 radians (~11.5 deg) or translates > 0.15m in a SINGLE frame (33ms),
    // it implies superhuman speed. The math is hallucinating. Reject it.
    if (rotation_angle > 0.5 || translation_dist > 0.5)  {
        if constexpr (DEBUG_MOTION) {
            std::cerr << "[Odometry Warning] Unrealistic motion jump. Angle: "
                      << rotation_angle << " rad, Dist: " << translation_dist << " m.\n";
        }
        return Eigen::Isometry3d::Identity();
    }

    // 5. Convert Rotation Vector (3x1) to Rotation Matrix (3x3)
    cv::Mat R;
    cv::Rodrigues(rvec, R);

    // 6. Build the Eigen::Isometry3d Matrix
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    Eigen::Matrix3d eigenR;
    Eigen::Vector3d eigenT;

    // Manually transfer 64-bit float data from OpenCV cv::Mat to Eigen matrices
    for (int i = 0; i < 3; i++) {
        eigenT(i) = tvec.at<double>(i);
        for (int j = 0; j < 3; j++) {
            eigenR(i, j) = R.at<double>(i, j);
        }
        
    }

    pose.linear() = eigenR;       // Apply Rotation
    pose.translation() = eigenT;  // Apply Translation

    // 7. The Inversion Trap Fix
    // solvePnP outputs the transformation from the World to the Camera.
    // To track trajectory, we need the transformation from the Camera to the World.
    return pose.inverse(); 
}

} // namespace navio