#pragma once

#include "camera.hpp"
#include "frame.hpp"
#include "landmark_map.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace navio {

// =============================================================================
// V1 Data Structures
// =============================================================================

/**
 * @brief A matched 3D-2D correspondence between two consecutive frames.
 *
 * Used exclusively by the V1 relative MotionEstimator front-end.
 * point3d_A is unprojected from frameA using calibrated depth.
 * point2d_B is the undistorted pixel in frameB where the same physical
 * point was observed. Together they form the input to solvePnPRansac.
 */
struct Match3D2D {
    Eigen::Vector3d point3d_A  {};  ///< 3D position in frameA camera frame
    Eigen::Vector2d point2d_B  {};  ///< Undistorted pixel in frameB
    cv::Mat         descriptor {};  ///< 1x32 CV_8U ORB descriptor from frameA
};

// =============================================================================
// V2 Data Structures
// =============================================================================

/**
 * @brief Links a known 3D map landmark to a 2D keypoint in the current frame.
 *
 * Produced by matchLocalMap and consumed by both solvePnPRansac (for pose
 * estimation) and updateMap (for map mutation). Carrying the keypoint_index
 * allows updateMap to identify exactly which keypoints were RANSAC inliers
 * without re-running the match.
 */
struct MapMatch {
    int             landmark_id;    ///< ID of the matched landmark in LandmarkMap
    int             keypoint_index; ///< Index into the current frame keypoint vector
    Eigen::Vector2d pixel_undist;   ///< Undistorted pixel — ready for PnP and BA
};

// =============================================================================
// FeatureManager
// =============================================================================

/**
 * @brief Front-end feature pipeline for both V1 and V2 architectures.
 *
 * V1 Architecture (computeCorrespondences)
 * -----------------------------------------
 * Matches features between two consecutive frames and returns 3D-2D
 * correspondences for frame-to-frame relative pose estimation.
 * Simple, robust, but produces no persistent map.
 *
 * V2 Architecture (extractFeatures + matchLocalMap + updateMap)
 * -------------------------------------------------------------
 * Strictly separates read-only tracking from map mutation. This separation
 * is critical for correctness — it prevents outlier observations from
 * poisoning the map before pose is confirmed by RANSAC.
 *
 * The three-step V2 pipeline:
 *   Step 1 — extractFeatures:  Detect ORB keypoints and descriptors.
 *   Step 2 — matchLocalMap:    Read-only search against the local window.
 *                               No map writes. Returns match candidates.
 *   Step 3 — updateMap:        Write-only. Called ONLY after RANSAC confirms
 *                               the pose. Adds only geometrically verified
 *                               inlier observations and new landmarks.
 *
 * Why local map matching?
 * -----------------------
 * Matching against the full global map with brute-force BFMatcher scales
 * as O(N) where N is the total landmark count. With 50,000 landmarks the
 * ratio test degrades severely — the best and second-best match distances
 * converge, causing the test to reject true matches and accept false ones
 * simultaneously. Restricting matching to the local window of ~500 active
 * landmarks keeps matching O(k) and the ratio test meaningful.
 *
 * Why only inliers in updateMap?
 * ------------------------------
 * A match that passes the ratio test is a descriptor-level hypothesis.
 * A RANSAC inlier is a geometrically verified fact — the matched point
 * reprojects within the reprojection threshold given the confirmed camera
 * pose. Adding outlier observations to landmarks would corrupt their
 * descriptor history and 3D position estimates, degrading future matching
 * and BA accuracy over time.
 */
class FeatureManager {
public:
    FeatureManager();

    // =========================================================================
    // V1 API — Relative Odometry
    // =========================================================================

    /**
     * @brief Matches features between two consecutive frames.
     *
     * Detects ORB keypoints in both frames, matches descriptors using
     * Lowe's ratio test, undistorts matched pixels, and unprojects frameA
     * keypoints into 3D using depth. Returns 3D-2D pairs for solvePnPRansac.
     *
     * @param frameA  Source frame — provides 3D points via depth unprojection.
     * @param frameB  Target frame — provides 2D pixel observations.
     * @param camera  Calibrated camera for unprojection and undistortion.
     * @return        Vector of 3D-2D correspondences for motion estimation.
     */
    std::vector<Match3D2D> computeCorrespondences(
        const Frame&  frameA,
        const Frame&  frameB,
        const Camera& camera) const;

    // =========================================================================
    // V2 API — Local Bundle Adjustment
    // =========================================================================

    /**
     * @brief Step 1: Extracts ORB keypoints and descriptors from a frame.
     *
     * Separated from matching to allow the caller to reuse keypoints and
     * descriptors across multiple pipeline stages without re-detecting.
     *
     * @param frame        Input RGB-D frame.
     * @param keypoints    Output keypoint vector.
     * @param descriptors  Output descriptor matrix (Nx32 CV_8U).
     */
    void extractFeatures(
        const Frame&               frame,
        std::vector<cv::KeyPoint>& keypoints,
        cv::Mat&                   descriptors) const;

    /**
     * @brief Step 2: Read-only match against the active local map window.
     *
     * Builds a descriptor matrix from only the landmarks active in the
     * current sliding window, runs kNN matching with ratio test, and
     * returns match candidates. Does NOT write to the LandmarkMap.
     *
     * Matching is restricted to active_landmark_ids for two reasons:
     * 1. Performance — avoids O(N) global search as the map grows.
     * 2. Correctness — the ratio test is meaningful only when the
     *    candidate set is small enough that descriptors are distinguishable.
     *
     * @param descriptors          Current frame descriptor matrix.
     * @param keypoints            Current frame keypoints.
     * @param map                  Global landmark map (read-only).
     * @param active_landmark_ids  IDs of landmarks in the local window.
     * @param camera               Camera for undistortion.
     * @return                     Vector of match candidates for PnP.
     */
    std::vector<MapMatch> matchLocalMap(
        const cv::Mat&                   descriptors,
        const std::vector<cv::KeyPoint>& keypoints,
        const LandmarkMap&               map,
        const std::vector<int>&          active_landmark_ids,
        const Camera&                    camera) const;

    /**
     * @brief Step 3: Write confirmed inliers to map and seed new landmarks.
     *
     * Called ONLY after solvePnPRansac confirms the camera pose. Two writes:
     *
     * Write 1 — Inlier observations:
     *   For each RANSAC inlier, adds a new observation to the matched
     *   landmark. Using only inliers ensures the map contains only
     *   geometrically verified observations. Outliers are silently discarded.
     *
     * Write 2 — New landmarks:
     *   Unmatched keypoints with valid depth are unprojected into world
     *   coordinates using the confirmed T_wc and added as new landmarks.
     *   Using the confirmed pose here is critical — seeding landmarks with
     *   an unconfirmed pose would place 3D points at wrong world positions,
     *   corrupting all future matches against those landmarks.
     *
     * @param frame      Current frame for depth lookup.
     * @param keypoints  All detected keypoints in current frame.
     * @param descriptors All detected descriptors in current frame.
     * @param matches    All match candidates from matchLocalMap.
     * @param inliers    RANSAC inlier indices from solvePnPRansac.
     * @param T_wc       Confirmed camera-to-world pose.
     * @param camera     Camera model for unprojection.
     * @param map        Global landmark map — mutated by this call.
     */
    void updateMap(
        const Frame&                     frame,
        const std::vector<cv::KeyPoint>& keypoints,
        const cv::Mat&                   descriptors,
        const std::vector<MapMatch>&     matches,
        const cv::Mat&                   inliers,
        const Eigen::Matrix4d&           T_wc,
        const Camera&                    camera,
        LandmarkMap&                     map) const;

private:
    /**
     * @brief Applies Lowe's ratio test to kNN match pairs.
     *
     * Keeps only matches where the best candidate distance is less than
     * ratio_thresh times the second-best distance. Rejects ambiguous matches
     * where two landmarks look equally similar to the current keypoint.
     */
    std::vector<cv::DMatch> filterMatches(
        const std::vector<std::vector<cv::DMatch>>& knn_matches) const;

    cv::Ptr<cv::Feature2D>         feature_detector_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
};

} // namespace navio