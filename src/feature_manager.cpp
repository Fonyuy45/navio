#include "feature_manager.hpp"

namespace navio {

// =============================================================================
// Internal file-scope utility
// =============================================================================

/**
 * @brief Extracts a single descriptor row as an independent clone.
 *
 * cv::Mat rows are shallow references to the parent matrix. Cloning
 * ensures the stored descriptor owns its memory independently — without
 * this, all stored descriptors would share the same underlying buffer
 * and become invalid when the original matrix is deallocated.
 */
static cv::Mat extractDescriptor(const cv::Mat& desc_matrix, int idx)
{
    return desc_matrix.row(idx).reshape(1, 1).clone();
}

// =============================================================================
// Construction
// =============================================================================

/**
 * ORB is configured with 2500 features for V2 mapping — more features means
 * more landmarks per frame which improves map density and tracking robustness.
 * BruteForce-Hamming is the correct matcher for binary ORB descriptors.
 */
FeatureManager::FeatureManager()
    : feature_detector_   {cv::ORB::create(2500)}
    , descriptor_matcher_ {cv::DescriptorMatcher::create("BruteForce-Hamming")}
{}

// =============================================================================
// Private: Lowe's Ratio Test
// =============================================================================

std::vector<cv::DMatch> FeatureManager::filterMatches(
    const std::vector<std::vector<cv::DMatch>>& knn_matches) const
{
    std::vector<cv::DMatch> good_matches;
    good_matches.reserve(knn_matches.size());
    constexpr float ratio_thresh{0.8f};

    for (const auto& pair : knn_matches) {
        if (pair.size() != 2) continue;
        // Keep only matches where the best is significantly better than second-best.
        // A ratio close to 1.0 means two landmarks look equally similar — ambiguous.
        if (pair[0].distance < ratio_thresh * pair[1].distance) {
            good_matches.push_back(pair[0]);
        }
    }
    return good_matches;
}

// =============================================================================
// V1: computeCorrespondences
// =============================================================================

std::vector<Match3D2D> FeatureManager::computeCorrespondences(
    const Frame&  frameA,
    const Frame&  frameB,
    const Camera& camera) const
{
    // Convert to grayscale — ORB operates on single-channel images
    cv::Mat grayA, grayB;
    cv::cvtColor(frameA.getRgbImage(), grayA, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameB.getRgbImage(), grayB, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> kpA, kpB;
    cv::Mat descA, descB;
    feature_detector_->detectAndCompute(grayA, cv::noArray(), kpA, descA);
    feature_detector_->detectAndCompute(grayB, cv::noArray(), kpB, descB);

    if (kpA.empty() || kpB.empty()) return {};

    std::vector<std::vector<cv::DMatch>> knn_matches;
    descriptor_matcher_->knnMatch(descA, descB, knn_matches, 2);
    const std::vector<cv::DMatch> good_matches = filterMatches(knn_matches);
    if (good_matches.empty()) return {};

    const cv::Mat K = camera.getIntrinsicMatrix();
    const cv::Mat D = camera.getDistCoeffs();

    // Collect distorted pixel positions for both frames
    std::vector<cv::Point2f> ptsA_distorted, ptsB_distorted;
    for (const auto& m : good_matches) {
        ptsA_distorted.push_back(kpA[m.queryIdx].pt);
        ptsB_distorted.push_back(kpB[m.trainIdx].pt);
    }

    // Undistort both sets — depth lookup uses distorted coords,
    // unprojection and PnP use undistorted coords
    std::vector<cv::Point2f> ptsA_undistorted, ptsB_undistorted;
    cv::undistortPoints(ptsA_distorted, ptsA_undistorted, K, D, cv::noArray(), K);
    cv::undistortPoints(ptsB_distorted, ptsB_undistorted, K, D, cv::noArray(), K);

    std::vector<Match3D2D> result;
    result.reserve(good_matches.size());

    for (std::size_t i = 0; i < good_matches.size(); ++i) {

        // Depth lookup uses distorted coordinates — the depth image is in
        // the distorted camera frame, not the undistorted frame
        const uint16_t raw_depth = frameA.getDepthImage().at<uint16_t>(
            cvRound(ptsA_distorted[i].y),
            cvRound(ptsA_distorted[i].x));

        if (raw_depth == 0) continue;

        // Unproject uses undistorted coordinates for geometric correctness
        const Eigen::Vector3d point3d = camera.unproject(
            Eigen::Vector2d{
                static_cast<double>(ptsA_undistorted[i].x),
                static_cast<double>(ptsA_undistorted[i].y)},
            static_cast<double>(raw_depth));

        const Eigen::Vector2d pixel_B{
            static_cast<double>(ptsB_undistorted[i].x),
            static_cast<double>(ptsB_undistorted[i].y)};

        result.push_back(Match3D2D{
            point3d,
            pixel_B,
            extractDescriptor(descA, good_matches[i].queryIdx)});
    }
    return result;
}

// =============================================================================
// V2 Step 1: extractFeatures
// =============================================================================

void FeatureManager::extractFeatures(
    const Frame&               frame,
    std::vector<cv::KeyPoint>& keypoints,
    cv::Mat&                   descriptors) const
{
    cv::Mat gray;
    cv::cvtColor(frame.getRgbImage(), gray, cv::COLOR_BGR2GRAY);
    feature_detector_->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
}

// =============================================================================
// V2 Step 2: matchLocalMap (Read-Only)
// =============================================================================

std::vector<MapMatch> FeatureManager::matchLocalMap(
    const cv::Mat&                   descriptors,
    const std::vector<cv::KeyPoint>& keypoints,
    const LandmarkMap&               map,
    const std::vector<int>&          active_landmark_ids,
    const Camera&                    camera) const
{
    if (active_landmark_ids.empty() || descriptors.empty()) return {};

    // Build descriptor matrix from ONLY the local window landmarks.
    // This is O(k) where k is the window size — not O(N) over the full map.
    cv::Mat      landmark_descriptors;
    std::vector<int> landmark_ids;

    for (int id : active_landmark_ids) {
        try {
            const auto& lm = map.getLandmark(id);
            if (lm.descriptor.empty()) continue;
            landmark_descriptors.push_back(lm.descriptor);
            landmark_ids.push_back(id);
        } catch (...) {
            continue; // Landmark was culled — skip safely
        }
    }

    if (landmark_descriptors.empty()) return {};

    // kNN matching with k=2 — required for Lowe's ratio test
    std::vector<std::vector<cv::DMatch>> knn_matches;
    descriptor_matcher_->knnMatch(descriptors, landmark_descriptors, knn_matches, 2);
    std::vector<cv::DMatch> good_matches = filterMatches(knn_matches);

    if (good_matches.empty()) return {};

    // Batch undistort only the matched keypoints — avoids undistorting all
    // detected keypoints when only a fraction will be matched
    std::vector<cv::Point2f> matched_distorted;
    matched_distorted.reserve(good_matches.size());
    for (const auto& m : good_matches) {
        matched_distorted.push_back(keypoints[m.queryIdx].pt);
    }

    std::vector<cv::Point2f> matched_undistorted;
    const cv::Mat K = camera.getIntrinsicMatrix();
    const cv::Mat D = camera.getDistCoeffs();
    cv::undistortPoints(matched_distorted, matched_undistorted, K, D,
                        cv::noArray(), K);

    std::vector<MapMatch> result;
    result.reserve(good_matches.size());

    for (std::size_t i = 0; i < good_matches.size(); ++i) {
        result.push_back({
            landmark_ids[good_matches[i].trainIdx],
            good_matches[i].queryIdx,
            Eigen::Vector2d(matched_undistorted[i].x, matched_undistorted[i].y)
        });
    }

    return result;
}

// =============================================================================
// V2 Step 3: updateMap (Write-Only)
// =============================================================================

void FeatureManager::updateMap(
    const Frame&                     frame,
    const std::vector<cv::KeyPoint>& keypoints,
    const cv::Mat&                   descriptors,
    const std::vector<MapMatch>&     matches,
    const cv::Mat&                   inliers,
    const Eigen::Matrix4d&           T_wc,
    const Camera&                    camera,
    LandmarkMap&                     map) const
{
    std::vector<bool> is_matched(keypoints.size(), false);

    // -------------------------------------------------------------------------
    // Write Phase 1: Add RANSAC-verified observations to existing landmarks
    //
    // Only RANSAC inliers are written. A match that passed the ratio test is
    // a descriptor-level hypothesis. A RANSAC inlier is a geometric fact —
    // the point reprojects within the threshold given the confirmed pose.
    // Adding outlier observations would corrupt landmark descriptor history
    // and degrade future matching accuracy.
    // -------------------------------------------------------------------------
    if (!inliers.empty()) {
        for (int i = 0; i < inliers.rows; ++i) {
            const int inlier_idx             = inliers.at<int>(i, 0);
            const MapMatch& valid_match      = matches[inlier_idx];

            is_matched[valid_match.keypoint_index] = true;

            map.addObservation(
                valid_match.landmark_id,
                frame.getId(),
                valid_match.pixel_undist,
                extractDescriptor(descriptors, valid_match.keypoint_index));
        }
    }

    // -------------------------------------------------------------------------
    // Write Phase 2: Seed new landmarks from unmatched keypoints
    //
    // Unmatched keypoints represent scene points not yet in the map.
    // They are unprojected into world coordinates using the CONFIRMED T_wc.
    // Using an unconfirmed pose here would place landmarks at wrong world
    // positions — all future matches against them would fail or corrupt BA.
    //
    // A hard cap prevents unbounded map growth which would slow matching.
    // -------------------------------------------------------------------------
    constexpr std::size_t MAX_NEW_LANDMARKS{400};
    std::size_t new_count{0};

    std::vector<cv::Point2f> unmatched_distorted;
    std::vector<int>         unmatched_indices;

    for (std::size_t i = 0; i < keypoints.size(); ++i) {
        if (!is_matched[i]) {
            unmatched_distorted.push_back(keypoints[i].pt);
            unmatched_indices.push_back(static_cast<int>(i));
        }
    }

    if (unmatched_distorted.empty()) return;

    std::vector<cv::Point2f> unmatched_undistorted;
    const cv::Mat K = camera.getIntrinsicMatrix();
    const cv::Mat D = camera.getDistCoeffs();
    cv::undistortPoints(unmatched_distorted, unmatched_undistorted, K, D,
                        cv::noArray(), K);

    for (std::size_t j = 0; j < unmatched_distorted.size(); ++j) {

        if (new_count >= MAX_NEW_LANDMARKS) break;

        // Depth lookup uses distorted coordinates — depth image is distorted
        const int col{cvRound(unmatched_distorted[j].x)};
        const int row{cvRound(unmatched_distorted[j].y)};

        if (col < 0 || col >= frame.getDepthImage().cols ||
            row < 0 || row >= frame.getDepthImage().rows) continue;

        const uint16_t raw_depth{
            frame.getDepthImage().at<uint16_t>(row, col)};
        if (raw_depth == 0) continue;

        // Unproject to local camera frame using undistorted coordinates
        const Eigen::Vector2d undist_px{
            unmatched_undistorted[j].x,
            unmatched_undistorted[j].y};
        const Eigen::Vector3d local_point{
            camera.unproject(undist_px, static_cast<double>(raw_depth))};

        // Transform to world frame using the confirmed camera pose.
        // homogeneous() converts Vector3d to Vector4d for Matrix4d multiply.
        // .head<3>() extracts the XYZ components from the result.
        const Eigen::Vector3d global_point{
            (T_wc * local_point.homogeneous()).head<3>()};

        const int orig_idx{unmatched_indices[j]};

        Landmark lm;
        lm.id          = map.nextId();
        lm.position_3d = global_point;
        lm.descriptor  = extractDescriptor(descriptors, orig_idx);
        lm.all_descriptors.push_back(lm.descriptor.clone());
        lm.observations.push_back(Observation{frame.getId(), undist_px});

        map.addLandmark(lm);
        ++new_count;
    }
}

} // namespace navio