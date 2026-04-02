#include "feature_manager.hpp"
#include "landmark_map.hpp"

namespace navio {

/**
 * @brief Constructs a FeatureManager and initialises the ORB detector
 *        and Brute-Force Hamming descriptor matcher.
 *
 * ORB (Oriented FAST and Rotated BRIEF) is chosen for its speed and
 * robustness, making it well-suited for real-time RGB-D odometry.
 * Hamming distance is the correct metric for binary ORB descriptors.
 */
FeatureManager::FeatureManager()
    : feature_detector_{cv::ORB::create()}
    , descriptor_matcher_{cv::DescriptorMatcher::create("BruteForce-Hamming")}
{}

// -----------------------------------------------------------------------------

std::vector<cv::DMatch> FeatureManager::filterMatches(
    const std::vector<std::vector<cv::DMatch>>& knn_matches) const
{
    std::vector<cv::DMatch> good_matches;

    // Lowe's ratio test threshold — 0.8 is the standard value from the
    // original SIFT paper and works well with ORB descriptors in practice
    constexpr float ratio_thresh{0.8f};

    for (const auto& match_pair : knn_matches) {

        // Safety check — matcher may return fewer than 2 neighbours for
        // keypoints near image boundaries or with very few candidates
        if (match_pair.size() != 2) {
            continue;
        }

        const cv::DMatch& best_match        = match_pair[0];
        const cv::DMatch& second_best_match = match_pair[1];

        // Ratio test — keep only matches where the best candidate is
        // significantly closer than the second best. A small ratio means
        // the match is distinctive and unlikely to be a false correspondence.
        if (best_match.distance < ratio_thresh * second_best_match.distance) {
            good_matches.push_back(best_match);
        }
    }

    return good_matches;
}

// -----------------------------------------------------------------------------

std::vector<Match3D2D> FeatureManager::computeCorrespondences(
    const Frame&  frameA,
    const Frame&  frameB,
    const Camera& camera) const
{
    // --- Step 1: Convert RGB images to grayscale for feature detection --------
    // ORB operates on single-channel images. Grayscale conversion reduces
    // computational load without losing the structural information ORB needs.
    cv::Mat frame1Gray;
    cv::Mat frame2Gray;
    cv::cvtColor(frameA.getRgbImage(), frame1Gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(frameB.getRgbImage(), frame2Gray, cv::COLOR_BGR2GRAY);

    // --- Step 2: Detect keypoints and compute descriptors --------------------
    // detectAndCompute runs detection and description in a single pass,
    // which is more efficient than calling detect() and compute() separately.
    std::vector<cv::KeyPoint> keypoints1;
    std::vector<cv::KeyPoint> keypoints2;
    cv::Mat descriptors1;
    cv::Mat descriptors2;

    feature_detector_->detectAndCompute(
        frame1Gray, cv::noArray(), keypoints1, descriptors1);
    feature_detector_->detectAndCompute(
        frame2Gray, cv::noArray(), keypoints2, descriptors2);

    // --- Step 3: Match descriptors and filter --------------------------------
    // knnMatch returns the 2 nearest descriptor neighbours for each keypoint.
    // This is required for the ratio test — we need both best and second-best.
    std::vector<std::vector<cv::DMatch>> knn_matches;
    descriptor_matcher_->knnMatch(descriptors1, descriptors2, knn_matches, 2);

    const std::vector<cv::DMatch> good_matches{filterMatches(knn_matches)};

// --- Step 4: Lift matched keypoints to 3D and build correspondences ------
    
    // Extract camera matrices to handle the distortion
    cv::Mat K = camera.getIntrinsicMatrix();
    cv::Mat D = camera.getDistCoeffs();

    // 4a. Gather all the distorted 2D pixels for Frame A
    std::vector<cv::Point2f> ptsA_distorted;
    ptsA_distorted.reserve(good_matches.size());
    for (const auto& match : good_matches) {
        ptsA_distorted.push_back(keypoints1[match.queryIdx].pt);
    }

    // 4b. UNDISTORT them mathematically so our 3D points form straight geometry!
    std::vector<cv::Point2f> ptsA_undistorted;
    // We pass K twice so the output points are in standard pixel coordinates, not normalized coordinates
    cv::undistortPoints(ptsA_distorted, ptsA_undistorted, K, D, cv::noArray(), K);

    // 4c. Build the final 3D-2D matches
    std::vector<Match3D2D> result;
    result.reserve(good_matches.size());

    for (size_t i = 0; i < good_matches.size(); ++i) {
        const auto& match = good_matches[i];
        
        const cv::Point2f& distorted_pixelA = ptsA_distorted[i];
        const cv::Point2f& undistorted_pixelA = ptsA_undistorted[i];
        const cv::Point2f& pixelB = keypoints2[match.trainIdx].pt;

        // Look up depth using the DISTORTED pixel (because the depth map itself is distorted)
        // Use cvRound to prevent grabbing background depths near object edges!
        const auto raw_depth = frameA.getDepthImage().at<uint16_t>(
            cvRound(distorted_pixelA.y),
            cvRound(distorted_pixelA.x));

        // Skip invalid depth holes
        if (raw_depth == 0) {
            continue;
        }

        const double depth{static_cast<double>(raw_depth)};

        // Unproject using the UNDISTORTED pixel to create perfect physical 3D geometry
        const Eigen::Vector2d ideal_pixel_A{
            static_cast<double>(undistorted_pixelA.x),
            static_cast<double>(undistorted_pixelA.y)};

        const Eigen::Vector3d point3d{camera.unproject(ideal_pixel_A, depth)};

        // Target pixel in Frame B (solvePnP will handle its distortion internally later)
        const Eigen::Vector2d target_pixel_B{
            static_cast<double>(pixelB.x),
            static_cast<double>(pixelB.y)};

        result.push_back(Match3D2D{point3d, target_pixel_B});
    }

    return result;
}

std::vector<std::pair<int, Eigen::Vector2d>> FeatureManager::trackLandmarks(
    const Frame&  current_frame,
    LandmarkMap&  map,
    const Camera& camera) const
{
    // --- Step 1: Detect keypoints and compute descriptors --------------------
    cv::Mat frame_gray;
    cv::cvtColor(current_frame.getRgbImage(), frame_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    feature_detector_->detectAndCompute(frame_gray, cv::noArray(), keypoints, descriptors);

    // --- Step 2: Handle empty map — all features become new landmarks --------
    // On the very first frame there are no existing landmarks to match against.
    // Every feature with valid depth is registered as a new landmark.
    if (map.size() == 0) {

        // Undistort all keypoints before unprojection
        std::vector<cv::Point2f> pts_distorted;
        pts_distorted.reserve(keypoints.size());
        for (const auto& kp : keypoints) {
            pts_distorted.push_back(kp.pt);
        }

        std::vector<cv::Point2f> pts_undistorted;
        cv::undistortPoints(pts_distorted, pts_undistorted,
                            camera.getIntrinsicMatrix(), camera.getDistCoeffs(),
                            cv::noArray(), camera.getIntrinsicMatrix());

        for (size_t i = 0; i < keypoints.size(); ++i) {

            const int col{cvRound(pts_distorted[i].x)};
            const int row{cvRound(pts_distorted[i].y)};

            // Bounds check — skip keypoints outside image boundaries
            if (col < 0 || col >= current_frame.getDepthImage().cols ||
                row < 0 || row >= current_frame.getDepthImage().rows) {
                continue;
            }

            const uint16_t raw_depth{
                current_frame.getDepthImage().at<uint16_t>(row, col)};

            // Skip invalid depth measurements
            if (raw_depth == 0) continue;

            const double depth{static_cast<double>(raw_depth)};

            const Eigen::Vector2d undist_px{
                static_cast<double>(pts_undistorted[i].x),
                static_cast<double>(pts_undistorted[i].y)};

            const Eigen::Vector3d point3d{camera.unproject(undist_px, depth)};

            // Register as a new landmark
            Landmark lm;
            lm.id           = map.nextId();
            lm.position_3d  = point3d;
            lm.descriptor   = descriptors.row(i).clone();
            lm.all_descriptors.push_back(descriptors.row(i).clone());
            lm.observations.push_back(Observation{
                current_frame.getId(),
                Eigen::Vector2d(pts_distorted[i].x, pts_distorted[i].y)});

            map.addLandmark(lm);
        }

        // No tracked observations to return on the first frame
        return {};
    }

    // --- Step 3: Build landmark descriptor matrix for matching ---------------
    // Stack all representative landmark descriptors into a single cv::Mat.
    // Keep a parallel vector of IDs so we know which row belongs to which landmark.
    const auto& all_landmarks = map.getAllLandmarks();

    cv::Mat     landmark_descriptors;
    std::vector<int> landmark_ids;
    landmark_ids.reserve(all_landmarks.size());

    for (const auto& [id, landmark] : all_landmarks) {
        if (landmark.descriptor.empty()) continue;
        landmark_descriptors.push_back(landmark.descriptor);
        landmark_ids.push_back(id);
    }

    // Safety check — if all landmarks have empty descriptors return early
    if (landmark_descriptors.empty()) return {};

    // --- Step 4: Match current descriptors against the landmark database -----
    std::vector<std::vector<cv::DMatch>> knn_matches;
    descriptor_matcher_->knnMatch(descriptors, landmark_descriptors, knn_matches, 2);

    const std::vector<cv::DMatch> good_matches{filterMatches(knn_matches)};

    // --- Step 5: Process matched features ------------------------------------
    // Track which keypoints were successfully matched to existing landmarks.
    std::vector<bool> is_matched(keypoints.size(), false);
    std::vector<std::pair<int, Eigen::Vector2d>> result;
    result.reserve(good_matches.size());

    for (const auto& match : good_matches) {

        const int kp_idx{match.queryIdx};  // index in current frame keypoints
        const int lm_idx{match.trainIdx};  // index in landmark descriptor matrix

        is_matched[kp_idx] = true;

        const int landmark_id{landmark_ids[lm_idx]};
        const cv::Point2f& pixel{keypoints[kp_idx].pt};
        const Eigen::Vector2d pixel_eigen{pixel.x, pixel.y};

        // Update the landmark with the new observation and descriptor
        map.addObservation(landmark_id, current_frame.getId(),
                           pixel_eigen, descriptors.row(kp_idx));

        // Add to result — BA needs landmark ID paired with observed pixel
        result.push_back({landmark_id, pixel_eigen});
    }

    // --- Step 6: Register unmatched keypoints as new landmarks ---------------
    // Features that did not match any existing landmark represent newly
    // visible scene points and must be added to the map.
    std::vector<cv::Point2f> unmatched_pts;
    std::vector<int>         unmatched_indices;

    for (size_t i = 0; i < keypoints.size(); ++i) {
        if (!is_matched[i]) {
            unmatched_pts.push_back(keypoints[i].pt);
            unmatched_indices.push_back(static_cast<int>(i));
        }
    }

    if (!unmatched_pts.empty()) {

        std::vector<cv::Point2f> unmatched_undistorted;
        cv::undistortPoints(unmatched_pts, unmatched_undistorted,
                            camera.getIntrinsicMatrix(), camera.getDistCoeffs(),
                            cv::noArray(), camera.getIntrinsicMatrix());

        for (size_t j = 0; j < unmatched_pts.size(); ++j) {

            const int col{cvRound(unmatched_pts[j].x)};
            const int row{cvRound(unmatched_pts[j].y)};

            // Bounds check
            if (col < 0 || col >= current_frame.getDepthImage().cols ||
                row < 0 || row >= current_frame.getDepthImage().rows) {
                continue;
            }

            const uint16_t raw_depth{
                current_frame.getDepthImage().at<uint16_t>(row, col)};

            if (raw_depth == 0) continue;

            const double depth{static_cast<double>(raw_depth)};

            const Eigen::Vector2d undist_px{
                static_cast<double>(unmatched_undistorted[j].x),
                static_cast<double>(unmatched_undistorted[j].y)};

            const Eigen::Vector3d point3d{camera.unproject(undist_px, depth)};

            const int orig_idx{unmatched_indices[j]};

            Landmark lm;
            lm.id          = map.nextId();
            lm.position_3d = point3d;
            lm.descriptor  = descriptors.row(orig_idx).clone();
            lm.all_descriptors.push_back(descriptors.row(orig_idx).clone());
            lm.observations.push_back(Observation{
                current_frame.getId(),
                Eigen::Vector2d(unmatched_pts[j].x, unmatched_pts[j].y)});

            map.addLandmark(lm);
        }
    }

    return result;
}


} // namespace navio