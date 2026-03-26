#include "feature_manager.hpp"

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

} // namespace navio