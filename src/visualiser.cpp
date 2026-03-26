#include "visualiser.hpp"

namespace navio {

/**
 * @brief Constructs a Visualiser instance.
 *
 * No initialisation required — all methods are stateless and operate
 * purely on the data passed to them.
 */
Visualiser::Visualiser()
{}

// -----------------------------------------------------------------------------

/**
 * @brief Visualises ORB feature matches between two consecutive frames.
 *
 * Draws lines connecting matched keypoints side-by-side between frameA
 * and frameB. Useful for visually verifying the quality of feature matching
 * and diagnosing issues with the FeatureManager pipeline stage.
 *
 * @param frameA      RGB image of the source frame (CV_8UC3, BGR)
 * @param keypoints1  ORB keypoints detected in frameA
 * @param frameB      RGB image of the target frame (CV_8UC3, BGR)
 * @param keypoints2  ORB keypoints detected in frameB
 * @param matches     Filtered feature matches from FeatureManager
 * @return            Side-by-side image with match lines drawn (CV_8UC3)
 */
cv::Mat Visualiser::visualiseMatches(
    const cv::Mat&                     frameA,
    const std::vector<cv::KeyPoint>&   keypoints1,
    const cv::Mat&                     frameB,
    const std::vector<cv::KeyPoint>&   keypoints2,
    const std::vector<cv::DMatch>&     matches) const
{
    cv::Mat image_matches;

    // cv::drawMatches renders both images side by side and connects
    // corresponding keypoints with coloured lines
    cv::drawMatches(frameA, keypoints1, frameB, keypoints2,
                    matches, image_matches);

    return image_matches;
}

// -----------------------------------------------------------------------------

/**
 * @brief Renders a top-down 2D trajectory from a sequence of camera poses.
 *
 * Projects each camera pose onto the X-Z plane (horizontal plane in camera
 * convention where Z points forward and X points right) and draws a
 * connected line path on a fixed-size canvas.
 *
 * Coordinate convention:
 *   - Canvas centre (300, 300) corresponds to the world origin
 *   - X axis maps to canvas horizontal (right = positive X)
 *   - Z axis maps to canvas vertical   (up    = positive Z, i.e. forward motion)
 *   - Scale: 50 pixels per metre
 *
 * Poses outside the visible canvas range are still connected — no clipping
 * is applied. For long sequences consider increasing the canvas size or
 * reducing the scale factor.
 *
 * @param pose_list  Sequence of absolute camera poses in world frame,
 *                   ordered from oldest to newest
 * @return           600x600 BGR canvas with trajectory drawn in blue
 */
cv::Mat Visualiser::drawTrajectory(
    const std::vector<Eigen::Isometry3d,
                       Eigen::aligned_allocator<Eigen::Isometry3d>>& pose_list) const
{
    //  white three-channel canvas — large enough for most short sequences
    cv::Mat plot(1000, 1000, CV_8UC3, cv::Scalar(255, 255, 255));

    // Canvas origin offset — maps the world origin (0,0) to the canvas centre
    const int    offset_x{300};
    const int    offset_y{300};

    // Scale factor — pixels per metre. Increase for small motions, decrease
    // for large environments
    const double scale{50.0};

    // Line rendering parameters
    const int    thickness{2};
    const cv::Scalar trajectory_colour{255, 0, 0}; // Blue in BGR

    // Start the previous point at the canvas centre, corresponding to the
    // world origin where the camera begins
    cv::Point previous_point{offset_x, offset_y};
    cv::Point current_point{};

    for (const auto& pose : pose_list) {

        // Extract X (lateral) and Z (forward) translation components
        // Y (vertical) is ignored for the top-down 2D projection
        const int canvas_x{
            static_cast<int>(offset_x + scale * pose.translation().x())};
        const int canvas_y{
            static_cast<int>(offset_y + scale * pose.translation().z())};

        current_point = cv::Point(canvas_x, canvas_y);

        // Connect the previous pose position to the current one
        cv::line(plot, previous_point, current_point,
                 trajectory_colour, thickness, cv::LINE_8);

        // Advance the previous point for the next iteration
        previous_point = current_point;
    }

    return plot;
}

} // namespace navio