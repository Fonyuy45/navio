#pragma once
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <vector>

namespace navio {

/**
 * @brief Represents a single 2D observation of a 3D landmark from one frame.
 *
 * Observations store ONLY undistorted pixel coordinates. Undistortion is
 * applied once upstream by FeatureManager before the observation is stored.
 * This design choice has two motivations:
 *
 * 1. Correctness — the ReprojectionError functor uses a simple pinhole model
 *    with no distortion terms. Comparing a pinhole projection (undistorted)
 *    against a distorted stored pixel would produce systematically wrong
 *    residuals and corrupt the optimisation.
 *
 * 2. Efficiency — Ceres calls the cost functor hundreds of times per landmark
 *    during optimisation. Storing undistorted pixels means distortion correction
 *    happens once at observation time, not inside the inner optimisation loop.
 */
struct Observation {
    int             frame_id;
    Eigen::Vector2d pixel_undist; ///< Undistorted (u,v) pixel — pinhole-ready
};

/**
 * @brief A persistent 3D point in the world map.
 *
 * Each Landmark represents a physical scene point that has been observed
 * across one or more frames. It maintains:
 *
 * - A world-space 3D position optimised by Bundle Adjustment
 * - A representative descriptor for efficient matching
 * - A full descriptor history for robust descriptor election
 * - A list of all frame observations for BA factor graph construction
 *
 * Descriptor design
 * -----------------
 * Two descriptor fields are maintained deliberately:
 *
 * `descriptor` — the single "winner" descriptor used for real-time matching.
 *   Elected by the median Hamming distance algorithm across all observed
 *   descriptors. Updated every time a new observation is added.
 *
 * `all_descriptors` — the complete history of every descriptor seen across
 *   all frames. As the camera observes the same point from different angles
 *   and lighting conditions, the appearance changes. Keeping all versions
 *   allows the election algorithm to always find the most representative
 *   descriptor rather than being locked into the first or last one seen.
 *
 */
struct Landmark {
    int                  id;
    Eigen::Vector3d      position_3d;      ///< World-space position (metres), optimised by BA

    cv::Mat              descriptor;       ///< Representative descriptor — used for matching
    std::vector<cv::Mat> all_descriptors;  ///< Full descriptor history — used for election

    std::vector<Observation> observations; ///< All frame observations of this landmark
};

} // namespace navio