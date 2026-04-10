#include "frame.hpp"

namespace navio {

/**
 * @brief Constructs a Frame from raw RGB-D sensor data.
 *
 * Stores the RGB and depth images along with a unique identifier
 * and timestamp for use in sequential pose estimation.
 *
 * @param id           Unique frame index in the sequence (zero-based)
 * @param timestamp    Capture time in seconds (e.g. from TUM dataset header)
 * @param rgb_image    Three-channel BGR image from the colour sensor (CV_8UC3)
 * @param depth_image  Single-channel raw depth image from the depth sensor (CV_16UC1)
 */
Frame::Frame(int id, double timestamp, const cv::Mat& rgb_image, const cv::Mat& depth_image)
    : id_{id}
    , timestamp_{timestamp}
    , rgb_image_{rgb_image}
    , depth_image_{depth_image}
{}

/**
 * @brief Generates a 3D point cloud from the depth image using the camera model.
 *
 * Iterates over every pixel in the depth image. For each pixel with a valid
 * (non-zero) depth value, the raw depth is unprojected into a 3D point in
 * camera coordinate frame using the provided Camera model.
 *
 * Pixels with zero depth are skipped as they represent invalid measurements
 * caused by reflective surfaces, out-of-range objects, or sensor noise.
 *
 * @param camera  Calibrated camera model used for unprojection
 * @return        Vector of 3D points in camera coordinate frame (metres)
 */
std::vector<Eigen::Vector3d> Frame::generatePointCloud(const Camera& camera) const
{
    const int height{depth_image_.rows};
    const int width{depth_image_.cols};

    // Reserve approximate capacity to avoid repeated reallocations
    // as the vector grows. Most depth images have roughly 30-50% valid pixels.
    std::vector<Eigen::Vector3d> point_cloud;
    point_cloud.reserve(height * width / 2);

    for (int row{0}; row < height; ++row) {
        for (int col{0}; col < width; ++col) {

            // Read raw depth value — stored as 16-bit unsigned integer
            const double depth{
                static_cast<double>(depth_image_.at<uint16_t>(row, col))
            };

            // Skip invalid measurements — zero depth has no geometric meaning
            if (depth == 0.0) {
                continue;
            }

            // col maps to u (horizontal, x-axis)
            // row maps to v (vertical,  y-axis)
            const Eigen::Vector2d pixel{static_cast<double>(col),
                                        static_cast<double>(row)};

            point_cloud.push_back(camera.unproject(pixel, depth));
        }
    }

    return point_cloud;
}



} // namespace navio