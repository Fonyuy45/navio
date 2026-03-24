#pragma once

#include "camera.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace navio {

class Frame {
public:
    Frame (const int id, double timestamp, const cv::Mat& rgb_image, const cv::Mat& depth_image);



    std::vector<Eigen::Vector3d> generatePointCloud (const Camera& camera) const;

    // --- Accessors ---------------------------------------------------------------

    /**
     * @brief Returns the unique frame index in the sequence.
     */
    int getId() const { return id_; }

    /**
     * @brief Returns the capture timestamp in seconds.
     */
    double getTimestamp() const { return timestamp_; }

    /**
     * @brief Returns a const reference to the RGB image.
     */
    const cv::Mat& getRgbImage() const { return rgb_image_; }

    /**
     * @brief Returns a const reference to the raw depth image.
     */
    const cv::Mat& getDepthImage() const { return depth_image_; }


private:
    int id_{};
    double timestamp_{};
    cv::Mat rgb_image_{};
    cv::Mat depth_image_{};
};

}