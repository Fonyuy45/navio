#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>

namespace navio {

class Visualiser {


public:
    Visualiser();

  
  cv::Mat visualiseMatches(
    const cv::Mat& frameA,
    const std::vector<cv::KeyPoint>& keypoints1,
    const cv::Mat& frameB,
    const std::vector<cv::KeyPoint>& keypoints2,
    const std::vector<cv::DMatch>& matches) const;
  

  cv::Mat drawTrajectory( const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& pose_list_) const;


};

} // namespace navio