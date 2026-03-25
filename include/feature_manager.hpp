#pragma once
#include "camera.hpp"
#include "frame.hpp"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

namespace navio {


    struct  Match3D2D{

        Eigen::Vector3d point3d_A;
        Eigen::Vector2d point2d_B;
    };

    

class FeatureManager {
public:
    FeatureManager ();



    std::vector<Match3D2D> computeCorrespondences(const Frame& frameA, const Frame& frameB, const Camera& camera) const;



private:
    cv::Ptr<cv::Feature2D> feature_detector_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
    
    std::vector<cv::DMatch> filterMatches(const std::vector<std::vector<cv::DMatch>>& knn_matches) const;

};
} //navio namespace