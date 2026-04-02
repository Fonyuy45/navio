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


    struct  Match3D2D{

        Eigen::Vector3d point3d_A;
        Eigen::Vector2d point2d_B;
        int landmark_id;
    };

    

class FeatureManager {
public:
    FeatureManager ();



    std::vector<Match3D2D> computeCorrespondences(const Frame& frameA, const Frame& frameB, const Camera& camera) const;

    std::vector<std::pair<int, Eigen::Vector2d>> trackLandmarks( const Frame& current_frame,  LandmarkMap& map, const Camera& camera) const;




private:
    cv::Ptr<cv::Feature2D> feature_detector_;
    cv::Ptr<cv::DescriptorMatcher> descriptor_matcher_;
    
    std::vector<cv::DMatch> filterMatches(const std::vector<std::vector<cv::DMatch>>& knn_matches) const;

};
} //navio namespace