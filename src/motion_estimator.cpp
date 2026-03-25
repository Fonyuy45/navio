#include "motion_estimator.hpp"
#include "feature_manager.hpp"



namespace navio {



    MotionEstimator::MotionEstimator ()
    {}

    Eigen::Isometry3d MotionEstimator::estimateRelativePose( const std::vector<Match3D2D>& correspondences, const Camera& camera ) const{

    // 1. Get 3D-2D matches from FeatureManager

    // 2. Split into separate 3D points and 2D points vectors

    if (correspondences.size() < 4){

        return Eigen::Isometry3d::Identity();
    }


    std::vector<cv::Point3f> points3D;
    std::vector<cv::Point2f> points2D;

    for ( const auto& match : correspondences ) {
        points3D.push_back( cv::Point3f (
            
            static_cast <float> (match.point3d_A.x()),
            static_cast<float> (match.point3d_A.y()),
            static_cast<float> (match.point3d_A.z())));

        points2D.push_back(cv::Point2f(
            static_cast<float>(match.point2d_B.x()),
            static_cast<float>(match.point2d_B.y())));
    }

    // 3. Get camera intrinsics
    cv::Mat K = camera.getIntrinsicMatrix();  
    cv::Mat distCoeffs = camera.getDistCoeffs();

    // 4. Solve PnP — finds rotation and translation
    cv::Mat rvec, tvec;
    cv::solvePnPRansac( points3D, points2D, K, distCoeffs, rvec, tvec );

    // 5. Convert rotation vector to rotation matrix
    cv::Mat R;
    cv::Rodrigues( rvec, R );

    // 6. Build Eigen::Isometry3d from R and t
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();

    Eigen::Matrix3d eigenR;
    Eigen::Vector3d eigenT;

    // Copy OpenCV matrix into Eigen
    for ( int i = 0; i < 3; i++ ) {
        eigenT(i) = tvec.at<double>(i);
        for ( int j = 0; j < 3; j++ ) {
            eigenR(i,j) = R.at<double>(i,j);
        }
    }

    pose.linear()      = eigenR;  // rotation part
    pose.translation() = eigenT;  // translation part

    return pose;
}

}//namespace navio