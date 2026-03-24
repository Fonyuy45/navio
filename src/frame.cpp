#include "frame.hpp"

namespace navio {


Frame::Frame (const int id, double timestamp, const cv::Mat& rgb_image, const cv::Mat& depth_image)
    : id_{id}, timestamp_{timestamp}, rgb_image_{rgb_image}, depth_image_{depth_image}
{}


std::vector<Eigen::Vector3d> Frame::generatePointCloud (const Camera& camera) const {

    int height {depth_image_.rows} ; 
    int width {depth_image_.cols};


     
    std::vector<Eigen::Vector3d> cloud_sequence;

    for (int row {0}; row < height; row++){

        for (int col {0}; col < width; col++){

            double depth = depth_image_.at<uint16_t>(row, col);
           if (depth == 0){

            continue;

           }


           cloud_sequence.push_back(camera.unproject( Eigen::Vector2d(col,row), depth));
        }
        

    }

    return cloud_sequence;

}

} // namespace navio