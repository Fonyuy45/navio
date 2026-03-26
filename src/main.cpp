#include "camera.hpp"
#include "feature_manager.hpp"
#include "frame.hpp"
#include "trajectory.hpp"
#include "motion_estimator.hpp"
#include "visualiser.hpp"
#include <iomanip>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include <sstream>  // REQUIRED FOR std::istringstream
#include <iostream>

int main() {
    try {
        // 1. Load Calibration
        YAML::Node config = YAML::LoadFile("../config/camera_params.yaml");

        const double fx = config["fx"].as<double>();
        const double fy = config["fy"].as<double>();
        const double cx = config["cx"].as<double>();
        const double cy = config["cy"].as<double>();
        const double depth_scale = config["depth_scale"].as<double>();

        const double k1 = config["k1"].as<double>();
        const double k2 = config["k2"].as<double>();
        const double p1 = config["p1"].as<double>();
        const double p2 = config["p2"].as<double>();
        const double k3 = config["k3"].as<double>();

        std::cout << "Successfully loaded YAML!\n";
        
        // 2. Initialize Core Objects (No parentheses for default constructors!)
        navio::Camera camera(fx, fy, cx, cy, depth_scale, k1, k2, p1, p2, k3);
        std::cout << "K_matrix = \n" << camera.getIntrinsicMatrix() << "\n";
        
        navio::FeatureManager featureManager;
        navio::MotionEstimator motionEstimator;
        navio::Trajectory trajectory;
        navio::Visualiser visualiser;

        // 3. Load Associated File
        std::fstream assoc_file("../rgbd_dataset_freiburg1_xyz/associated.txt");
        if (!assoc_file.is_open()) {
            throw std::runtime_error("Cannot open associated.txt");
        }

        std::vector<std::string> rgb_paths;
        std::vector<std::string> depth_paths;
        std::string line;
        std::vector<double> timestamps; 

        while (std::getline(assoc_file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::istringstream ss(line);
            std::string ts_rgb, rgb_file, ts_depth, depth_file;
            
            
            // Only push if we successfully read all 4 items
            if (ss >> ts_rgb >> rgb_file >> ts_depth >> depth_file) {
                rgb_paths.push_back("../rgbd_dataset_freiburg1_xyz/" + rgb_file);
                depth_paths.push_back("../rgbd_dataset_freiburg1_xyz/" + depth_file);
                timestamps.push_back(std::stod(ts_rgb));
            }
        }
        std::cout << "Loaded  " << rgb_paths.size() << "  frame pairs \n";

        // 4. Setup Initial Frame (Index 0)
        cv::Mat rgb_prev = cv::imread(rgb_paths[0], cv::IMREAD_COLOR);
        cv::Mat depth_prev = cv::imread(depth_paths[0], cv::IMREAD_ANYDEPTH);
        navio::Frame frame_prev(0, 0.0, rgb_prev, depth_prev);

        // 5. Main Odometry Loop (Start at Index 1!)
        for (size_t i = 1; i < rgb_paths.size(); ++i) {
            

            cv::Mat rgb_curr = cv::imread(rgb_paths[i], cv::IMREAD_COLOR);
            cv::Mat depth_curr = cv::imread(depth_paths[i], cv::IMREAD_ANYDEPTH);
            navio::Frame frame_curr(static_cast<int>(i), 0.0, rgb_curr, depth_curr);

            // Use dot notation to call methods on your objects
            auto correspondences = featureManager.computeCorrespondences(frame_prev, frame_curr, camera);

            Eigen::Isometry3d relative_pose = motionEstimator.estimateRelativePose(correspondences, camera);

            trajectory.updatePose(relative_pose);

            cv::Mat traj_image = visualiser.drawTrajectory(trajectory.getTrajectory());
            cv::imshow("Navio - Trajectory", traj_image);

            if (i % 10 == 0) {
                std::cout << "Frame " << i << "/" << rgb_paths.size()
                          << " | Correspondences: " << correspondences.size() << "\n";
            } 

            if (cv::waitKey(1) == 'q') break;

            // Shift the current frame to become the previous frame for the next loop
            frame_prev = frame_curr;
        }

        std::ofstream traj_file ("../results/estimated_trajectory.txt");
        const auto& poses = trajectory.getTrajectory();

        for (size_t i = 0 ; i < poses.size(); ++i){


            Eigen::Vector3d t = poses[i].translation();


            Eigen::Quaterniond q (poses[i].rotation());

            // write in TUM format: timestamp tx, ty, qx,qy,qz, qw
            traj_file << std::fixed << std::setprecision(6)
            << timestamps[i] << " "
            << t.x() << " " << t.y() << " " << t.z() << " "
            <<q.x() << " " << q.y() << " " << q.z() <<" " << q.w() 
            << "\n";
        }
        traj_file.close();

        std::cout << " Trajectory saved to estimated_trajectory.txt\n";
        
        cv::waitKey(0);

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML Error: " << e.what() << "\n";
        return -1;
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV Error: " << e.what() << "\n";
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return -1;
    }

    return 0;
}