#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>

namespace navio {

class Trajectory {


public:
    Trajectory();

    void updatePose( const Eigen::Isometry3d& pose) ;
    
    Eigen::Isometry3d getCurrentPose () const;

    

    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& getTrajectory() const;


private:
    std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>> pose_list_;
};

} // namespace navio