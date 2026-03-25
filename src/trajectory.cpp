#include"trajectory.hpp"


namespace navio{   


Trajectory::Trajectory()
    : pose_list_ {Eigen::Isometry3d::Identity()}
    
{}

    void Trajectory::updatePose( const Eigen::Isometry3d& pose) {

        pose_list_.push_back(pose_list_.back()*pose);

    }

    Eigen::Isometry3d Trajectory::getCurrentPose () const{

        return pose_list_.back();
    }

    

    const std::vector<Eigen::Isometry3d, Eigen::aligned_allocator<Eigen::Isometry3d>>& Trajectory::getTrajectory() const{


        return pose_list_;
    }

} //namespace navio