#include "trajectory.hpp"

namespace navio {

Trajectory::Trajectory()
    : pose_list_{Eigen::Isometry3d::Identity()}
{}

void Trajectory::updatePose(const Eigen::Isometry3d& pose)
{
    // Compose: new global pose = current global pose * relative transform
    pose_list_.push_back(pose_list_.back() * pose);
}

Eigen::Isometry3d Trajectory::getCurrentPose() const
{
    return pose_list_.back();
}

const std::vector<Eigen::Isometry3d,
                   Eigen::aligned_allocator<Eigen::Isometry3d>>&
Trajectory::getTrajectory() const
{
    return pose_list_;
}

void Trajectory::updatePoseAt(int index, const Eigen::Isometry3d& pose)
{
    // Bounds check — silently ignore out-of-range indices
    if (index >= 0 && index < static_cast<int>(pose_list_.size())) {
        pose_list_[index] = pose;
    }
}

void Trajectory::addAbsolutePose(const Eigen::Isometry3d& pose)
{
    // V2: store directly — no accumulation needed for absolute poses
    pose_list_.push_back(pose);
}

void Trajectory::addAbsolutePose(const Eigen::Matrix4d& pose)
{
    // Convert Matrix4d to Isometry3d by extracting rotation and translation
    Eigen::Isometry3d iso = Eigen::Isometry3d::Identity();
    iso.linear()      = pose.block<3,3>(0,0);
    iso.translation() = pose.block<3,1>(0,3);
    pose_list_.push_back(iso);
}

} // namespace navio