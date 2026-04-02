#include "landmark_map.hpp"

namespace navio {

LandmarkMap::LandmarkMap()
    : landmarks_{},
      next_id_{0}
{}

// -----------------------------------------------------------------------------

void LandmarkMap::addLandmark(const Landmark& landmark) {
    landmarks_[landmark.id] = landmark;
}

const Landmark& LandmarkMap::getLandmark(int landmark_id) const {
    return landmarks_.at(landmark_id);
}

void LandmarkMap::addObservation(int landmark_id, int frame_id,
                                 const Eigen::Vector2d& pixel) {
    // Using .at() ensures we don't accidentally create an empty 
    // landmark if the ID doesn't exist!
    landmarks_.at(landmark_id).observations.push_back(Observation{frame_id, pixel});
}


/*  for the nested loops below: keep in mind that for Version 3.0, 
you might want to create a reverse-lookup 
dictionary: std::unordered_map<int, std::vector<int>> frame_to_landmarks_*/

std::vector<Landmark> LandmarkMap::getLandmarksInFrames(
    const std::vector<int>& frame_ids) const 
{
    std::vector<Landmark> result;

    for (const auto& [id, landmark] : landmarks_) {   
        bool found = false;
        for (const auto& obs : landmark.observations) { 
            for (int fid : frame_ids) {                 
                if (obs.frame_id == fid) {              
                    found = true;
                    break;
                }                                       
            }                                           
            if (found) break;
        }                                              

        if (found) {                                   
            result.push_back(landmark);
        }                                              
    }                                                  
    return result; 
}

const std::unordered_map<int, Landmark>& LandmarkMap::getAllLandmarks() const {
    return landmarks_;
}

// Changed to size_t to match C++ standard library sizing
size_t LandmarkMap::size() const {
    return landmarks_.size();
}

} // namespace navio