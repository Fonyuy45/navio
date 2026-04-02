#include "landmark_map.hpp"
#include <limits>
#include <algorithm>

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
                                 const Eigen::Vector2d& pixel,
                                 const cv::Mat& new_descriptor) {
    
    auto& landmark = landmarks_.at(landmark_id);
    
    // 1. Store the new observation
    landmark.observations.push_back(Observation{frame_id, pixel});
    
    // 2. Store the new descriptor in the history (use .clone() to be safe with memory)
    landmark.all_descriptors.push_back(new_descriptor.clone());

    // 3. Recalculate the winner
    updateRepresentativeDescriptor(landmark_id);
}


void LandmarkMap::updateRepresentativeDescriptor(int landmark_id) {
    auto& landmark = landmarks_.at(landmark_id);
    int N = static_cast<int>(landmark.all_descriptors.size());;
    
    // Base cases
    if (N == 0) return;
    if (N == 1) {
        landmark.descriptor = landmark.all_descriptors[0].clone();
        return;
    }

    // Step 1: Create an N x N distance matrix
    std::vector<std::vector<int>> distances(N, std::vector<int>(N, 0));

    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            // cv::norm with NORM_HAMMING is extremely fast for binary descriptors like ORB
            int dist = cv::norm(landmark.all_descriptors[i], landmark.all_descriptors[j], cv::NORM_HAMMING);
            distances[i][j] = dist;
            distances[j][i] = dist; 
        }
    }

    // Step 2: Find the descriptor with the minimum median distance
    int best_median = std::numeric_limits<int>::max();
    int best_idx = 0;

    for (int i = 0; i < N; i++) {
        std::vector<int> dists_i = distances[i];
        
        // Sort to find the median
        std::sort(dists_i.begin(), dists_i.end());
        int median = dists_i[N / 2];

        // Keep track of the lowest median
        if (median < best_median) {
            best_median = median;
            best_idx = i;
        }
    }

    // Step 3: Overwrite the active descriptor with the new winner
    landmark.descriptor = landmark.all_descriptors[best_idx].clone();
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