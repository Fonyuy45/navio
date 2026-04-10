#include "landmark_map.hpp"
#include <algorithm>
#include <limits>
#include <set>

namespace navio {

// -----------------------------------------------------------------------------

/**
 * @brief Inserts a new landmark and synchronises the reverse index.
 *
 * Both `landmarks_` and `frame_to_landmark_ids_` must always be consistent —
 * a landmark present in one must be reachable through the other. Failing to
 * update the reverse index here would cause window queries to silently miss
 * this landmark, degrading BA and matching without any error.
 */
void LandmarkMap::addLandmark(const Landmark& landmark)
{
    landmarks_[landmark.id] = landmark;

    // Register this landmark in the reverse index for each frame that saw it.
    // Typically a new landmark has exactly one observation (its creation frame).
    for (const auto& obs : landmark.observations) {
        frame_to_landmark_ids_[obs.frame_id].push_back(landmark.id);
    }
}

// -----------------------------------------------------------------------------

const Landmark& LandmarkMap::getLandmark(int landmark_id) const
{
    // .at() throws std::out_of_range if not found — caller must handle this.
    return landmarks_.at(landmark_id);
}

// -----------------------------------------------------------------------------

/**
 * @brief Overwrites the 3D position of an existing landmark in-place.
 *
 * Called by BundleAdjuster after each optimisation round to write back
 * the refined position computed by Ceres. Using .at() rather than []
 * ensures we never accidentally create a default-constructed landmark
 * for an ID that does not exist.
 */
void LandmarkMap::updatePosition(int landmark_id,
                                 const Eigen::Vector3d& new_position)
{
    landmarks_.at(landmark_id).position_3d = new_position;
}

// -----------------------------------------------------------------------------

/**
 * @brief Appends a new observation to an existing landmark.
 *
 * Three things happen in order:
 * 1. The observation and descriptor are recorded.
 * 2. The reverse index is updated so this frame can find the landmark.
 * 3. The representative descriptor is re-elected from the full history.
 *
 * Step 3 happens on every observation because the "best" descriptor may
 * change as the landmark is seen from new viewpoints. Keeping the election
 * current ensures matching always uses the most representative appearance.
 */
void LandmarkMap::addObservation(int                    landmark_id,
                                 int                    frame_id,
                                 const Eigen::Vector2d& pixel_undist,
                                 const cv::Mat&         new_descriptor)
{
    auto& lm = landmarks_.at(landmark_id);

    // 1. Record the new observation and descriptor
    lm.observations.push_back({frame_id, pixel_undist});
    lm.all_descriptors.push_back(new_descriptor.clone());

    // 2. Keep reverse index in sync — this frame can now find this landmark
    frame_to_landmark_ids_[frame_id].push_back(landmark_id);

    // 3. Re-elect the best representative descriptor
    updateRepresentativeDescriptor(landmark_id);
}

// -----------------------------------------------------------------------------

/**
 * @brief Returns full landmark copies for all unique landmarks in the window.
 *
 * Uses the reverse index to avoid scanning the entire map. The set ensures
 * each landmark appears only once even if it was seen in multiple window frames.
 *
 * This is the heavy query — it copies full Landmark structs including all
 * descriptor history and observations. Only BundleAdjuster should call this.
 */
std::vector<Landmark> LandmarkMap::getLandmarksInFrames(
    const std::vector<int>& frame_ids) const
{
    std::set<int> unique_ids;

    for (int fid : frame_ids) {
        auto it = frame_to_landmark_ids_.find(fid);
        if (it != frame_to_landmark_ids_.end()) {
            for (int lm_id : it->second) {
                unique_ids.insert(lm_id);
            }
        }
    }

    std::vector<Landmark> result;
    result.reserve(unique_ids.size());
    for (int id : unique_ids) {
        result.push_back(landmarks_.at(id));
    }
    return result;
}

// -----------------------------------------------------------------------------

/**
 * @brief Returns only landmark IDs for all unique landmarks in the window.
 *
 * Lightweight version of getLandmarksInFrames — returns integers only.
 * Used by FeatureManager::matchLocalMap to restrict the descriptor search
 * to the local window without copying large landmark data structures.
 */
std::vector<int> LandmarkMap::getLandmarkIdsInFrames(
    const std::vector<int>& frame_ids) const
{
    std::set<int> unique_ids;

    for (int fid : frame_ids) {
        auto it = frame_to_landmark_ids_.find(fid);
        if (it != frame_to_landmark_ids_.end()) {
            for (int lm_id : it->second) {
                unique_ids.insert(lm_id);
            }
        }
    }

    return std::vector<int>(unique_ids.begin(), unique_ids.end());
}

// -----------------------------------------------------------------------------

/**
 * @brief Re-elects the representative descriptor using median Hamming distance.
 *
 * Builds an NxN pairwise distance matrix between all stored descriptors.
 * For each descriptor, computes the median of its distances to all others.
 * The descriptor with the lowest median is the most "central" in appearance
 * space — it is the best representative across all observed viewpoints.
 *
 * This mirrors the descriptor management strategy in ORB-SLAM2 and ensures
 * the matching descriptor stays robust as the camera angle changes over time.
 *
 * Complexity: O(N^2) where N is the number of observations. Acceptable for
 * typical landmark lifetimes but may need approximation for very long sequences.
 */
void LandmarkMap::updateRepresentativeDescriptor(int landmark_id)
{
    auto& lm = landmarks_.at(landmark_id);
    const int N = static_cast<int>(lm.all_descriptors.size());

    if (N == 0) return;
    if (N == 1) {
        lm.descriptor = lm.all_descriptors[0].clone();
        return;
    }

    // Build symmetric Hamming distance matrix
    std::vector<std::vector<int>> distances(N, std::vector<int>(N, 0));
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            int dist = cv::norm(lm.all_descriptors[i],
                                lm.all_descriptors[j],
                                cv::NORM_HAMMING);
            distances[i][j] = distances[j][i] = dist;
        }
    }

    // Find the descriptor with the minimum median distance to all others
    int best_idx  = 0;
    int min_median = std::numeric_limits<int>::max();

    for (int i = 0; i < N; ++i) {
        std::vector<int> row = distances[i];
        std::sort(row.begin(), row.end());
        const int median = row[N / 2];
        if (median < min_median) {
            min_median = median;
            best_idx   = i;
        }
    }

    lm.descriptor = lm.all_descriptors[best_idx].clone();
}

// -----------------------------------------------------------------------------

/**
 * @brief Removes stale landmarks to prevent unbounded map growth.
 *
 * A landmark is culled when two conditions are both true:
 * - It has not been observed for more than max_age_frames frames (stale)
 * - It has fewer than 3 total observations (poorly established)
 *
 * The dual condition prevents removing well-established landmarks that
 * happen to be temporarily out of view, while cleaning up transient noise
 * points that were seen once and never matched again.
 *
 * The reverse index is cleaned before erasing from landmarks_ to maintain
 * the consistency invariant between the two data structures.
 */
void LandmarkMap::cullStaleLandmarks(int current_frame_id, int max_age_frames)
{
    for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
        const auto& lm = it->second;
        const int last_obs_frame = lm.observations.back().frame_id;

        const bool is_stale = (current_frame_id - last_obs_frame) > max_age_frames;
        const bool is_weak  = lm.observations.size() < 3;

        if (is_stale && is_weak) {
            // Remove from reverse index first to maintain consistency
            for (const auto& obs : lm.observations) {
                auto& vec = frame_to_landmark_ids_[obs.frame_id];
                vec.erase(std::remove(vec.begin(), vec.end(), lm.id), vec.end());
            }
            it = landmarks_.erase(it);
        } else {
            ++it;
        }
    }
}

// -----------------------------------------------------------------------------

const std::unordered_map<int, Landmark>& LandmarkMap::getAllLandmarks() const
{
    return landmarks_;
}

// -----------------------------------------------------------------------------

std::size_t LandmarkMap::size() const
{
    return landmarks_.size();
}

} // namespace navio