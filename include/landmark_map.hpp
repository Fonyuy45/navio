#pragma once

#include "landmark.hpp"
#include <unordered_map>
#include <vector>

namespace navio {

/**
 * @brief Persistent 3D landmark map for the visual odometry and BA pipeline.
 *
 * LandmarkMap is the memory of the Navio system. It stores all 3D landmarks
 * observed across the lifetime of the sequence and provides efficient queries
 * for both the BA back-end and the feature matching front-end.
 *
 * Architecture
 * ------------
 * Two internal data structures work together:
 *
 * `landmarks_` — primary storage keyed by landmark ID. O(1) lookup by ID.
 *
 * `frame_to_landmark_ids_` — reverse index mapping frame ID to the list of
 *   landmark IDs observed in that frame. This makes sliding window queries
 *   O(k) where k is the window size, rather than O(N) where N is the total
 *   number of landmarks. Without this index, every window query would require
 *   scanning the entire map — a bottleneck that causes cascade failure as the
 *   map grows beyond a few thousand landmarks.
 *
 * Thread safety
 * -------------
 * This class is NOT thread-safe. All calls must be made from a single thread.
 * A future version may add mutex protection for parallel front-end/back-end.
 */
class LandmarkMap {
public:
    LandmarkMap() = default;

    // Stage 1: Core Lifecycle


    /**
     * @brief Inserts a new landmark into the map and updates the reverse index.
     * @param landmark  Fully constructed Landmark with at least one observation.
     */
    void addLandmark(const Landmark& landmark);

    /**
     * @brief Retrieves a landmark by ID.
     * @throws std::out_of_range if the ID does not exist.
     */
    const Landmark& getLandmark(int landmark_id) const;

    /**
     * @brief Overwrites the 3D position of an existing landmark.
     *
     * Called by BundleAdjuster after optimisation to write back the
     * refined world-space position computed by Ceres.
     *
     * @param landmark_id   ID of the landmark to update.
     * @param new_position  Optimised world-space position (metres).
     */
    void updatePosition(int landmark_id, const Eigen::Vector3d& new_position);

    /**
     * @brief Appends a new observation to an existing landmark.
     *
     * Stores the observation, updates the descriptor history, and
     * re-elects the representative descriptor using the median
     * Hamming distance algorithm. Also updates the reverse index.
     *
     * @param landmark_id    ID of the landmark that was observed.
     * @param frame_id       Frame in which it was observed.
     * @param pixel_undist   Undistorted pixel coordinate (u, v).
     * @param new_descriptor 1x32 CV_8U ORB descriptor from this observation.
     */
    void addObservation(int                    landmark_id,
                        int                    frame_id,
                        const Eigen::Vector2d& pixel_undist,
                        const cv::Mat&         new_descriptor);

    // Stage 2: Sliding Window Queries


    /**
     * @brief Heavy query — returns full landmark copies for all frames in window.
     *
     * Used exclusively by BundleAdjuster which needs complete landmark data
     * (position, observations, descriptors) to construct the Ceres factor graph.
     *
     * Uses the reverse index for O(k) performance where k is the window size.
     *
     * @param frame_ids  List of keyframe IDs defining the local window.
     * @return           Vector of full Landmark copies — one per unique landmark.
     */
    std::vector<Landmark> getLandmarksInFrames(const std::vector<int>& frame_ids) const;

    /**
     * @brief Lightweight query — returns only landmark IDs for frames in window.
     *
     * Used by FeatureManager::matchLocalMap to restrict descriptor matching
     * to the local window rather than the full global map. Returning only IDs
     * avoids copying large descriptor matrices and observation lists — the
     * caller fetches only the representative descriptor it needs via getLandmark.
     *
     * This is the key architectural decision that keeps matching O(k) rather
     * than O(N) as the map grows.
     *
     * @param frame_ids  List of keyframe IDs defining the local window.
     * @return           Vector of unique landmark IDs observed in those frames.
     */
    std::vector<int> getLandmarkIdsInFrames(const std::vector<int>& frame_ids) const;


    // Stage 3: Map Hygiene
  
    /**
     * @brief Removes landmarks that have not been observed recently.
     *
     * A landmark is culled if its most recent observation is older than
     * max_age_frames AND it has fewer than 3 total observations. This removes
     * transient noise points while preserving well-established landmarks.
     *
     * Called periodically (every 20 frames) to prevent unbounded map growth.
     *
     * @param current_frame_id  The current frame index.
     * @param max_age_frames    Frames since last observation before culling.
     */
    void cullStaleLandmarks(int current_frame_id, int max_age_frames);

    /**
     * @brief Returns a const reference to the full internal landmark map.
     */
    const std::unordered_map<int, Landmark>& getAllLandmarks() const;

    /**
     * @brief Returns the total number of landmarks currently in the map.
     */
    std::size_t size() const;

    /**
     * @brief Returns the next available unique landmark ID and increments counter.
     *
     * Inline for performance — called once per new landmark creation.
     */
    int nextId() { return next_id_++; }

private:
    /**
     * @brief Re-elects the representative descriptor after a new observation.
     *
     * Computes pairwise Hamming distances between all stored descriptors and
     * selects the one with the minimum median distance — the most "central"
     * descriptor in the appearance space. This is more robust than keeping
     * the first or most recent descriptor as viewpoint changes over time.
     */
    void updateRepresentativeDescriptor(int landmark_id);

    std::unordered_map<int, Landmark>            landmarks_;
    std::unordered_map<int, std::vector<int>>    frame_to_landmark_ids_; ///< Reverse index
    int                                          next_id_{0};
};

} // namespace navio