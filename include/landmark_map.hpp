#pragma once

#include "landmark.hpp"
#include <unordered_map>
#include <vector>

namespace navio {

/**
 * @brief Persistent 3D landmark map for the visual odometry pipeline.
 *
 * LandmarkMap maintains a collection of 3D landmarks observed across multiple
 * frames. It serves as the memory of the pipeline — bridging the VO front-end
 * (feature detection and matching) with the BA back-end (pose and landmark
 * optimisation).
 *
 * Each landmark has a unique integer ID, a 3D position in world space, and a
 * list of 2D observations recording which frames saw it and at which pixel.
 */
class LandmarkMap {
public:

    /**
     * @brief Constructs an empty LandmarkMap with no landmarks.
     */
    LandmarkMap();

    /**
     * @brief Inserts a new landmark into the map.
     *
     * The landmark must already have a valid ID assigned. Use this method
     * when a feature is observed for the first time and a new Landmark struct
     * has been constructed with an initial 3D position and first observation.
     *
     * @param landmark  The landmark to insert
     */
    void addLandmark(const Landmark& landmark);

    /**
     * @brief Retrieves a landmark by its unique ID.
     *
     * Returns a const reference to avoid copying and to prevent external
     * modification of the internal map state.
     *
     * @param landmark_id  Unique integer ID of the landmark to retrieve
     * @return             Const reference to the requested Landmark
     * @throws             std::out_of_range if the ID does not exist
     */
    const Landmark& getLandmark(int landmark_id) const;

    /**
     * @brief Appends a new observation to an existing landmark.
     *
     * Called when a previously seen landmark is observed again in a new frame.
     * Adds the frame ID and pixel coordinate to the landmark's observation list
     * without creating a duplicate landmark entry.
     *
     * @param landmark_id  ID of the landmark that was observed
     * @param frame_id     ID of the frame in which it was observed
     * @param pixel        2D pixel coordinate of the observation (u, v)
     */
    void addObservation(int landmark_id, int frame_id,
                        const Eigen::Vector2d& pixel);

    /**
     * @brief Returns all landmarks observed in any of the given frames.
     *
     * Used by the BA back-end to retrieve the subset of landmarks relevant
     * to a sliding window of keyframes. A landmark is included if at least
     * one of its observations belongs to one of the queried frame IDs.
     *
     * @param frame_ids  List of frame IDs defining the query window
     * @return           Vector of landmarks observed in those frames
     */
    std::vector<Landmark> getLandmarksInFrames(
        const std::vector<int>& frame_ids) const;

    /**
     * @brief Returns a const reference to the entire landmark map.
     *
     * Provides read-only access to all landmarks for visualisation,
     * evaluation, or export. The const reference avoids copying the
     * potentially large map.
     *
     * @return  Const reference to the internal unordered_map of landmarks
     */
    const std::unordered_map<int, Landmark>& getAllLandmarks() const;

    /**
     * @brief Returns the total number of landmarks currently in the map.
     */
    size_t size() const;

private:

    /// Internal storage — keyed by landmark ID for O(1) lookup
    std::unordered_map<int, Landmark> landmarks_;

    /// Auto-incrementing counter for generating unique landmark IDs
    int next_id_{0};
};

} // namespace navio