#pragma once
#include <ceres/ceres.h>
#include <ceres/rotation.h>

namespace navio {

/**
 * @brief Pinhole reprojection error functor for Ceres AutoDiff.
 *
 * Computes the 2D residual between a predicted projection and a stored
 * observation. The observation pixel MUST be undistorted before it is
 * stored in the LandmarkMap — this functor uses the pinhole model only
 * and contains no distortion terms. Undistortion is enforced upstream
 * by FeatureManager, which stores only undistorted pixels in Observation.
 *
 * Parameter blocks
 * ----------------
 * camera_pose[6]  — {r0, r1, r2, t0, t1, t2}
 *                   Angle-axis rotation vector (world-to-camera) followed
 *                   by translation (world-to-camera). This is Tcw, NOT Twc.
 *                   BundleAdjuster is responsible for the Twc -> Tcw
 *                   inversion before handing poses to Ceres, and the
 *                   Tcw -> Twc inversion on write-back.
 *
 * point_3d[3]     — {X, Y, Z} world-space landmark position.
 *
 * Residual blocks
 * ---------------
 * residuals[0]    — predicted_u - observed_u   (pixels)
 * residuals[1]    — predicted_v - observed_v   (pixels)
 *
 * Projection model
 * ----------------
 *   P_cam = R * P_world + t          (AngleAxisRotatePoint + translate)
 *   x'    = P_cam.x / P_cam.z       (perspective division)
 *   y'    = P_cam.y / P_cam.z
 *   u     = fx * x' + cx            (apply intrinsics)
 *   v     = fy * y' + cy
 */
struct ReprojectionError {

    const double observed_u;
    const double observed_v;
    const double fx, fy, cx, cy;

    ReprojectionError(double u, double v,
                      double fx, double fy,
                      double cx, double cy)
        : observed_u{u}, observed_v{v}
        , fx{fx}, fy{fy}, cx{cx}, cy{cy}
    {}

    template <typename T>
    bool operator()(const T* const camera_pose,
                    const T* const point_3d,
                    T*             residuals) const
    {
        // --- Rotate the world point into camera space ---
        // camera_pose[0..2] is the angle-axis vector (Tcw rotation).
        T p[3];
        ceres::AngleAxisRotatePoint(camera_pose, point_3d, p);

        // --- Translate into camera space ---
        // camera_pose[3..5] is the translation vector (Tcw translation).
        p[0] += camera_pose[3];
        p[1] += camera_pose[4];
        p[2] += camera_pose[5];

        // --- Guard against points behind the camera ---
        // A non-positive depth produces a degenerate projection.
        // Returning true with zero residuals lets Ceres skip this
        // constraint gracefully rather than producing a NaN gradient.
        if (p[2] <= T(0.0)) {
            residuals[0] = T(0.0);
            residuals[1] = T(0.0);
            return true;
        }

        // --- Perspective division ---
        const T xp = p[0] / p[2];
        const T yp = p[1] / p[2];

        // --- Apply pinhole intrinsics ---
        const T predicted_u = T(fx) * xp + T(cx);
        const T predicted_v = T(fy) * yp + T(cy);

        // --- Residual ---
        residuals[0] = predicted_u - T(observed_u);
        residuals[1] = predicted_v - T(observed_v);

        return true;
    }

    /**
     * @brief Factory — creates a Ceres AutoDiffCostFunction.
     *
     * Template parameters: <Functor, num_residuals, pose_params, point_params>
     * 2 residuals (u, v), 6 pose parameters, 3 point parameters.
     */
    static ceres::CostFunction* Create(double observed_u, double observed_v,
                                       double fx, double fy,
                                       double cx, double cy)
    {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 6, 3>(
            new ReprojectionError(observed_u, observed_v, fx, fy, cx, cy));
    }
};

} // namespace navio