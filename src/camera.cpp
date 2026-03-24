#include "camera.hpp"

namespace navio {

/**
 * @brief Constructs a Camera with given intrinsic parameters.
 *
 * @param fx  Focal length in pixels along the x-axis
 * @param fy  Focal length in pixels along the y-axis
 * @param cx  Principal point x-coordinate (optical centre column)
 * @param cy  Principal point y-coordinate (optical centre row)
 * @param depth_scale  Converts raw depth integer to metres (e.g. 1000.0 for
 *                     millimetre-encoded depth maps)
 */
Camera::Camera(double fx, double fy, double cx, double cy, double depth_scale)
    : fx_{fx}, fy_{fy}, cx_{cx}, cy_{cy}, depth_scale_{depth_scale}
{}

/**
 * @brief Projects a 3D point in camera space to a 2D image pixel.
 *
 * Uses the standard pinhole projection model:
 *   u = fx * (X/Z) + cx
 *   v = fy * (Y/Z) + cy
 *
 * @param point  3D point in camera coordinate frame (metres)
 * @return       Corresponding 2D pixel coordinates (u, v)
 */
Eigen::Vector2d Camera::project(const Eigen::Vector3d& point) const
{
    // Extract individual coordinates for readability
    const double X{point.x()};
    const double Y{point.y()};
    const double Z{point.z()};

    // Normalise to the unit plane by dividing by depth (Z)
    // This removes the depth component, leaving angular position
    const double x_prime{X / Z};
    const double y_prime{Y / Z};

    // Apply intrinsics to map from normalised plane to pixel coordinates
    const double u{fx_ * x_prime + cx_};
    const double v{fy_ * y_prime + cy_};

    return Eigen::Vector2d(u, v);
}

/**
 * @brief Unprojects a 2D pixel and depth value to a 3D point in camera space.
 *
 * Reverses the pinhole projection:
 *   Z = depth / depth_scale
 *   X = (u - cx) / fx * Z
 *   Y = (v - cy) / fy * Z
 *
 * @param pixel  2D pixel coordinates (u, v)
 * @param depth  Raw depth value from the depth image (integer units)
 * @return       3D point in camera coordinate frame (metres)
 */
Eigen::Vector3d Camera::unproject(const Eigen::Vector2d& pixel, double depth) const
{
    // Extract pixel coordinates for readability
    const double u{pixel.x()};
    const double v{pixel.y()};

    // Convert raw depth to metres using the depth scale factor
    // depth_scale_ is typically 1000.0 for millimetre-encoded depth maps
    const double Z{depth / depth_scale_};

    // Shift pixel to be relative to the principal point,
    // then scale by inverse focal length to recover normalised coordinates
    const double x_prime{(u - cx_) / fx_};
    const double y_prime{(v - cy_) / fy_};

    // Recover metric 3D coordinates by scaling with depth
    const double X{x_prime * Z};
    const double Y{y_prime * Z};

    return Eigen::Vector3d(X, Y, Z);
}

} // namespace navio