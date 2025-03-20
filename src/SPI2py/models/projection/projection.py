"""
Geometry Project by Norato...
"""
import numpy as np
import jax
from jax import lax
import jax.numpy as jnp
from chex import assert_shape, assert_type, assert_equal

from ..geometry.cylinders import create_cylinders
from ..geometry.intersection import volume_intersection_two_spheres
from ..mechanics.distance import minimum_distances_points_segments, minimum_distances_segments_segments, distances_points_points
from ..projection.mesh_kernels import apply_kernel
from ..geometry.spheres import get_aabb_indices
from ..utilities.aggregation import kreisselmeier_steinhauser_max, kreisselmeier_steinhauser_min
from jax import vmap




def project_component(grid_centers, grid_size,
                      obj_points, obj_radii,
                      kernel_points, kernel_radii):
    """
    Projects object points to the grid and calculates pseudo-densities.

    Parameters:
    ----------
    grid_centers : array, shape (nx, ny, nz, nk, 3)
        Grid element centers.
    grid_size : float
        Size of each grid element.
    obj_points : array, shape (no, 3)
        Object points in 3D space.
    obj_radii : array, shape (no, 1)
        Radii of the object points.
    kernel_points : array, shape (nk, 3)
        Points representing the mesh kernel.
    kernel_radii : array, shape (nk, 1)
        Radii of kernel points.

    Returns:
    --------
    all_densities : array, shape (nx, ny, nz)
        Pseudo-densities calculated for each grid element.
    kernel_points : array, shape (nxs, nys, nzs, nk, 3)
        Kernel points in the grid's active area.
    kernel_radii : array, shape (nxs, nys, nzs, nk, 1)
        Kernel radii in the grid's active area.
    """

    # Check the input shapes
    assert_shape(grid_centers, (None, None, None, None, 3))
    assert_shape(grid_size, (1,))
    assert_shape(obj_points, (None, 3))
    assert_shape(obj_radii, (None, 1))
    assert_shape(kernel_points, (None, 3))
    assert_shape(kernel_radii, (None, 1))

    # Check the input types
    assert_type(grid_centers, "float64")
    assert_type(grid_size, "float64")
    assert_type(obj_points, "float64")
    assert_type(obj_radii, "float64")
    assert_type(kernel_points, "float64")
    assert_type(kernel_radii, "float64")

    # Unpack the AABB indices
    i1, i2, j1, j2, k1, k2 = get_aabb_indices(grid_centers, grid_size, obj_points, obj_radii)

    # Extract grid dimensions
    grid_nx, grid_ny, grid_nz, _, _ = grid_centers.shape
    aabb_nx, aabb_ny, aabb_nz = (i2 - i1 + 1), (j2 - j1 + 1), (k2 - k1 + 1)
    obj_count, _ = obj_points.shape
    kernel_count, _ = kernel_points.shape

    # Initialize the output density array
    all_densities = jnp.zeros((grid_nx, grid_ny, grid_nz), dtype='float64')

    # Extract the active grid region within the object's AABB
    active_grid_centers = grid_centers[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1]

    # Apply the kernel to active grid elements
    kernel_points, kernel_radii = apply_kernel(active_grid_centers, grid_size, kernel_points, kernel_radii)

    # Calculate sample volumes and element volumes
    sample_volumes = (4 / 3) * jnp.pi * kernel_radii ** 3
    element_volumes = jnp.sum(sample_volumes, axis=3, keepdims=True)

    # Expand dimensions for broadcasting
    # Transpose object radii for broadcasting
    kernel_points_bc = kernel_points.reshape(aabb_nx, aabb_ny, aabb_nz, kernel_count, 1, 3)
    obj_points_bc = obj_points.reshape(1, 1, 1, 1, obj_count, 3)
    obj_radii_bc = obj_radii.T.reshape(1, 1, 1, 1, obj_count)

    # Compute distances between kernel and object points
    distances = distances_points_points(kernel_points_bc, obj_points_bc)

    # Calculate volume overlaps
    overlaps = volume_intersection_two_spheres(obj_radii_bc, kernel_radii, distances)
    element_overlaps = jnp.sum(overlaps, axis=4, keepdims=True)

    # Calculate volume fractions
    volume_fractions = (element_overlaps / element_volumes).reshape(aabb_nx, aabb_ny, aabb_nz, kernel_count)

    # Sum fractions to compute pseudo-densities
    densities = jnp.sum(volume_fractions, axis=3)

    # Store the densities in the output array
    all_densities = all_densities.at[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1].set(densities)

    # Penalize the densities
    all_penalized_densities = penalize_densities(all_densities)

    # Check the output values
    assert_equal(jnp.all(0 <= all_densities), True)
    assert_equal(jnp.all(all_densities <= 1), True)
    assert_equal(jnp.all(0 <= all_penalized_densities), True)
    assert_equal(jnp.all(all_penalized_densities <= 1), True)

    return all_densities, all_penalized_densities

def project_capsules(grid_centers, grid_size,
                     kernel_points, kernel_radii,
                     start_points, end_points, radii):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 3) tensor
    mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor

    mesh_positions_expanded: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor

    cylinder_starts: (n_segments, 3) tensor
    cylinder_stops: (n_segments, 3) tensor
    cylinder_radii: (n_segments, 1) tensor

    cylinder_starts_expanded: (1, 1, 1, 1, n_segments, 3) tensor
    cylinder_stops_expanded: (1, 1, 1, 1, n_segments, 3) tensor
    cylinder_radii_expanded: (1, 1, 1, 1, n_segments) tensor

    pseudo_densities: (n_el_x, n_el_y, n_el_z) tensor
    """

    # Combine
    capsule_points = jnp.concatenate([start_points, end_points], axis=0)
    capsule_radii = jnp.concatenate([radii, radii], axis=0)

    # Unpack the AABB indices
    i1, i2, j1, j2, k1, k2 = get_aabb_indices(grid_centers, grid_size, capsule_points, capsule_radii)

    # Extract grid dimensions
    grid_nx, grid_ny, grid_nz, _, _ = grid_centers.shape
    aabb_nx, aabb_ny, aabb_nz = (i2 - i1 + 1), (j2 - j1 + 1), (k2 - k1 + 1)
    capsule_count, _ = start_points.shape
    kernel_count, _ = kernel_points.shape

    # Initialize the output density array
    all_densities = jnp.zeros((grid_nx, grid_ny, grid_nz), dtype='float64')
    all_penalized_densities = jnp.zeros((grid_nx, grid_ny, grid_nz), dtype='float64')

    # Extract the active grid region within the object's AABB
    active_grid_centers = grid_centers[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1]

    # Apply the kernel to active grid elements
    kernel_points, kernel_radii = apply_kernel(active_grid_centers, grid_size, kernel_points, kernel_radii)

    # Expand the arrays to allow broadcasting
    # Transpose object radii for broadcasting
    kernel_points_bc = kernel_points.reshape(aabb_nx, aabb_ny, aabb_nz, kernel_count, 1, 3)
    start_points_bc = start_points.reshape(1, 1, 1, 1, capsule_count, 3)
    end_points_bc = end_points.reshape(1, 1, 1, 1, capsule_count, 3)
    radii_bc = radii.T.reshape(1, 1, 1, 1, capsule_count)

    # Vectorized signed distance and density calculations using your distance function
    distances = radii_bc - minimum_distances_points_segments(kernel_points_bc, start_points_bc, end_points_bc)

    # Calculate the pseudo-densities
    densities = density(distances, kernel_radii)

    # For mesh elements that are sub-sampled,
    # Add up the individual density contributions of each kernel point
    densities = jnp.sum(densities, axis=3)

    # Penalize the density of each element, for each capsule
    densities_penalized = penalize_densities(densities, penalty_factor=3)

    # Now, combine the densities of each capsule
    # Collapse the last axis to get the combined density for each kernel sphere
    densities_combined = kreisselmeier_steinhauser_max(densities, axis=3)
    penalized_densities_combined = kreisselmeier_steinhauser_max(densities_penalized, axis=3)

    # Store the densities in the output array
    all_densities = all_densities.at[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1].set(densities_combined)
    all_penalized_densities = all_penalized_densities.at[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1].set(penalized_densities_combined)

    # Check the output values
    assert_equal(jnp.all(0 <= all_densities), True)
    assert_equal(jnp.all(all_densities <= 1), True)
    assert_equal(jnp.all(0 <= all_penalized_densities), True)
    assert_equal(jnp.all(all_penalized_densities <= 1), True)

    return all_densities, all_penalized_densities


def project_interconnect(grid_centers, grid_size,
                         cyl_points, cyl_radius,
                         kernel_points, kernel_radii):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 3) tensor
    mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor

    mesh_positions_expanded: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor

    cylinder_starts: (n_segments, 3) tensor
    cylinder_stops: (n_segments, 3) tensor
    cylinder_radii: (n_segments, 1) tensor

    cylinder_starts_expanded: (1, 1, 1, 1, n_segments, 3) tensor
    cylinder_stops_expanded: (1, 1, 1, 1, n_segments, 3) tensor
    cylinder_radii_expanded: (1, 1, 1, 1, n_segments) tensor

    pseudo_densities: (n_el_x, n_el_y, n_el_z) tensor
    """

    # Create the cylinders
    cyl_starts, cyl_stops, cyl_rad = create_cylinders(cyl_points, cyl_radius)

    # Unpack the AABB indices
    i1, i2, j1, j2, k1, k2 = get_aabb_indices(grid_centers, grid_size, cyl_points, cyl_radius)

    # Extract grid dimensions
    grid_nx, grid_ny, grid_nz, _, _ = grid_centers.shape
    aabb_nx, aabb_ny, aabb_nz = (i2 - i1 + 1), (j2 - j1 + 1), (k2 - k1 + 1)
    cyl_count, _ = cyl_rad.shape
    kernel_count, _ = kernel_points.shape

    # Initialize the output density array
    all_densities = jnp.zeros((grid_nx, grid_ny, grid_nz), dtype='float64')

    # Extract the active grid region within the object's AABB
    active_grid_centers = grid_centers[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1]

    # Apply the kernel to active grid elements
    kernel_points, kernel_radii = apply_kernel(active_grid_centers, grid_size, kernel_points, kernel_radii)

    # # Calculate sample volumes and element volumes
    # sample_volumes = (4 / 3) * jnp.pi * kernel_radii ** 3
    # element_volumes = jnp.sum(sample_volumes, axis=3, keepdims=True)

    # Expand the arrays to allow broadcasting
    # Transpose object radii for broadcasting
    kernel_points_bc = kernel_points.reshape(aabb_nx, aabb_ny, aabb_nz, kernel_count, 1, 3)
    cyl_starts_bc = cyl_starts.reshape(1, 1, 1, 1, cyl_count, 3)
    cyl_stops_bc = cyl_stops.reshape(1, 1, 1, 1, cyl_count, 3)
    cyl_rad_bc = cyl_rad.T.reshape(1, 1, 1, 1, cyl_count)

    # Vectorized signed distance and density calculations using your distance function
    distances = cyl_rad_bc - minimum_distances_points_segments(kernel_points_bc, cyl_starts_bc, cyl_stops_bc)

    # Fix rho for mesh_radii?
    densities = density(distances, kernel_radii)

    # For mesh elements that are sub-sampled,
    # Add up the individual density contributions of each kernel point
    densities = jnp.sum(densities, axis=3)

    # Penalize the density of each element, for each capsule
    densities_penalized = penalize_densities(densities, penalty_factor=3)

    # Now, combine the densities of each capsule
    # Collapse the last axis to get the combined density for each kernel sphere
    densities_combined = kreisselmeier_steinhauser_max(densities, axis=3)
    penalized_densities_combined = kreisselmeier_steinhauser_max(densities_penalized, axis=3)

    # Store the densities in the output array
    all_densities = all_densities.at[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1].set(densities_combined)
    all_penalized_densities = all_penalized_densities.at[i1:i2 + 1, j1:j2 + 1, k1:k2 + 1].set(
        penalized_densities_combined)

    # Check the output values
    assert_equal(jnp.all(0 <= all_densities), True)
    assert_equal(jnp.all(all_densities <= 1), True)
    assert_equal(jnp.all(0 <= all_penalized_densities), True)
    assert_equal(jnp.all(all_penalized_densities <= 1), True)

    return all_densities, all_penalized_densities



# def projected_density(self, phi, r):
#     x = phi / r
#     return jnp.where(x > 1, 1, jnp.where(x < -1, 0, self.H_tilde(x, self.options['dimension'])))


# def phi(x, x1, x2, r_b):
#
#     x21 = x2 - x1
#     l_b = jnp.linalg.norm(x21)
#     a_b = x21 / l_b
#     xe1 = x - x1
#     l_be = jnp.dot(xe1, a_b)
#     r_be = jnp.linalg.norm(xe1 - l_be * a_b)
#
#     d1 = jnp.linalg.norm(xe1)
#     d2 = jnp.linalg.norm(x - x2)
#
#     d_b = jnp.where(l_be <= 0, d1,
#                     jnp.where(l_be > l_b, d2, r_be))
#
#     # EQ 8 (order reversed, this is a confirmed typo in the paper)
#     phi_b = r_b - d_b
#
#     return phi_b

# def phi(x, x1, x2, r_b):
#     """
#     Compute the regularized signed distance (phi) between point(s) x and a
#     degenerate cylinder defined by endpoints x1 and x2 with radius r_b.
#
#     Parameters:
#       x: array, shape (..., 3) -- point(s) at which to evaluate phi.
#       x1: array, shape (..., 3) -- first endpoint of the cylinder.
#       x2: array, shape (..., 3) -- second endpoint of the cylinder.
#       r_b: array or scalar, shape (...) -- cylinder radius.
#
#     Returns:
#       phi_b: array, shape (...) -- computed signed distances.
#     """
#     # # Vector from x1 to x2.
#     # x21 = x2 - x1
#     #
#     # # Compute the length along the last axis.
#     # l_b = jnp.linalg.norm(x21, axis=-1, keepdims=True)  # shape (...,1)
#     #
#     # # Compute unit vector along the segment.
#     # a_b = x21 / l_b  # broadcast along last axis
#     # # Vector from x1 to x.
#     # xe1 = x - x1
#     #
#     # # Instead of using jnp.dot, sum the elementwise product along the last axis.
#     # l_be = jnp.sum(xe1 * a_b, axis=-1, keepdims=True)  # shape (...,1)
#     # # Distance perpendicular to the segment: norm of (x - x1 - projection).
#     # r_be = jnp.linalg.norm(xe1 - l_be * a_b, axis=-1)  # shape (...)
#     #
#     # # Distances from x to x1 and x2.
#     # d1 = jnp.linalg.norm(xe1, axis=-1)  # shape (...)
#     # d2 = jnp.linalg.norm(x - x2, axis=-1)  # shape (...)
#     #
#     # # Squeeze l_be and l_b so that we can compare as scalars.
#     # l_be_squeezed = jnp.squeeze(l_be, axis=-1)  # shape (...)
#     # l_b_squeezed = jnp.squeeze(l_b, axis=-1)  # shape (...)
#     #
#     # # Choose d_b based on where x projects relative to the segment.
#     # d_b = jnp.where(l_be_squeezed <= 0, d1,
#     #                 jnp.where(l_be_squeezed > l_b_squeezed, d2, r_be))
#
#     d_b = jnp.linalg.norm(x - x1, axis=-1)
#
#     # Equation (with order reversed):
#     phi_b = r_b - d_b
#     return phi_b


# def penalized_density(rho, alpha, q):
#     return (alpha * rho) ** q


# def phi(x, x1, x2, r_b):
#
#     # Convert output from JAX.numpy to numpy
#     d_be = jnp.array(min_dist_segment_segment(x, x, x1, x2))
#
#     # EQ 8 (order reversed, this is a confirmed typo in the paper)
#     phi_b = r_b - d_be
#
#     return phi_b


def regularized_Heaviside(x):
    H_tilde = 0.5 + 0.75 * x - 0.25 * x ** 3  # EQ 3 in 3D
    return H_tilde


def density(phi_b, r):
    ratio = phi_b / r  # EQ 2
    rho = jnp.where(ratio < -1, 0,
                    jnp.where(ratio > 1, 1,
                              regularized_Heaviside(ratio)))
    return rho


def penalize_densities(densities, penalty_factor=3, min_density=1e-3):
    """
    Penalizes the projected densities using SIMP and applies a minimum density.

    Parameters:
    - densities: Projected densities (array of shape (nx, ny, nz)).
    - penalty_factor: Exponent for the SIMP penalty.
    - min_density: Minimum allowable density for numerical stability.

    Returns:
    - Penalized densities.
    """

    # Apply SIMP penalization
    penalized_densities = densities ** penalty_factor

    return penalized_densities


def apply_minimum_density(densities, min_density=1e-3):
    """
    Applies a minimum density to the projected densities.

    Parameters:
    - densities: Projected densities (array of shape (nx, ny, nz)).
    - min_density: Minimum allowable density for numerical stability.

    Returns:
    - Densities with a minimum density applied.
    """

    # Apply a minimum density for stability
    stabilized_densities = jnp.maximum(densities, min_density)

    return stabilized_densities


def combine_densities(*densities, min_density=1e-3, penalty_factor=3):
    """
    Combines multiple densities by taking the minimum value at each grid point.

    Parameters:
    ----------
    densities : list of arrays
        List of densities to combine.

    Returns:
    --------
    combined_density : array
        Combined density values.
    """

    # Stack the densities along the last axis
    densities = jnp.stack(densities, axis=3)

    # Combine the densities
    combined_density = kreisselmeier_steinhauser_max(densities, axis=3)

    # Apply Solid Isotropic Material Penalization (SIMP) to the densities
    combined_density = penalize_densities(combined_density, penalty_factor=penalty_factor)

    # Apply a minimum density
    combined_density = apply_minimum_density(combined_density, min_density=min_density)

    return combined_density



