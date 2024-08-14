"""
From Norato's paper...
"""

import numpy as np
from ..kinematics.distance_calculations import minimum_distance_segment_segment


def signed_distance(x, x1, x2, r_b):

    # Convert output from JAX.numpy to numpy
    d_be = np.array(minimum_distance_segment_segment(x, x, x1, x2))

    phi_b = r_b - d_be

    return phi_b


def regularized_Heaviside(x):
    H_tilde = 0.5 + 0.75 * x - 0.25 * x ** 3  # EQ 3 in 3D
    return H_tilde


def density(phi_b, r):
    ratio = phi_b / r
    rho = np.where(ratio < -1, 0,
                   np.where(ratio > 1, 1,
                            regularized_Heaviside(ratio)))
    return rho


def calculate_densities(positions, radii, x1, x2, r):

    # Expand dimensions to allow broadcasting
    positions_expanded = positions[..., np.newaxis, :]  # shape becomes (n, m, o, p, 1, 3)
    x1_expanded = x1[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
    x2_expanded = x2[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 3)
    r_T_expanded = r.T[np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, q, 1)

    # Vectorized signed distance and density calculations using your distance function
    phi = signed_distance(positions_expanded, x1_expanded, x2_expanded, r_T_expanded)

    rho = density(phi, radii)

    # Sum densities across all cylinders
    combined_density = np.clip(np.sum(rho, axis=4), 0, 1)

    return combined_density