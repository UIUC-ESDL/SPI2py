"""
From Norato's paper...
"""

import numpy as np
from ..kinematics.distance_calculations_vectorized import minimum_distance_segment_segment


def phi_b(x, x1, x2, r_b):

    d_be = float(minimum_distance_segment_segment(x, x, x1, x2))

    # EQ 8 (order reversed)
    phi_b = r_b - d_be

    return phi_b


def H_tilde(x):
    # EQ 3 in 3D
    return 0.5 + 0.75 * x - 0.25 * x ** 3


def rho_b(phi_b, r):
    # EQ 2
    if phi_b / r < -1:
        return 0

    elif -1 <= phi_b / r <= 1:
        return H_tilde(phi_b / r)

    elif phi_b / r > 1:
        return 1

    else:
        raise ValueError('Something went wrong')


def calculate_density(position, radius, x1, x2, r):
    phi = phi_b(position, x1, x2, r)
    rho = rho_b(phi, radius)
    return rho

def calculate_combined_density(position, radius, X1, X2, R):

    combined_density = 0
    for x1, x2, r in zip(X1, X2, R):
        density = calculate_density(position, radius, x1, x2, r)
        combined_density += density

    # Clip the combined densities to be between 0 and 1
    combined_density = max(0, min(combined_density, 1))

    return combined_density

def calculate_combined_densities(positions, radii, X1, X2, R):
    n, m, o = positions.shape
    densities = np.zeros((n, m, o))
    for i in range(n):
        for j in range(m):
            for k in range(o):
                position = positions[i, j, k]
                radius = radii[i, j, k]
                combined_density = calculate_combined_density(position, radius, X1, X2, R)
                densities[i, j, k] += combined_density