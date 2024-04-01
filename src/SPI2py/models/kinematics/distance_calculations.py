"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import jax.numpy as np


def distances_points_points(a: np.ndarray,
                            b: np.ndarray) -> np.ndarray:

    # # Reshape the arrays for broadcasting
    aa = a.reshape(-1, 1, 3)
    bb = b.reshape(1, -1, 3)
    cc = aa-bb

    c = np.linalg.norm(cc, axis=2)

    return c

def sum_radii(a, b):

    aa = a.reshape(-1, 1)
    bb = b.reshape(1, -1)

    c = aa + bb

    return c



def signed_distances_spheres_spheres(centers_a: np.ndarray,
                                     radii_a:   np.ndarray,
                                     centers_b: np.ndarray,
                                     radii_b:   np.ndarray) -> np.ndarray:

    delta_positions = distances_points_points(centers_a, centers_b)
    delta_radii     = sum_radii(radii_a, radii_b)

    signed_distances = delta_radii - delta_positions

    return signed_distances
