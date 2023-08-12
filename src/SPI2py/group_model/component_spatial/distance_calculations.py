"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import jax.numpy as np


def distances_points_points(a: np.ndarray,
                            b: np.ndarray) -> np.ndarray:
    """
    Calculates the pairwise distance between two sets of 3D points.

    This implementation utilizes array broadcasting to calculate the pairwise distance between two sets of points.
    Specifically, it uses array broadcasting to generate the outer (Cartesian) product of each set of points. It then
    calculates the elementwise Euclidean distance between the two outer product matrices (the norm of a_i - b_i).

    Example:

    a = np.array([[11, 12, 13], [21, 22, 23], [31, 32, 33]])
    b = np.array([[41, 42, 43], [51, 52, 53]])

    aa = a[:, None, :] = array([[[11, 12, 13]],
                                [[21, 22, 23]],
                                [[31, 32, 33]]])

    bb = b[None, :, :] = array([[[41, 42, 43],
                                 [51, 52, 53]]])

    c = np.linalg.norm(aa - bb, axis=-1) = array([[[-30, -30, -30],
                                                   [-40, -40, -40]],
                                                  [[-20, -20, -20],
                                                   [-30, -30, -30]],
                                                  [[-10, -10, -10],
                                                   [-20, -20, -20]]])

    :param a: Set of 3D points, (-1, 3) ndarray
    :param b: Set of 3D points, (-1, 3) ndarray
    :return: Euclidean distances, (-1,) np.ndarray
    """

    # # Reshape the arrays for broadcasting


    aa = a.reshape(-1, 1, 3)
    bb = b.reshape(1, -1, 3)

    c = np.linalg.norm(aa-bb, axis=-1)

    # Reshape the output to a 1D array
    c = c.flatten()

    return c

def sum_radii(a, b):
    # Sum radii
    aa = a.reshape(-1, 1, 1)
    bb = b.reshape(1, -1, 1)

    c = aa + bb

    c = c.flatten()

    return c


def signed_distances_spheres_spheres(centers_a: np.ndarray,
                                     radii_a:   np.ndarray,
                                     centers_b: np.ndarray,
                                     radii_b:   np.ndarray) -> np.ndarray:
    """
    Calculates the pairwise signed distance between two sets of spheres.

    Convention:
    Signed Distance < 0 means no overlap
    Signed Distance = 0 means tangent
    Signed Distance > 0 means overlap

    TODO Write unit tests
    TODO Reformat Radii shape so we don't have to keep reshaping it

    :param centers_a: Set of 3D points, (-1, 3) ndarray
    :param radii_a: Set of radii, (-1) ndarray
    :param centers_b: Set of 3D points, (-1, 3) ndarray
    :param radii_b: Set of radii, (-1) ndarray
    :return: Signed distance, float
    """

    delta_positions = distances_points_points(centers_a, centers_b)
    delta_radii     = sum_radii(radii_a, radii_b)

    signed_distances = delta_radii - delta_positions

    signed_distances.flatten()

    return signed_distances



