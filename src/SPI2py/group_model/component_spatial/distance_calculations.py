"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import jax.numpy as np
from jax import numpy as np


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

    # TODO REMOVE SQUARED?

    # # Reshape the arrays for broadcasting
    aa = a.reshape(-1, 1, 3)
    bb = b.reshape(1, -1, 3)
    cc = aa-bb

    cc = cc.reshape(-1, 3)

    # c_squared_euclidean_distances =

    c = np.linalg.norm(cc, axis=-1)

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


def signed_distances(x, model, object_pair):
    """
    Returns the signed distances between all pairs of objects in the layout.

    To be consistent with constraint function notation, this function returns negative values
    for objects that are not interfering with each other, and positive values for objects that
    are interfering with each other.

    TODO Preallocate array and vectorize this function
    TODO Write unit tests

    :param x: Design vector (1D array)
    :param model: The SpatialConfiguration object used to query positions at x
    :param object_pair: The list of object pairs to calculate the signed distance between
    :return: An array of signed distances between each object pair
    """
    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = model.calculate_positions(design_vector=x)

    # Calculate the interferences between each sphere of each object pair
    all_signed_distances = []

    # TODO MAKE ORDER STATIC
    for obj1, obj2 in object_pair:

        positions_a = positions_dict[str(obj1)]['positions']
        radii_a = positions_dict[str(obj1)]['radii']

        positions_b = positions_dict[str(obj2)]['positions']
        radii_b = positions_dict[str(obj2)]['radii']

        signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b).flatten()
        all_signed_distances.append(signed_distances)


    all_signed_distances = np.concatenate(all_signed_distances, axis=0)

    return all_signed_distances
