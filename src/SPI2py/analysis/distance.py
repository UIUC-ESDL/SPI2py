"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import numpy as np
from numba import njit


# @njit(cache=True)
def minimum_distance_points_points(a: np.ndarray,
                                   b: np.ndarray) -> float:
    """
    Calculates the minimum distance between two sets of points.

    This implementation utilizes array broadcasting to calculate the pairwise distance between two sets of points.
    Specifically, it calculates the Cartesian product of the Euclidean distance for two sets of points.

    TODO Write unit tests with Scipy cdist
    TODO Enable NJIT
    TODO Write unit tests

    :param a: Set of 3D points, (n, 3) ndarray
    :param b: Set of 3D points, (m, 3) ndarray
    :return: Minimum distance, float
    """

    c = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    return c.reshape(-1)

def minimum_signed_distance_points_points(a:        np.ndarray,
                                          a_radii:  np.ndarray,
                                          b:        np.ndarray,
                                          b_radii:  np.ndarray) -> float:
    """
    Calculates the minimum signed distance between two sets of spheres.
    """





# @njit(cache=True)
def min_spheres_linesegment_distance(points, a, b):
    """
    Finds the minimum distance between a set 3D point and a line segment [a,b].

    With hierarchical collision detection we represent classes as recursive sphere trees.
    First,

    interference<0 means no overlap
    interference=0 means tangent
    interference>0 means overlap

    TODO Modify function to handle points not spheres
    TODO Modify function calls to provide a list of point(s) instead of a single point
    TODO Fix documentation
    TODO Vectorize with broadcasting
    TODO Enable NJIT
    TODO Compare runtime to Scipy's cdist
    TODO Compare runtime to min_linesegment_linesegment_distance
    TODO Write unit tests

    :param points: list of
    :param a: (3,)
    :param b: (3,)
    :return:
    """

    min_distances = []
    for point in points:
        min_point_distance = np.linalg.norm(np.dot(point - b, a - b) / np.dot(a - b, a - b) * (a - b) + b - point)
        min_distances.append(min_point_distance)

    min_distance = np.min(min_distances)

    return min_distance


# @njit(cache=True)
def minimum_distance_linesegment_linesegment(a: np.ndarray,
                                             b: np.ndarray,
                                             c: np.ndarray,
                                             d: np.ndarray) -> float:
    """
    Returns the minimum distance between two line segments.

    This function also works for calculating the distance between a line segment and a point and a point and point.

    Based on the algorithm described in:

    Vladimir J. Lumelsky,
    "On Fast Computation of Distance Between Line Segments",
    Information Processing Letters 21 (1985) 55-61
    https://doi.org/10.1016/0020-0190(85)90032-8

    TODO Vectorize
    TODO Enable NJIT
    TODO Compare runtime against Scipy's cdist for moderately sized arrays

    :param a: (1,3) numpy array
    :param b: (1,3) numpy array
    :param c: (1,3) numpy array
    :param d: (1,3) numpy array

    :return: Minimum distance between line segments, float
    """

    def clamp_bound(num):
        """
        If the number is outside the range [0,1] then clamp it to the nearest boundary.
        """
        if num < 0.:
            return 0.
        elif num > 1.:
            return 1.
        else:
            return num

    d1  = b - a
    d2  = d - c
    d12 = c - a

    D1  = np.dot(d1, d1.T)
    D2  = np.dot(d2, d2.T)
    S1  = np.dot(d1, d12.T)
    S2  = np.dot(d2, d12.T)
    R   = np.dot(d1, d2.T)
    den = np.dot(D1, D2) - np.square(R)

    # Check if one or both line segments are points
    if D1 == 0. or D2 == 0.:

        # Both AB and CD are points
        if D1 == 0. and D2 == 0.:
            t = 0.
            u = 0.

        # AB is a line segment and CD is a point
        elif D1 != 0.:
            u = 0.
            t = S1/D1
            t = clamp_bound(t)

        # AB is a point and CD is a line segment
        elif D2 != 0.:
            t = 0.
            u = -S2/D2
            u = clamp_bound(u)

    # Check if line segments are parallel
    elif den == 0.:
        t = 0.
        u = -S2/D2
        uf = clamp_bound(u)

        if uf != u:
            t = (uf*R + S1)/D1
            t = clamp_bound(t)
            u = uf

    # General case for calculating the minimum distance between two line segments
    else:

        t = (S1 * D2 - S2 * R) / den

        t = clamp_bound(t)

        u = (t * R - S2) / D2
        uf = clamp_bound(u)

        if uf != u:
            t = (uf * R + S1) / D1
            t = clamp_bound(t)

            u = uf

    dist = np.linalg.norm(d1*t - d2*u - d12)

    return dist



# TODO Implement KD Tree distance
# def min_kdtree_distance(tree, positions):
#     """
#     Returns the minimum distance between a KD-Tree and object.
#
#     In some cases, we need to check the distance between two sets of points, but one
#     set is extremely large (e.g., a complex structure) and thus the number of distance
#     calculations combinatorially grows and becomes prohibitive.
#
#     For static structures (e.g., a structure) we can construct a data structure (i.e., KD Tree)
#     once and then use it to efficiently perform distance calculations. Since the cost of
#     constructing a KD Tree is relatively high, and you must reconstruct it every time positions
#     change we do not use this for moving classes.
#
#     This function presumes the KD Tree is created when the object is initialized and thus
#     takes the tree as an argument instead of trying to create a tree from points every
#     function call.
#
#     interference<0 means no overlap
#     interference=0 means tangent
#     interference>0 means overlap
#
#     TODO Complete documentation
#     TODO Write unit tests
#
#     :param tree:
#     :param positions:
#     :return:
#     """
#
#     # tree.query returns distances and IDs. We only care about the distances
#     dist, _ = tree.query(positions)
#     min_dist = np.min(dist)
#
#     return min_dist



