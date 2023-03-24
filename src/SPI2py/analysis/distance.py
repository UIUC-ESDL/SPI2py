"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import numpy as np
from numba import njit
from typing import Union


# @njit(cache=True)
def distances_points_points(a: np.ndarray,
                            b: np.ndarray) -> np.ndarray:
    """
    Calculates the pairwise distance between two sets of points.

    This implementation utilizes array broadcasting to calculate the pairwise distance between two sets of points.
    Specifically, it calculates the Cartesian product of the Euclidean distance for two sets of points.

    TODO Write unit tests with Scipy cdist
    TODO Enable NJIT
    TODO Write unit tests

    :param a: Set of 3D points, (-1, 3) ndarray
    :param b: Set of 3D points, (-1, 3) ndarray
    :return: Euclidean distances, (-1,) np.ndarray
    """

    c = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    return c.reshape(-1)


# @njit(cache=True)
def signed_distances_spheres_spheres(a:        np.ndarray,
                                     a_radii:  np.ndarray,
                                     b:        np.ndarray,
                                     b_radii:  np.ndarray) -> np.ndarray:
    """
    Calculates the pairwise signed distance between two sets of spheres.

    Convention:
    Signed Distance < 0 means no overlap
    Signed Distance = 0 means tangent
    Signed Distance > 0 means overlap

    TODO Write unit tests
    TODO Enable NJIT
    TODO Reformat Radii shape so we don't have to keep reshaping it
    TODO Update constraint functions to use this function

    :param a: Set of 3D points, (-1, 3) ndarray
    :param a_radii: Set of radii, (-1) ndarray
    :param b: Set of 3D points, (-1, 3) ndarray
    :param b_radii: Set of radii, (-1) ndarray
    :return: Signed distance, float
    """

    delta_positions = distances_points_points(a, b)
    delta_radii     = distances_points_points(a_radii.reshape(-1, 1), b_radii.reshape(-1, 1))

    signed_distances = delta_radii - delta_positions

    return signed_distances


@njit(cache=True)
def minimum_distance_segment_segment(a: np.ndarray,
                                     b: np.ndarray,
                                     c: np.ndarray,
                                     d: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Returns the minimum Euclidean distance between two line segments.

    This function also works for calculating the distance between a line segment and a point and a point and point.

    Based on the algorithm described in:

    Vladimir J. Lumelsky,
    "On Fast Computation of Distance Between Line Segments",
    Information Processing Letters 21 (1985) 55-61
    https://doi.org/10.1016/0020-0190(85)90032-8

    Values 0 <= t <= 1 correspond to points being inside segment AB whereas values < 0  correspond to being 'left' of AB
    and values > 1 correspond to being 'right' of AB.

    Values 0 <= u <= 1 correspond to points being inside segment CD whereas values < 0  correspond to being 'left' of CD
    and values > 1 correspond to being 'right' of CD.

    Step 1: Check for special cases; compute D1, D2, and the denominator in (11)
        (a) If one of the two segments degenerates into a point, assume that this segment corresponds to the parameter
        u, take u=0, and go to Step 4.
        (b) If both segments degenerate into points, take t=u=0, and go to Step 5.
        (c) If neither of two segments degenerates into a point and the denominator in (11) is zero, take t=0 and go to
        Step 3.
        (d) If none of (a), (b), (c) takes place, go to Step 2.
    Step 2: Using (11) compute t. If t is not in the range [0,1], modify t using (12).
    Step 3: Using (10) compute u. If u is not in the range [0,1], modify u using (12); otherwise, go to Step 5.
    Step 4: Using (10) compute t. If t is not in the range [0,1], modify t using (12).
    Step 5: With current values of t and u, compute the actual MinD using (7).

    (7):
    (10):
    (11):
    (12):

    TODO Vectorize
    TODO Compare runtime against Scipy's cdist for moderately sized arrays
    TODO Review publication

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
    den = D1 * D2 - R**2

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

    min_dist          = np.linalg.norm(d1*t - d2*u - d12)
    min_dist_position = a + d1*t

    return min_dist, min_dist_position


def minimum_signed_distance_capsule_capsule(a:        np.ndarray,
                                            b:        np.ndarray,
                                            ab_radii: np.ndarray,
                                            c:        np.ndarray,
                                            d:        np.ndarray,
                                            cd_radii: np.ndarray) -> float:
    """
    Returns the minimum signed distance between two capsules.

    Since we approximate objects such as line segments with a collection of spheres, approximating a line segment with a
    large number of spheres will begin to resemble a capsule.

    Convention:
    Signed Distance < 0 means no overlap
    Signed Distance = 0 means tangent
    Signed Distance > 0 means overlap

    Assumes:
    1. All radii in line AB are the same and all radii in line CD are the same

    TODO Validate that this function works
    TODO Write unit tests
    TODO Enable NJIT
    """

    # Verify assumption 1
    assert np.all(ab_radii == ab_radii[0])
    assert np.all(cd_radii == cd_radii[0])

    minimum_distance, _ = minimum_distance_segment_segment(a, b, c, d)

    # TODO Verify this is the correct convention
    minimum_signed_distance = minimum_distance - (ab_radii[0] + cd_radii[0])

    return minimum_signed_distance



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



