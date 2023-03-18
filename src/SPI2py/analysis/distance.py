"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import numpy as np
from numba import njit


def euclidean_distances_of_cartesian_product(a, b):
    """
    Calculates the pairwise Euclidean distance between two sets of points.

    Utilizes array broadcasting to calculate the pairwise distance between two sets of points.

    TODO Write unit tests with Scipy cdist
    TODO Apply algorithmic differentiation

    :param a:
    :param b:
    :return:
    """

    c = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)

    return c.reshape(-1)


def min_kdtree_distance(tree, positions):
    """
    Returns the minimum distance between a KD-Tree and object.

    In some cases, we need to check the distance between two sets of points, but one
    set is extremely large (e.g., a complex structure) and thus the number of distance
    calculations combinatorially grows and becomes prohibitive.

    For static structures (e.g., a structure) we can construct a data structure (i.e., KD Tree)
    once and then use it to efficiently perform distance calculations. Since the cost of
    constructing a KD Tree is relatively high, and you must reconstruct it every time positions
    change we do not use this for moving classes.

    This function presumes the KD Tree is created when the object is initialized and thus
    takes the tree as an argument instead of trying to create a tree from points every
    function call.

    interference<0 means no overlap
    interference=0 means tangent
    interference>0 means overlap

    TODO Complete documentation
    TODO Write unit tests

    :param tree:
    :param positions:
    :return:
    """

    # tree.query returns distances and IDs. We only care about the distances
    dist, _ = tree.query(positions)
    min_dist = np.min(dist)

    return min_dist


# @njit(cache=True)
def min_spheres_linesegment_distance(points, a, b):
    """
    Finds the minimum distance between a set 3D point and a line segment [a,b].

    With hierarchical collision detection we represent classes as recursive sphere trees.
    First,

    interference<0 means no overlap
    interference=0 means tangent
    interference>0 means overlap

    TODO Modify function calls to provide a list of point(s) instead of a single point
    TODO Fix documentation
    TODO Vectorize

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


# # TODO Implement this function
# # @njit(cache=True)
# def min_linesegment_linesegment_distance(a0, a1, b0, b1):
#     """
#     Returns the minimum distance between two line segments.
#
#     Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
#     Return the closest points on each segment and their distance
#
#     TODO Write unit tests
#     TODO Document function logic more clearly
#     TODO Vectorize
#     """
#
#     # Calculate denominator
#     A = a1 - a0
#     B = b1 - b0
#
#     norm_a = np.linalg.norm(A)
#     norm_b = np.linalg.norm(B)
#
#     a_normalized = A / norm_a
#     b_normalized = B / norm_b
#
#     cross_product = np.cross(a_normalized, b_normalized)
#     denom = np.linalg.norm(cross_product) ** 2
#
#     # If lines are parallel (denom=0) test if lines overlap.
#     # If they don't overlap then there is a closest point solution.
#     # If they do overlap, there are infinite closest positions, but there is a closest distance
#
#     parallel = denom == 0
#     if not parallel:
#
#         d0 = np.dot(a_normalized, (b0 - a0))
#
#         # Overlap only possible with clamping
#
#         d1 = np.dot(a_normalized, (b1 - a0))
#
#         # Is segment B before A?
#         if d0 <= 0 >= d1:
#
#             if np.absolute(d0) < np.absolute(d1):
#                 # Explain case
#                 return np.linalg.norm(a0 - b0)
#
#             # Explain case
#             return np.linalg.norm(a0 - b1)
#
#         # Is segment B after A?
#         elif d0 >= norm_a <= d1:
#
#             if np.absolute(d0) < np.absolute(d1):
#                 # Explain case
#                 return np.linalg.norm(a1 - b0)
#
#             # Explain case
#             return np.linalg.norm(a1 - b1)
#
#         # Case: Segments overlap --> Return distance between parallel segments
#         return np.linalg.norm(((d0 * a_normalized) + a0) - b0)
#
#     elif parallel:
#
#         # Lines criss-cross: Calculate the projected closest points
#
#
#
#         t = (b0 - a0)
#         detA = np.linalg.det([t, b_normalized, cross_product])
#         detB = np.linalg.det([t, a_normalized, cross_product])
#
#         t0 = detA / denom
#         t1 = detB / denom
#
#         pA = a0 + (a_normalized * t0)  # Projected closest point on segment A
#         pB = b0 + (b_normalized * t1)  # Projected closest point on segment B
#
#         # Clamp projections
#
#         if t0 < 0:
#             pA = a0
#         elif t0 > norm_a:
#             pA = a1
#
#         if t1 < 0:
#             pB = b0
#         elif t1 > norm_b:
#             pB = b1
#
#         # Clamp projection A
#         if (t0 < 0) or (t0 > norm_a):
#             dot = np.dot(b_normalized, (pA - b0))
#             if dot < 0:
#                 dot = 0
#             elif dot > norm_b:
#                 dot = norm_b
#             pB = b0 + (b_normalized * dot)
#
#         # Clamp projection B
#         if (t1 < 0) or (t1 > norm_b):
#             dot = np.dot(a_normalized, (pB - a0))
#             if dot < 0:
#                 dot = 0
#             elif dot > norm_a:
#                 dot = norm_a
#             pA = a0 + (a_normalized * dot)
#
#         return np.linalg.norm(pA - pB)



def min_linesegment_linesegment_distance(a: np.ndarray,
                                         b: np.ndarray,
                                         c: np.ndarray,
                                         d: np.ndarray) -> float:
    """
    Returns the minimum distance between two line segments.

    Based on the algorithm described in:

    Vladimir J. Lumelsky,
    "On Fast Computation of Distance Between Line Segments",
    Information Processing Letters 21 (1985) 55-61
    https://doi.org/10.1016/0020-0190(85)90032-8

    :param a: (1,3) numpy array
    :param b: (1,3) numpy array
    :param c: (1,3) numpy array
    :param d: (1,3) numpy array

    :return: Minimum distance between line segments, float
    """

    def fix_bound(num):
        """
        If the number is outside the range [0,1] it is clamped to the nearest boundary.
        """
        if num < 0:
            return 0
        elif num > 1:
            return 1
        else:
            return num

    d1 = b - a
    d2 = d - c
    d12 = c - a

    D1 = np.dot(d1, d1.T)
    D2 = np.dot(d2, d2.T)

    S1 = np.dot(d1, d12.T)
    S2 = np.dot(d2, d12.T)
    R = d1*d2.T
    den = np.dot(D1, D2) - np.square(R)


    # Check if one or both line segments are points
    if D1 == 0 or D2 == 0:

        # Both AB and CD are points
        if D1 == 0 and D2 == 0:
            t = 0
            u = 0

        # AB is a line segment and CD is a point
        elif D1 != 0:
            u = 0
            t = S1/D1
            t = fix_bound(t)

        # AB is a point and CD is a line segment
        elif D2 != 0:
            t = 0
            u = -S2/D2
            u = fix_bound(u)


    # Check if line segments are parallel
    elif den == 0:
        t = 0
        u = -S2/D2
        uf = fix_bound(u)

        if uf != u:
            t = (uf*R + S1)/D1
            t = fix_bound(t)
            u = uf


    # General case
    else:

        t = (S1 * D2 - S2 * R) / den;

        t = fix_bound(t);

        u = (t * R - S2) / D2;
        uf = fix_bound(u);

        if uf != u:
            t = (uf * R + S1) / D1;
            t = fix_bound(t);

            u = uf;

    dist = np.linalg.norm(d1 * t - d2 * u - d12)

    if



