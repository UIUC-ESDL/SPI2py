"""Distance calculations

Provides functions to calculate the distance between classes in various ways...

TODO I think it makes sense to get rid of the calculate gap function and to work it into the existing functions
    This will require adding the radii as an additional argument. It works now for uniform spheres, but MDBD will
    have different sized spheres.
"""

import numpy as np
from numba import njit
from scipy.spatial.distance import cdist


def max_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b):
    """
    Computes the minimum distance between two sets of spheres.

    interference<0 means no overlap
    interference=0 means tangent
    interference>0 means overlap

    TODO Complete documentation
    TODO Write unit tests
    TODO Vectorize?

    :param radii_b:
    :param positions_b:
    :param radii_a:
    :param positions_a:
    :return:
    """

    pairwise_distances = cdist(positions_a, positions_b)

    # TODO Reshape radii to 2D
    radii_a = radii_a.reshape(-1, 1)
    radii_b = radii_b.reshape(-1, 1)

    # TODO Shouldn't I be adding these... (?)
    pairwise_radii = cdist(radii_a, radii_b)

    pairwise_interferences = pairwise_radii - pairwise_distances

    max_interference = np.max(pairwise_interferences)

    return max_interference


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
    # TODO explain this comment better
    dist, _ = tree.query(positions)
    min_dist = np.min(dist)

    return min_dist


@njit(cache=True)
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
    TODO Vectorize?

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


@njit(cache=True)
def min_linesegment_linesegment_distance(a0, a1, b0, b1):
    """
    Returns the minimum distance between two line segments.

    Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance

    interference<0 means no overlap
    interference=0 means tangent
    interference>0 means overlap

    TODO Write unit tests
    TODO Document function logic more clearly
    TODO Vectorize?
    """

    # Calculate denominator
    A = a1 - a0
    B = b1 - b0
    magA = np.linalg.norm(A)
    magB = np.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = np.cross(_A, _B)
    denom = np.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance

    if not denom:
        d0 = np.dot(_A, (b0 - a0))

        # Overlap only possible with clamping

        d1 = np.dot(_A, (b1 - a0))

        # Is segment B before A?
        if d0 <= 0 >= d1:

            if np.absolute(d0) < np.absolute(d1):
                # Explain case
                return np.linalg.norm(a0 - b0)

            # Explain case
            return np.linalg.norm(a0 - b1)

        # Is segment B after A?
        elif d0 >= magA <= d1:

            if np.absolute(d0) < np.absolute(d1):
                # Explain case
                return np.linalg.norm(a1 - b0)

            # Explain case
            return np.linalg.norm(a1 - b1)

        # Case: Segments overlap --> Return distance between parallel segments
        return np.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points

    t = (b0 - a0)
    detA = np.linalg.det([t, _B, cross])
    detB = np.linalg.det([t, _A, cross])

    t0 = detA / denom
    t1 = detB / denom

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections

    if t0 < 0:
        pA = a0
    elif t0 > magA:
        pA = a1

    if t1 < 0:
        pB = b0
    elif t1 > magB:
        pB = b1

    # Clamp projection A
    if (t0 < 0) or (t0 > magA):
        dot = np.dot(_B, (pA - b0))
        if dot < 0:
            dot = 0
        elif dot > magB:
            dot = magB
        pB = b0 + (_B * dot)

    # Clamp projection B
    if (t1 < 0) or (t1 > magB):
        dot = np.dot(_A, (pB - a0))
        if dot < 0:
            dot = 0
        elif dot > magA:
            dot = magA
        pA = a0 + (_A * dot)

    return np.linalg.norm(pA - pB)

