"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import autograd.numpy as np
from typing import Union

import numpy as np
from autograd import grad
from itertools import combinations
from scipy.spatial.distance import cdist

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

    # Radii
    if a.shape[1] != 3 or b.shape[1] != 3:
        aa = a.reshape(-1, 1, 1)
        bb = b.reshape(1, -1, 1)

    # Points
    else:
        aa = a.reshape(-1, 1, 3)
        bb = b.reshape(1, -1, 3)

    c = np.linalg.norm(aa-bb, axis=-1)

    # Reshape the output to a 1D array
    c = c.flatten()

    return c


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
    TODO Reformat Radii shape so we don't have to keep reshaping it

    :param a: Set of 3D points, (-1, 3) ndarray
    :param a_radii: Set of radii, (-1) ndarray
    :param b: Set of 3D points, (-1, 3) ndarray
    :param b_radii: Set of radii, (-1) ndarray
    :return: Signed distance, float
    """

    # Reshape radii
    a_radii = a_radii.reshape(-1, 1)
    b_radii = b_radii.reshape(-1, 1)

    delta_positions = distances_points_points(a, b)
    delta_radii     = distances_points_points(a_radii, b_radii)

    signed_distances = delta_radii - delta_positions

    signed_distances.flatten()

    return signed_distances


# @njit(cache=True)
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

    minimum_distance          = np.linalg.norm(d1*t - d2*u - d12)
    minimum_distance_position = a + d1*t

    return minimum_distance, minimum_distance_position


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

def aggregate_pairwise_distance(x, model):
    """
    Aggregates the distance between each 2-pair of classes

    This function does not work well because its value becomes very large very quickly.
    The gradient-based solver tends to "ignore" constraint functions that produce consraints
    several orders of magnitude smaller, even when you force feasibility.

    :param x:
    :param model:
    :return:
    """

    # Calculate the position of every sphere based on design vector x
    positions_dict = model.calculate_positions(design_vector=x)

    # Create a list of object pairs
    object_pairs = list(combinations(positions_dict.keys(), 2))

    objective = 0
    for object_pair in object_pairs:
        object_1 = object_pair[0]
        object_2 = object_pair[1]

        positions_1 = positions_dict[object_1][0]
        positions_2 = positions_dict[object_2][0]

        objective += sum(sum(cdist(positions_1, positions_2)))

    return objective


def normalized_aggregate_gap_distance(x, model):
    """
    Returns the normalized gap

    :param x:
    :param model:
    :return:
    """

    # Evaluate the model at the design vector x
    # Calculate the position of every sphere based on design vector x
    positions_dict = model.calculate_positions(design_vector=x)

    # Create a list of object pairs
    object_pairs = list(combinations(positions_dict.keys(), 2))

    objective = []
    for object_pair in object_pairs:
        object_1 = object_pair[0]
        object_2 = object_pair[1]

        positions_1 = positions_dict[object_1][0]
        positions_2 = positions_dict[object_2][0]

        objective.append(sum(sum(cdist(positions_1, positions_2))))

    # Divide by number of components
    objective = np.sum(objective) / len(objective)

    return objective


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

    for obj1, obj2 in object_pair:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = positions_dict[str(obj1)][1]

        positions_b = positions_dict[str(obj2)][0]
        radii_b = positions_dict[str(obj2)][1]

        # If line line vs cdist

        # # TODO Reformat Radii shape so we don't have to keep reshaping it
        # dPositions = distances_points_points(positions_a, positions_b)
        # dRadii     = distances_points_points(radii_a.reshape(-1, 1), radii_b.reshape(-1, 1))
        #
        # all_signed_distances.append(dRadii - dPositions)
        signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b).flatten()


        all_signed_distances.append(signed_distances)

    all_signed_distances = np.concatenate(all_signed_distances, axis=0)

    return all_signed_distances
