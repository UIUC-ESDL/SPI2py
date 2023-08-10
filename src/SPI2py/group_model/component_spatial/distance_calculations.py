"""Distance calculations

Provides functions to calculate the distance between classes in various ways.
"""

import jax.numpy as np
from itertools import combinations, product

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


# TODO Figure out jax slowdown...
# TODO REPLACE THE FLOAT VALUE FOR ARRAY
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
            return np.array(0.)
        elif num > 1.:
            return np.array(1.)
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
            t = np.array(0.)
            u = np.array(0.)

        # AB is a line segment and CD is a point
        elif D1 != 0.:
            u = np.array(0.)
            t = S1/D1
            t = clamp_bound(t)

        # AB is a point and CD is a line segment
        elif D2 != 0.:
            t = np.array(0.)
            u = -S2/D2
            u = clamp_bound(u)

    # Check if line segments are parallel
    elif den == 0.:
        t = np.array(0.)
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


class DistanceFunctions3D:
    pass



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


def signed_distances_capsules_capsules(a:        np.ndarray,
                                       b:        np.ndarray,
                                       ab_radii: np.ndarray,
                                       c:        np.ndarray,
                                       d:        np.ndarray,
                                       cd_radii: np.ndarray) -> np.ndarray:

    capsules_ab = [(ai, bi, ab_radius) for ai, bi, ab_radius in zip(a, b, ab_radii)]
    capsules_cd = [(ci, di, cd_radius) for ci, di, cd_radius in zip(c, d, cd_radii)]

    # TODO Precalculate this value
    capsule_pairs = list(product(capsules_ab, capsules_cd))

    signed_distances = []
    for capsule_pair in capsule_pairs:
        ai, bi, ab_radius = capsule_pair[0]
        ci, di, cd_radius = capsule_pair[1]

        minimum_distance, _ = minimum_distance_segment_segment(ai, bi, ci, di)
        minimum_signed_distance = (ab_radius + cd_radius) - minimum_distance

        signed_distances.append(minimum_signed_distance)

    return np.array(signed_distances)



def signed_distances_spheres_capsules(centers:  np.ndarray,
                                      radii:    np.ndarray,
                                      a:        np.ndarray,
                                      b:        np.ndarray,
                                      ab_radii: np.ndarray) -> np.ndarray:

    return signed_distances_capsules_capsules(centers, centers, radii, a, b, ab_radii)




