"""Module...
...
"""

# import numpy as np
from scipy.spatial.distance import cdist, pdist

import jax.numpy as jnp
from jax import grad, jit, vmap


def min_cdist(a, b):
    return jnp.min(cdist(a, b))


def min_cdist_jit(a, b):
    return jnp.min(cdist(a, b))


def min_kdtree_distance(tree, positions):
    """
    Returns the minimum distance between a KD-Tree and object

    :param tree:
    :param positions:
    :return:
    """

    dist, _ = tree.query(positions)
    min_dist = jnp.min(dist)

    return min_dist


@jit
def min_point_line_distance(p, a, b):
    """
    Function not tested for all conditions yet

    :param p: (3,)
    :param a: (3,)
    :param b: (3,)
    :return:
    """

    return jnp.linalg.norm(jnp.dot(p - b, a - b) / jnp.dot(a - b, a - b) * (a - b) + b - p)


# @jit
def min_line_line_distance(a0, a1, b0, b1):
    """
    Function not tested for all conditions yet

    Given two lines defined by numpy.array pairs (a0,a1,b0,b1)
    Return the closest points on each segment and their distance

    Make clamp = True for all cases

    for developing tests use https://stackoverflow.com/questions/2824478/shortest-distance-between-two-line-segments
    """
    clampAll = True
    clampA0 = True
    clampA1 = True
    clampB0 = True
    clampB1 = True


    # Calculate denomitator
    A = a1 - a0
    B = b1 - b0
    magA = jnp.linalg.norm(A)
    magB = jnp.linalg.norm(B)

    _A = A / magA
    _B = B / magB

    cross = jnp.cross(_A, _B);
    denom = jnp.linalg.norm(cross) ** 2

    # If lines are parallel (denom=0) test if lines overlap.
    # If they don't overlap then there is a closest point solution.
    # If they do overlap, there are infinite closest positions, but there is a closest distance
    if not denom:
        d0 = jnp.dot(_A, (b0 - a0))

        # Overlap only possible with clamping
        if clampA0 or clampA1 or clampB0 or clampB1:
            d1 = jnp.dot(_A, (b1 - a0))

            # Is segment B before A?
            if d0 <= 0 >= d1:
                if clampA0 and clampB1:
                    if jnp.absolute(d0) < jnp.absolute(d1):
                        return a0, b0, jnp.linalg.norm(a0 - b0)
                    return a0, b1, jnp.linalg.norm(a0 - b1)


            # Is segment B after A?
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if jnp.absolute(d0) < jnp.absolute(d1):
                        return a1, b0, jnp.linalg.norm(a1 - b0)
                    return a1, b1, jnp.linalg.norm(a1 - b1)

        # Segments overlap, return distance between parallel segments
        return jnp.linalg.norm(((d0 * _A) + a0) - b0)

    # Lines criss-cross: Calculate the projected closest points
    t = (b0 - a0);
    detA = jnp.linalg.det([t, _B, cross])
    detB = jnp.linalg.det([t, _A, cross])

    t0 = detA / denom;
    t1 = detB / denom;

    pA = a0 + (_A * t0)  # Projected closest point on segment A
    pB = b0 + (_B * t1)  # Projected closest point on segment B

    # Clamp projections
    if clampA0 or clampA1 or clampB0 or clampB1:
        if clampA0 and t0 < 0:
            pA = a0
        elif clampA1 and t0 > magA:
            pA = a1

        if clampB0 and t1 < 0:
            pB = b0
        elif clampB1 and t1 > magB:
            pB = b1

        # Clamp projection A
        if (clampA0 and t0 < 0) or (clampA1 and t0 > magA):
            dot = jnp.dot(_B, (pA - b0))
            if clampB0 and dot < 0:
                dot = 0
            elif clampB1 and dot > magB:
                dot = magB
            pB = b0 + (_B * dot)

        # Clamp projection B
        if (clampB0 and t1 < 0) or (clampB1 and t1 > magB):
            dot = jnp.dot(_A, (pB - a0))
            if clampA0 and dot < 0:
                dot = 0
            elif clampA1 and dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

    return pA, pB, jnp.linalg.norm(pA - pB)


def calculate_gap(radius1, radius2, min_dist):
    """
    Calculate the gap between two spheres

    gap<0 means no overlap
    gap=0 means tangent
    gap>0 means overlap

    :param radius1: int
    :param radius2:
    :param min_dist:
    :return:
    """

    return radius1 + radius2 - min_dist



