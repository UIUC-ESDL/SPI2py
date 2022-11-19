"""

"""

import numpy as np
from numpy import sin, cos
from numba import njit


@njit(cache=True)
def translate(current_positions, delta_position):
    """
    Translates a set of points based on the change in position of a reference point

    TODO Write unit tests for this function
    TODO Vectorize?

    :param current_positions:
    :param delta_position:
    :return:
    """

    delta_x, delta_y, delta_z = delta_position

    new_positions = current_positions + np.array([delta_x, delta_y, delta_z])

    return new_positions


@njit(cache=True)
def rotate(positions, rotation):
    """
    Rotates a set of points about the first 3D point in the array.

    Note: Why do I manually define the rotation matrix instead of use
    scipy.spatial.transform.Rotation? Because in the future, the goal is to apply algorithmic
    differentiation to these functions (for objective and constraint function gradients).
    At the time being, I do not believe AD would be compatible with scipy functions that are compiled
    in a different langauge (most of SciPy is actually written in other languages).

    TODO Write unit tests for this function
    TODO Vectorize?

    :param positions:
    :param rotation: Angle in radians
    :return: new_positions:
    """

    # Shift the object to origin
    reference_position = positions[0]
    origin_positions = positions - reference_position

    alpha, beta, gamma = rotation

    # Rotation matrix Euler angle convention r = r_z(gamma) @ r_y(beta) @ r_x(alpha)

    r_x = np.array([[1., 0., 0.],
                    [0., cos(alpha), -sin(alpha)],
                    [0., sin(alpha), cos(alpha)]])

    r_y = np.array([[cos(beta), 0., sin(beta)],
                    [0., 1., 0.],
                    [-sin(beta), 0., cos(beta)]])

    r_z = np.array([[cos(gamma), -sin(gamma), 0.],
                    [sin(gamma), cos(gamma), 0.],
                    [0., 0., 1.]])

    r = r_z @ r_y @ r_x

    # Transpose positions from [[x1,y1,z1],[x2... ] to [[x1,x2,x3],[y1,... ]
    rotated_origin_positions = (r @ origin_positions.T).T

    # Shift back from origin
    new_positions = rotated_origin_positions + reference_position

    return new_positions

