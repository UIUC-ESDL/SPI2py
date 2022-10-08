"""

"""

import numpy as np
from numpy import sin, cos
from numba import njit


@njit(cache=True)
def translate(current_positions, delta_position):
    """
    Translates a set of points based on the change in position of a reference point

    TODO Write tests for this function

    :param current_positions:
    :param delta_position:
    :return:
    """

    delta_x, delta_y, delta_z = delta_position

    new_positions = current_positions + np.array([delta_x, delta_y, delta_z])

    return new_positions


@njit(cache=True)
def rotate(positions, delta_rotation):
    """
    Rotates a set of points based on the change of rotation of a reference point

    Need to verify the function rotates correctly (e.g., when not about the axis)
    Angles in radians...

    Note: Why do I manually define the rotation matrix instead of use
    scipy.spatial.transform.Rotation? Because in the future, the goal is to apply algorithmic
    differentiation to these functions (for objective and constraint function gradients).
    At the time being, I do not believe AD would be compatible with scipy functions that are compiled
    in a different langauge (most of SciPy is actually written in other languages).

    TODO Consider adding a reverse direction option
    TODO Write tests for this function

    :param positions:
    :param delta_rotation: Angle in radians!!!
    :return: new_positions:
    """

    # Shift the object to origin
    reference_position = positions[0]
    origin_positions = positions - reference_position

    alpha, beta, gamma = delta_rotation

    # Rotation matrix Euler angle convention r = r_z(gamma) @ r_y(beta) @ r_x(alpha)
    r = np.array([[cos(alpha)*cos(beta),
                   cos(alpha)*sin(beta)*sin(gamma)-sin(alpha)*cos(gamma),
                   cos(alpha)*sin(beta)*cos(gamma)+sin(alpha)*sin(gamma)],

                  [sin(alpha)*cos(beta),
                   sin(alpha)*sin(beta)*sin(gamma)+cos(alpha)*cos(gamma),
                   sin(alpha)*sin(beta)*cos(gamma)-cos(alpha)*sin(gamma)],

                  [-sin(beta),
                   cos(beta)*sin(gamma),
                   cos(beta)*cos(gamma)]])

    # Transpose positions from [[x1,y1,z1],[x2... ] to [[x1,x2,x3],[y1,... ]
    rotated_origin_positions = (r @ origin_positions.T).T

    # Shift back from origin
    new_positions = rotated_origin_positions + reference_position

    return new_positions

