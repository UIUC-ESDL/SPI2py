"""

"""

import numpy as np
from numpy import sin, cos


def translate(current_sphere_positions, current_reference_point, new_reference_point):
    """
    Translates a set of points based on the change in position of a reference point

    TODO Write unit tests for this function
    TODO Vectorize
    TODO Change function to take arguments: current_sphere_positions, current_reference_point, new_reference_point

    :param current_sphere_positions:
    :param current_reference_point:
    :param new_reference_point:
    :return:
    """

    delta_position = new_reference_point - current_reference_point

    new_sphere_positions = current_sphere_positions + delta_position

    return new_sphere_positions


# @njit(cache=True)
def rotate(positions, rotation):
    """
    Rotates a set of points about the first 3D point in the array.

    Note: Why do I manually define the rotation matrix instead of use
    scipy.spatial.transform.Rotation? Because in the future, the goal is to apply algorithmic
    differentiation to these functions (for objective and constraint function gradients).
    At the time being, I do not believe AD would be compatible with scipy functions that are compiled
    in a different langauge (most of SciPy is actually written in other languages).

    TODO TAKE REFERENCE POINT
    TODO Write unit tests for this function
    TODO Vectorize?

    :param positions:
    :param rotation: Angle in radians
    :return: new_positions:
    """

    # Shift the object to origin
    reference_position = positions[0]
    origin_positions = positions - reference_position

    # Unpack rotation angles
    # alpha, beta, gamma
    a, b, g = rotation

    # Rotation matrix Euler angle convention r = r_z(gamma) @ r_y(beta) @ r_x(alpha)
    #
    # r_x = np.array([[1., 0., 0.],
    #                 [0., cos(a), -sin(a)],
    #                 [0., sin(a), cos(a)]])
    #
    # r_y = np.array([[cos(b), 0., sin(b)],
    #                 [0., 1., 0.],
    #                 [-sin(b), 0., cos(b)]])
    #
    # r_z = np.array([[cos(g), -sin(g), 0.],
    #                 [sin(g), cos(g), 0.],
    #                 [0., 0., 1.]])
    #
    # r = r_z @ r_y @ r_x

    # Reassemble rotation matrix
    r = np.array(
        [[cos(b) * cos(g), sin(a) * sin(b) * cos(g) - cos(a) * sin(g), cos(a) * sin(b) * cos(g) + sin(a) * sin(g)],
         [cos(b) * sin(g), sin(a) * sin(b) * sin(g) + cos(a) * cos(g), cos(a) * sin(b) * sin(g) - sin(a) * cos(g)],
         [-sin(b), sin(a) * cos(b), cos(a) * cos(b)]])

    # Transpose positions from [[x1,y1,z1],[x2... ] to [[x1,x2,x3],[y1,... ]
    rotated_origin_positions = (r @ origin_positions.T).T

    # Shift back from origin
    new_positions = rotated_origin_positions + reference_position

    return new_positions


def rigid_transformation(reference_position, positions, x, y, z, rx, ry, rz, reference_axes='origin'):
    """
    Apply translation and rotation to an object.

    TODO Evaluate lower dimensional representations, e.g. screw
    <https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics)>

    TODO Add support for non-origin coordinate systems (e.g., object dependencies)


    """

    translation = np.array([[x, y, z]])
    rotation    = np.array([rx, ry, rz])

    translated_positions = positions + translation

    rotated_positions = rotate(translated_positions, rotation)

    return rotated_positions


def test_rotate():
    data = np.array(...)
    answer=runfunction(data)

    assert answer == expected_answer_calculatedbyhand
