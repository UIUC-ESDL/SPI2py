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

# def rigid_transformation(reference_position, positions, x, y, z, rx, ry, rz, reference_axes='origin'):
#     """
#     Apply translation and rotation to an object.

#     TODO Evaluate lower dimensional representations, e.g. screw
#     <https://en.wikipedia.org/wiki/Chasles%27_theorem_(kinematics)>

#     TODO Add support for non-origin coordinate systems (e.g., object dependencies)
#     """

#     # Convert rotation angles to radians
#     rx = np.radians(rx)
#     ry = np.radians(ry)
#     rz = np.radians(rz)

#     # Calculate rotation matrix
#     R_x = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
#     R_y = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
#     R_z = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
#     R = np.dot(R_z, np.dot(R_y, R_x))

#     # Create homogeneous transformation matrix
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = [x, y, z]

#     # Apply transformation
#     transformed_positions = np.dot(T, np.vstack((positions.T, np.ones(len(positions))))).T[:, :3]

#     return transformed_positions


# def rigid_transformation(reference_position, positions, x, y, z, rx, ry, rz, reference_axes='origin'):
#     # Convert rotation angles to radians
#     rx = np.radians(rx)
#     ry = np.radians(ry)
#     rz = np.radians(rz)

#     # Calculate rotation matrix
#     R_x = np.array([[1, 0, 0],
#                     [0, np.cos(rx), -np.sin(rx)],
#                     [0, np.sin(rx), np.cos(rx)]])
#     R_y = np.array([[np.cos(ry), 0, np.sin(ry)],
#                     [0, 1, 0],
#                     [-np.sin(ry), 0, np.cos(ry)]])
#     R_z = np.array([[np.cos(rz), -np.sin(rz), 0],
#                     [np.sin(rz), np.cos(rz), 0],
#                     [0, 0, 1]])
#     R = R_z @ R_y @ R_x

#     # Create homogeneous transformation matrix
#     T = np.eye(4)
#     T[:3, :3] = R
#     T[:3, 3] = [x, y, z]

#     # Apply transformation
#     positions_homogeneous = np.hstack((positions, np.ones((len(positions), 1))))
#     transformed_positions_homogeneous = positions_homogeneous @ T.T
#     transformed_positions = transformed_positions_homogeneous[:, :3]

#     return transformed_positions



# def test_translation_only():
#     # Test translation only
#     reference_position = np.array([0, 0, 0])
#     positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     x, y, z = 1, 2, 3
#     rx, ry, rz = 0, 0, 0
#     expected_transformed_positions = np.array([[2, 4, 6], [5, 7, 9], [8, 10, 12]])
#     transformed_positions = rigid_transformation(reference_position, positions, x, y, z, rx, ry, rz)
#     np.testing.assert_allclose(transformed_positions, expected_transformed_positions)

# def test_rotation_only():
#     # Test rotation only
#     reference_position = np.array([0, 0, 0])
#     positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     x, y, z = 0, 0, 0
#     rx, ry, rz = 90, 0, 0
#     expected_transformed_positions = np.array([[1, -3, 2], [4, -6, 5], [7, -9, 8]])
#     transformed_positions = rigid_transformation(reference_position, positions, x, y, z, rx, ry, rz)
#     np.testing.assert_allclose(transformed_positions, expected_transformed_positions)

# def test_translation_and_rotation():
#     # Test translation and rotation
#     reference_position = np.array([0, 0, 0])
#     positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     x, y, z = 1, 2, 3
#     rx, ry, rz = 90, 0, 0
#     expected_transformed_positions = np.array([[1, -1, 4], [4, -2, 7], [7, -3, 10]])
#     transformed_positions = rigid_transformation(reference_position, positions, x, y, z, rx, ry, rz)
#     np.testing.assert_allclose(transformed_positions, expected_transformed_positions)


# def test_translation_and_rotation():
#     # Test translation and rotation
#     reference_position = np.array([0, 0, 0])
#     positions = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     x, y, z = 1, 2, 3
#     rx, ry, rz = 90, 90, 90
#     expected_transformed_positions = np.array([[-2, 4, 8], [-5, 7, 10], [-8, 10, 12]])
#     transformed_positions = rigid_transformation(reference_position, positions, x, y, z, rx, ry, rz)
#     np.testing.assert_allclose(transformed_positions, expected_transformed_positions)