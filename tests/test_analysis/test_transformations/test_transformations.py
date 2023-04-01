"""
TODO write tests to ensure negative values are same e.g., 90 degrees ccw (-90) = 270 degrees cw (270)
"""

import numpy as np
from SPI2py.analysis import rotate


positions = np.array([[2., 2., 0.], [4., 2., 0.], [4., 0., 0.]])


def test_rotate_x_90():

    rotation_x_90 = np.array([np.pi / 2., 0., 0.])

    # Round to avoid 0.9999 != 1, but keep at least one decimal point to avoid njit errors
    # Njit requires floats not integers!
    new_positions = np.round(rotate(positions, rotation_x_90), 0)

    # See accompanying PPT file for test case derivations
    expected = np.array([[2., 2., 0.], [4., 2., 0.], [4., 2., -2.]])

    assert np.array_equal(new_positions, expected)


# def test_rotate_y_cc():
#     new_angle_y_90 = np.array([0., np.pi / 2, 0.])
#     delta_rotation = new_angle_y_90 - current_angle
#     new_positions = np.round(rotate(positions, delta_rotation), 0)
#
#     expected = np.array([[2., 2., 0.], [2., 2., -2.], [2., 0., -2.]])
#
#     assert np.array_equal(new_positions, expected)
#
#
# def test_rotate_z_cc():
#     new_angle_z_90 = np.array([0., 0., np.pi / 2])
#     delta_rotation = new_angle_z_90 - current_angle
#     new_positions = np.round(rotate(positions, delta_rotation), 0)
#
#     expected = np.array([[2., 2., 0.], [2., 4., 0.], [4., 4., 0.]])
#
#     assert np.array_equal(new_positions, expected)


def test_rotation_at_origin_zero():

    # Define a unit cube
    component_positions = np.array([[0, 0, 0],  # a
                                    [1, 0, 0],  # b
                                    [1, 1, 0],  # c
                                    [0, 1, 0],  # d

                                    [0, 0, 1],  # e
                                    [1, 0, 1],  # f
                                    [1, 1, 1],  # g
                                    [0, 1, 1]])  # h


    # Rotate about the origin by 0 degrees
    rotate_zero = np.array([0, 0, 0])

    # Rotate the cube

    rotated_component_positions = rotate(component_positions, rotate_zero)

    expected_rotated_component_positions_zero = np.array([[0, 0, 0],  # a
                                                          [1, 0, 0],  # b
                                                          [1, 1, 0],  # c
                                                          [0, 1, 0],  # d

                                                          [0, 0, 1],  # e
                                                          [1, 0, 1],  # f
                                                          [1, 1, 1],  # g
                                                          [0, 1, 1]])  # h

    assert np.allclose(rotated_component_positions, expected_rotated_component_positions_zero)
