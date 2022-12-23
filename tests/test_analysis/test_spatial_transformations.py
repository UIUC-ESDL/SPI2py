"""

Here is a helpful checker for test cases: https://www.emathhelp.net/calculators/algebra-2/rotation-calculator/

Use round to avoid small numerical differences

TODO write tests to ensure negative values are same e.g., 90 degrees ccw (-90) = 270 degrees cw (270)
"""

import numpy as np
from src.SPI2Py.analysis.transformations import rotate_about_point


positions = np.array([[2., 2., 0.], [4., 2., 0.], [4., 0., 0.]])


def test_rotate_x_90():

    rotation_x_90 = np.array([np.pi / 2., 0., 0.])

    # Round to avoid 0.9999 != 1, but keep at least one decimal point to avoid njit errors
    # Njit requires floats not integers!
    new_positions = np.round(rotate_about_point(positions, rotation_x_90), 0)

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
