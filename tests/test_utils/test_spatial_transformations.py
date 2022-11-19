"""

Here is a helpful checker for test cases: https://www.emathhelp.net/calculators/algebra-2/rotation-calculator/

Use round to avoid small numerical differences
"""

import numpy as np
from src.SPI2Py.utils.spatial_transformations import rotate
from scipy.spatial.transform import Rotation as R

positions = np.array([[2., 2., 0.], [4., 2., 0.], [4, 0, 0]])


def test_dummy():
    assert 1== 1

def test_rotate_x_cc():

    rotation_x_90 = np.array([np.pi / 2, 0., 0.])

    new_positions = np.round(rotate(positions, rotation_x_90), 0)

    r = R.from_euler('zyx',[0,0,90], degrees=True)
    new_pos = r.apply(positions)

    expected = np.array([[2., 2., 0.], [4., 2., 0.], [4., 2., -2.]])

    assert np.array_equal(new_pos, expected)


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
