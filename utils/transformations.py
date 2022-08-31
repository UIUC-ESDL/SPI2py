"""

"""

import numpy as np
from scipy.spatial.transform import Rotation
from numba import njit


def translate(current_reference_point, new_reference_point):
    """
    Takes the change in the object's reference point and returns the new position of each sphere
    :param current_position:
    :param new_position:
    :return:
    """
    x_current, y_current, z_current


def rotate(positions, angles, reverse_direction=False):
    """
    ...
    change for design vectors...?

    how make reversible for FD...
    https://math.stackexchange.com/questions/838885/reverse-of-a-rotation-matrix-for-superposition-colon-a-on-b-to-b-on-a


    :return:
    """

    r = Rotation.from_euler('xyz', [angles[0], angles[1], angles[2]], degrees=True)

    # If reverse then invert rotation matrix...

    positions_rotated = r.apply(positions)

    return positions_rotated