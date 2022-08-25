"""Module...
...
"""

import numpy as np


def generate_rectangular_prism(dimension, diameter, origin=None):
    """

    :param dimension:
    :param diameter:
    :param origin:
    :return:
    """
    if origin is None:
        origin_x, origin_y, origin_z = 0, 0, 0
    else:
        origin_x, origin_y, origin_z = origin

    radius = diameter / 2

    len_x, len_y, len_z = dimension

    num_x_spheres = int(len_x // diameter)
    num_y_spheres = int(len_y // diameter)
    num_z_spheres = int(len_z // diameter)

    # Positions
    pos_x = origin_x + radius
    pos_y = origin_y + radius
    pos_z = origin_z + radius

    x = np.linspace(pos_x, pos_x + len_x, num_x_spheres)
    y = np.linspace(pos_y, pos_y + len_y, num_y_spheres)
    z = np.linspace(pos_z, pos_z + len_z, num_z_spheres)

    xx, yy, zz = np.meshgrid(x, y, z)

    positions = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)))

    sphere_count = positions.shape[0]
    radii = np.repeat(radius, sphere_count)

    return positions, radii


def generate_rectangular_prisms(dimensions, diameters, origins= None):

    # Add an if statement for origins

    for dimension, diameter, origin in zip(dimensions, diameters, origins):
        pos, rad = generate_rectangular_prism(dimension,diameter,origin)

    return pos, rad