"""Module...
...
"""

import numpy as np


def generate_rectangular_prism(origin, dimension, diameter):
    """

    :param dimension:
    :param diameter:
    :param origin:
    :return:
    """

    radius = diameter / 2

    len_x, len_y, len_z = dimension

    num_x_spheres = int(len_x // diameter)
    num_y_spheres = int(len_y // diameter)
    num_z_spheres = int(len_z // diameter)

    # Positions
    pos_x = origin[0] + radius
    pos_y = origin[1] + radius
    pos_z = origin[2] + radius

    x = np.linspace(pos_x, pos_x + len_x, num_x_spheres)
    y = np.linspace(pos_y, pos_y + len_y, num_y_spheres)
    z = np.linspace(pos_z, pos_z + len_z, num_z_spheres)

    xx, yy, zz = np.meshgrid(x, y, z)

    positions = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1), zz.reshape(-1,1)))

    sphere_count = positions.shape[0]
    radii = np.repeat(radius, sphere_count)

    return positions, radii


def generate_rectangular_prisms(origins, dimensions, diameters):

    positions, radii = np.empty((0,3)), np.empty(0)

    for origin, dimension, diameter in zip(origins, dimensions, diameters):
        sphere_positions, sphere_radius = generate_rectangular_prism(origin, dimension,diameter)
        positions = np.vstack((positions, sphere_positions))
        radii = np.append(radii, sphere_radius)

    return positions, radii