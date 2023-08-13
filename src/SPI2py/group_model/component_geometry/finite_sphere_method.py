"""Module...
...
TODO Add functionality to rotate geometric primitives
"""

import jax.numpy as np
# import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import pyvista as pv


class GeometricRepresentation:
    """
    Class to represent the geometry of a system

    TODO add set_origin method diff than calc_pos
    also make ports a sub dict / state of the component
    """
    def __init__(self):
        self.origin = None
        self.signed_distances = None



def read_mdbd_file(filepath):
    """
    Reads a text file where each line contains x, y, z, r separated by spaces.
    Returns an (n,3) numpy array for x,y,z positions and an (n,1) array for radii.

    TODO Reshape radii from 1D to 2D or vice versa
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # TODO allow user to specify number of spheres
    lines = lines[0:50]

    positions = []
    radii = []

    for line in lines:
        x, y, z, r = line.split()

        positions.append([float(x), float(y), float(z)])
        radii.append(float(r))

    # TODO Remove scale factor
    scale_factor = 1

    positions = np.array(positions)*scale_factor
    radii = np.array(radii)*scale_factor

    return positions, radii




def generate_rectangular_prism(origin, dimension):
    """
    Generates...

    This is a rough representation, it doesn't actually package spheres precisely...
    :param origin:
    :param dimension:
    :return:
    """

    # TODO REPLACE w/ ARRAY INPUT
    dimension = np.array(dimension)

    # Set the diameter of the packing sphere to the smallest dimension
    diameter = np.min(dimension)
    radius = diameter / 2

    origin_x, origin_y, origin_z = origin
    len_x, len_y, len_z = dimension

    num_x_spheres = int(len_x // diameter)
    num_y_spheres = int(len_y // diameter)
    num_z_spheres = int(len_z // diameter)

    # Add 2 since we want to fit the spheres inside the box, not along its corners
    num_x_nodes = num_x_spheres + 2
    num_y_nodes = num_y_spheres + 2
    num_z_nodes = num_z_spheres + 2

    # [1:-1] Removes the first and last nodes because...
    x = np.linspace(origin_x, origin_x + len_x, num_x_nodes)[1:-1]
    y = np.linspace(origin_y, origin_y + len_y, num_y_nodes)[1:-1]
    z = np.linspace(origin_z, origin_z + len_z, num_z_nodes)[1:-1]

    xx, yy, zz = np.meshgrid(x, y, z)

    positions = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)))

    sphere_count = positions.shape[0]
    radii = np.repeat(radius, sphere_count)

    return positions, radii


def generate_rectangular_prism2(origin, dimension):
    """
    Generates...

    This is a rough representation, it doesn't actually package spheres precisely...
    :param origin:
    :param dimension:
    :return:
    """

    # Recursion depth
    rdepth = 2

    # Set the diameter of the packing sphere to the smallest dimension
    diameter = np.min(dimension)
    radius = diameter / 2

    origin_x, origin_y, origin_z = origin
    len_x, len_y, len_z = dimension

    num_x_spheres = int(len_x // diameter)
    num_y_spheres = int(len_y // diameter)
    num_z_spheres = int(len_z // diameter)

    # Add 2 since we want to fit the spheres inside the box, not along its corners
    num_x_nodes = num_x_spheres + 2
    num_y_nodes = num_y_spheres + 2
    num_z_nodes = num_z_spheres + 2

    # [1:-1] Removes the first and last nodes because...
    x = np.linspace(origin_x, origin_x + len_x, num_x_nodes)[1:-1]
    y = np.linspace(origin_y, origin_y + len_y, num_y_nodes)[1:-1]
    z = np.linspace(origin_z, origin_z + len_z, num_z_nodes)[1:-1]

    xx, yy, zz = np.meshgrid(x, y, z)

    positions = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)))

    sphere_count = positions.shape[0]
    radii = np.repeat(radius, sphere_count)

    return positions, radii


def generate_rectangular_prisms(origins, dimensions):
    positions, radii = np.empty((0, 3)), np.empty(0)

    for origin, dimension in zip(origins, dimensions):
        sphere_positions, sphere_radius = generate_rectangular_prism(origin, dimension)
        positions = np.vstack((positions, sphere_positions))
        radii = np.append(radii, sphere_radius)

    return positions, radii


