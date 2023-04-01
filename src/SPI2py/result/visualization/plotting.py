"""Module...
...
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv


def plot_sphere(position, radius, color, ax, resolution):
    # The effective resolution of the sphere

    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = position[0] + radius * (np.outer(np.cos(u), np.sin(v)))
    y = position[1] + radius * (np.outer(np.sin(u), np.sin(v)))
    z = position[2] + radius * (np.outer(np.ones(np.size(u)), np.cos(v)))

    ax.plot_surface(x, y, z, linewidth=0.0, color=color)


def plot_objects(plot_array, savefig, directory, config):
    """
    Plots classes...

    [[positions_1, radii_1, color_1],[],... ]

    :param plot_array:
    :param savefig:
    :param directory:
    :return:
    """

    # Unpack plotting config settings
    # xlim = config['results']['plot']['x limits']
    # ylim = config['results']['plot']['y limits']
    # zlim = config['results']['plot']['z limits']
    # resolution = config['results']['plot']['resolution']

    p = pv.Plotter(window_size=[1000, 1000])

    for positions, radii, color in plot_array:

        for position, radius in zip(positions, radii):

            sphere = pv.Sphere(radius=radius, center=list(position))
            p.add_mesh(sphere, color=color)


    p.show()


