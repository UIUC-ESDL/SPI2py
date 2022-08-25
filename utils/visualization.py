"""Module...
...
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_sphere(position, radius, color, ax):

    # The effective resolution of the sphere
    num_points = 12

    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)

    x = position[0] + radius * (np.outer(np.cos(u), np.sin(v)))
    y = position[1] + radius * (np.outer(np.sin(u), np.sin(v)))
    z = position[2] + radius * (np.outer(np.ones(np.size(u)), np.cos(v)))

    ax.plot_surface(x, y, z, linewidth=0.0, color=color)


def plot(plot_dict):

    """
    Plots objects...

    {
    object1:    {
                spheres:
                radii:
                color
                }
    object2: ...
    }

    :param plot_dict:
    :return:
    """

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    objects = list(plot_dict.keys())
    for obj in objects:
        positions = plot_dict[obj]['positions']
        radii = plot_dict[obj]['radius']
        color = plot_dict[obj]['color']

        for position, radius in zip(positions, radii):
            plot_sphere(position, radius, color, ax)





    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_zlim(0, 2)
    plt.show()