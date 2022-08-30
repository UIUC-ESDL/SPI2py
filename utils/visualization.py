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
                positions:
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



import os
import imageio.v2 as imageio
import tempfile

def generate_gif(layout, design_vector_log, frames_per_figure, name):

    temp = tempfile.TemporaryDirectory()
    tempDir = temp.name

    def plot_all(design_vectors):
        i = 1
        for xk in design_vectors:
            new_positions_dict = layout.calculate_positions(xk)
            layout.update_positions(new_positions_dict)

            layout.plot_layout(savefig=True, temp_directory=tempDir)

            i += 1

    plot_all(design_vector_log)

    files = os.listdir(tempDir)

    # Sort files based on numerical order (i.e., 1,2,... 11,12 not 1,11,12,2,...)
    order = [int(file[0:-4]) for file in files]
    files = [file for _, file in sorted(zip(order, files))]

    filenames = [tempDir + '/' + filename for filename in files]

    with imageio.get_writer(name, mode='I') as writer:

        for filename in filenames:
            image = imageio.imread(filename)

            for _ in range(frames_per_figure):
                writer.append_data(image)
