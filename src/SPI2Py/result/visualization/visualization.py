"""Module...
...
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import imageio.v2 as imageio
import tempfile


def plot_sphere(position, radius, color, ax, resolution):
    # The effective resolution of the sphere

    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)

    x = position[0] + radius * (np.outer(np.cos(u), np.sin(v)))
    y = position[1] + radius * (np.outer(np.sin(u), np.sin(v)))
    z = position[2] + radius * (np.outer(np.ones(np.size(u)), np.cos(v)))

    ax.plot_surface(x, y, z, linewidth=0.0, color=color)


def plot(plot_array, savefig, directory, config):
    """
    Plots classes...

    [[positions_1, radii_1, color_1],[],... ]

    :param plot_array:
    :param savefig:
    :param directory:
    :return:
    """

    # Unpack plotting config settings
    xlim = config['results']['plot']['x limits']
    ylim = config['results']['plot']['y limits']
    zlim = config['results']['plot']['z limits']
    resolution = config['results']['plot']['resolution']


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for positions, radii, color in plot_array:

        for position, radius in zip(positions, radii):
            plot_sphere(position, radius, color, ax, resolution)

    # Format Plot
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)


    plt.show()

    # Save Figure for GIF generation
    if savefig is True:
        fig.savefig(directory)


def generate_gif(layout, design_vector_log, frames_per_figure, gif_directory, gif_filename):
    """

    :param layout:
    :param design_vector_log:
    :param frames_per_figure:
    :param GIFfilepath:
    :return:
    """
    def plot_all(design_vectors):
        """

        :param design_vectors:
        :return:
        """
        i = 1
        for xk in design_vectors:
            filepath = tempDir+'/'+str(i)
            pos_dict = layout.calculate_positions(xk)
            layout.set_positions(pos_dict)

            layout.plot_layout(savefig=True, directory=filepath)

            i += 1

    # Create a temporary directory to save to plotted figures
    temp = tempfile.TemporaryDirectory()
    tempDir = temp.name

    plot_all(design_vector_log)

    files = os.listdir(tempDir)

    # Sort files based on numerical order (i.e., 1,2,... 11,12 not 1,11,12,2,...)
    order = [int(file[0:-4]) for file in files]
    files = [file for _, file in sorted(zip(order, files))]

    filenames = [tempDir + '/' + filename for filename in files]

    # Generate the GIF
    images = []
    for filename in filenames:
        for _ in range(frames_per_figure):
            images.append(imageio.imread(filename))

    # TODO Add config file input for GIF name
    output_filepath = gif_directory + gif_filename

    imageio.mimsave(output_filepath, images)

