"""Module...
...
"""

import pyvista as pv


def plot_2d():
    pass


def plot_3d(positions, radii, colors):
    """

    TODO Overlay the plot with the original image

    """

    p = pv.Plotter(window_size=[1000, 1000])

    for positions, radii, color in zip(positions, radii, colors):

        for position, radius in zip(positions, radii):

            sphere = pv.Sphere(radius=radius, center=list(position))
            p.add_mesh(sphere, color=color)

    p.view_vector((5.0, 2, 3))
    p.add_floor('-z', lighting=True, color='tan', pad=1.0)
    p.enable_shadows()
    p.show_axes()
    p.show()


