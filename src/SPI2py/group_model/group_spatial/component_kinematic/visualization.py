"""Module...
...
"""

import pyvista as pv


def plot_3d(objects, colors):
    """

    TODO Overlay the plot with the original image

    """

    p = pv.Plotter(window_size=[1000, 1000])

    for obj, color in zip(objects, colors):
        p.add_mesh(obj, color=color)

    p.view_vector((5.0, 2, 3))
    p.add_floor('-z', lighting=True, color='tan', pad=1.0)
    p.enable_shadows()
    p.show_axes()
    p.show()


