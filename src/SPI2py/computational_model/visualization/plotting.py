"""Module...
...
"""

import numpy as np
import pyvista as pv


def plot_2d():
    pass


def plot_3d(types, positions, radii, colors):
    """

    TODO Overlay the plot with the original image

    """

    p = pv.Plotter(window_size=[1000, 1000])

    # TODO Fix name plurality
    for obj_type, obj_positions, obj_radii, color in zip(types, positions, radii, colors):

        if obj_type == 'component':

            for position, radius in zip(obj_positions, obj_radii):

                sphere = pv.Sphere(radius=radius, center=list(position))
                p.add_mesh(sphere, color=color)
        elif obj_type == 'interconnect':

            direction = obj_positions[-1] - obj_positions[0]
            height = np.linalg.norm(direction)
            center = obj_positions[0] + direction / 2

            sphere1 = pv.Sphere(radius=obj_radii[0], center=list(obj_positions[0]))
            p.add_mesh(sphere1, color=color)

            cylinder = pv.Cylinder(radius=obj_radii[0], height=height, center=list(center), direction=list(direction))
            p.add_mesh(cylinder, color=color)

            sphere2 = pv.Sphere(radius=obj_radii[0], center=list(obj_positions[-1]))
            p.add_mesh(sphere2, color=color)

        else:
            raise NotImplementedError

    p.view_vector((5.0, 2, 3))
    p.add_floor('-z', lighting=True, color='tan', pad=1.0)
    p.enable_shadows()
    p.show_axes()
    p.show()


