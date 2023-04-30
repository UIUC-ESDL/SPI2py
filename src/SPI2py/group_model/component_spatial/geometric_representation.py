"""Module...
...
TODO Add functionality to rotate geometric primitives
"""

import numpy as np
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
















    p = pv.Plotter()

    super_ellipsoid = pv.ParametricSuperEllipsoid(xradius=ax, yradius=ay, zradius=az, n1=n1, n2=n2, center=[0, 0, 0])
    p.add_mesh(super_ellipsoid, color="tan", opacity=1)

    sphere = pv.Sphere(radius=0.5, center=[3, 0, 0])
    p.add_mesh(sphere, color="red", opacity=1)

    p.show()





