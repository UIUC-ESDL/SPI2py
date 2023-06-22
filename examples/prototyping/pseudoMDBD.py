"""

Ideas
-----
1. Seed Many, many, points.
2. Filter out ones that are not inside the object.
3. Filter out ones that are inside existing spheres.
4. Calculate the minimum distance from each remaining point to each surface
5. Calculate the maximum minimum distance --> use this as the starting point and radius (minus a small tolerance)
6. Remove that sphere from the mesh, repeat for maxiter or a minimum maximum distance toelrance

"""

import numpy as np
import pyvista as pv
import trimesh
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import matplotlib.pyplot as plt


# Load file

filepath = 'C:/Users/cpgui/PycharmProjects/SPI2py/examples/prototyping/files/part2.stl'

# Load the file
# mesh = pv.read(filepath)
# cpos = mesh.plot()
# grid = pv.read(filepath)
#
# centers = grid.cell_centers()
#
# pl = pv.Plotter()
# pl.add_mesh(grid.extract_all_edges(), color="k", line_width=1)
# pl.add_mesh(centers, color="r", point_size=8.0, render_points_as_spheres=True)
# pl.show()

mesh = trimesh.exchange.load.load(filepath)

# mesh.show()
# See trimesh.proximity.max_tangent_sphere



p1 = np.array([[5, 5, 5]])

min_dist = trimesh.proximity.signed_distance(mesh, p1)



# Define variable bounds

x_min = mesh.vertices[:, 0].min()
x_max = mesh.vertices[:, 0].max()
y_min = mesh.vertices[:, 1].min()
y_max = mesh.vertices[:, 1].max()
z_min = mesh.vertices[:, 2].min()
z_max = mesh.vertices[:, 2].max()
r_min = 0.01
r_max = min([x_max-x_min, y_max-y_min, z_max-z_min])/2


# Create an initial population of sample points
nx = 12
ny = 12
nz = 12

# Create a 3d meshgrid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
z = np.linspace(z_min, z_max, nz)
xx, yy, zz = np.meshgrid(x, y, z)

# Flatten the meshgrids and conert them to a list of points
points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

# Filter out points that are not inside the mesh
signed_distances = trimesh.proximity.signed_distance(mesh, points)
points = points[signed_distances > 0]

# Find the point with the greatest minimum distance to the surface
min_distances = []
for point in points:
    point_i = point.reshape(1, 3)
    min_distance = trimesh.proximity.signed_distance(mesh, point_i)
    min_distances.append(min_distance)

max_min_distance = max(min_distances)

max_min_point = points[np.argmax(min_distances)]


#
# # Initialize empty arrays
# points = np.empty((0, 3))
# radii = np.empty(0)
#
#
# def objective(d_i):
#     """Maximize radius of the sphere by minimizing the negative radius"""
#
#     x, y, z, r = d_i
#
#     return -r
#
#
# def constraint_1(d_i):
#     """The signed distance of the sphere must be greater than the radius of the sphere.
#
#     Note: The sign of dist is flipped to make it a constraint since the trimesh convention is that the sign is positive
#     if the point is inside the mesh.
#     """
#
#     x, y, z, r = d_i
#
#     point = np.array([[x, y, z]])
#
#     dist = trimesh.proximity.signed_distance(mesh, point)
#
#
#     dist = -float(dist)
#
#     dist_min = dist - r
#
#     return dist_min
#
# def constraint_2(di):
#     """The sphere must not overlap with existing spheres"""
#
#     x, y, z, r = di
#
#     point = np.array([[x, y, z]])
#     radius = np.array([r])
#
#     if points.shape[0] == 0:
#         overlap = -10 # Arbitrary negative number
#
#     else:
#
#         gaps = []
#
#         for k in range(len(points)):
#             gap = np.linalg.norm(point - points[k]) - (radius + radii[k])
#             gaps.append(gap)
#
#         overlap = float(min(gaps))
#
#     return overlap
#
#
#
# bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
# nlc_1 = NonlinearConstraint(constraint_1, -np.inf, 0)
# nlc_2 = NonlinearConstraint(constraint_2, -np.inf, 0)
#
#
# for i in range(1):
#
#     d = []
#     f = []
#     for j in range(5):
#         d0 = np.random.rand(4)
#         res = minimize(objective, d0,
#                        method='trust-constr',
#                        constraints=nlc_1,
#                        bounds=bounds,
#                        tol=1e-1)
#         d.append(res.x)
#         f.append(res.fun)
#
#     d_opt = d[np.argmax(f)]
#
#     points = np.vstack((points, d_opt[:3]))
#     radii = np.append(radii, d_opt[3])
#     print('point', d_opt[:3])
#     print('radius', d_opt[3])
#
#
# Plot object with PyVista
plotter = pv.Plotter()

part2 = pv.read(filepath)
plotter.add_mesh(part2, color='white', opacity=0.5)


# Plot points
points = pv.PolyData(points)
plotter.add_mesh(points, color='red', point_size=10, render_points_as_spheres=True)

# Plot the max min point
max_min_point = pv.PolyData(max_min_point)
plotter.add_mesh(max_min_point, color='blue', point_size=20, render_points_as_spheres=True)

# for i in range(len(points)):
#     sphere = pv.Sphere(center=points[i], radius=radii[i])
#     plotter.add_mesh(sphere, color='red', opacity=0.75)
#
plotter.show()
