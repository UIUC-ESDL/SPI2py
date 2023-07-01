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


def objective(d_i):
    """Maximize radius of the sphere by minimizing the negative radius"""

    _, _, _, ri = d_i

    return -ri


def constraint(d_i):
    """The distance from the sphere to the surface must be greater than the radius"""

    xi, yi, zi, ri = d_i

    point = np.array([[xi, yi, zi]])

    min_distance = trimesh.proximity.signed_distance(mesh, point)

    return ri - min_distance

# Load file

filepath = 'C:/Users/cpgui/PycharmProjects/SPI2py/examples/prototyping/files/part2.stl'
part = pv.read(filepath)

# mesh = trimesh.exchange.load.load(filepath)


# Define variable bounds

x_min = part.points[:, 0].min()
x_max = part.points[:, 0].max()
y_min = part.points[:, 1].min()
y_max = part.points[:, 1].max()
z_min = part.points[:, 2].min()
z_max = part.points[:, 2].max()
r_min = 0.01
r_max = min([x_max - x_min, y_max - y_min, z_max - z_min]) / 2

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



# for i in range(1):


# Filter out points that are not inside the mesh

# signed_distances = trimesh.proximity.signed_distance(mesh, points)
points_filtered = np.array([point for point in points if part.find_containing_cell(point) != -1])

# Find the point with the greatest minimum distance to the surface
min_distances = []
for point in points_filtered:

    point_i = point.reshape(1, 3)

    min_distance = trimesh.proximity.signed_distance(mesh, point_i)

    min_distances.append(min_distance)

max_min_distance = max(min_distances)

max_min_point = points_filtered[np.argmax(min_distances)]

# _, min_distance_point = part.find_closest_cell(point_i, return_closest_point=True)
#
# min_distance = np.linalg.norm(point - min_distance_point, axis=1)
#
# min_distances.append(min_distance)


#
#
#
# bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
# nlc = NonlinearConstraint(constraint, -np.inf, 2)
#
# x_0, y_0, z_0 = max_min_point
# r_0 = 0.5 * max_min_distance[0]
#
# d_0 = np.array([x_0, y_0, z_0, r_0])
#
# res = minimize(objective, d_0,
#                method='trust-constr',
#                constraints=nlc,
#                bounds=bounds,
#                tol=1e-3)
#







# Plot object with PyVista
plotter = pv.Plotter()


plotter.add_mesh(part, color='white', opacity=0.5)

# Plot points
points_filtered = pv.PolyData(points_filtered)
plotter.add_mesh(points_filtered, color='red', point_size=10, render_points_as_spheres=True)

# # Plot the max min point
# max_min_point = pv.PolyData(max_min_point)
# plotter.add_mesh(max_min_point, color='blue', point_size=20, render_points_as_spheres=True)

# Plot the sphere
# sphere = pv.Sphere(center=res.x[:3], radius=res.x[3])
# plotter.add_mesh(sphere, color='green', opacity=0.75)

plotter.show()




# Create a sphere mesh in trimesh
# sphere = trimesh.creation.uv_sphere(radius=res.x[3], transform=trimesh.transformations.translation_matrix(res.x[:3]))

# Convert the sphere primitive into a mesh


# Remove the sphere from the mesh
# newmesh = trimesh.boolean.difference([mesh, sphere])

# sphere = pv.Sphere(center=res.x[:3], radius=res.x[3])
# # # sphere = pv.Sphere(radius=50, center=(15, 15, 15))
#
# part3 = part2.boolean_difference(sphere)
# part3.plot(color='tan')

# sphere = pv.Sphere(center=res.x[:3], radius=res.x[3]).triangulate().subdivide(3)
#
# result = part2.boolean_difference(sphere)
# result.plot


# plotter = pv.Plotter()

# Create trimesh for sphere
# sphere = trimesh.primitives.Sphere(center=res.x[:3], radius=res.x[3], subdivisions=4)
# sphere = trimesh.creation.uv_sphere(radius=res.x[3], transform=trimesh.transformations.translation_matrix(res.x[:3]))

# Remove the sphere from the mesh
# result = trimesh.boolean.difference(mesh, sphere)

# # Plot the mesh
# plotter.add_mesh(mesh, color='white', opacity=0.5)

# plotter.show()








# part3_trimesh = trimesh.Trimesh(part3.points, faces=part3.faces)


# Filter out points that are not inside the mesh
# signed_distances = trimesh.proximity.signed_distance(part3_trimesh, points)
# points_filtered = points[signed_distances > 0]

# # Find the point with the greatest minimum distance to the surface
# min_distances = []
# for point in points_filtered:
#     point_i = point.reshape(1, 3)
#     min_distance = trimesh.proximity.signed_distance(part3_trimesh, point_i)
#     min_distances.append(min_distance)
#
# max_min_distance = max(min_distances)
#
# max_min_point = points_filtered[np.argmax(min_distances)]


# # Plot object with PyVista
# plotter = pv.Plotter()
#
# # part2 = pv.read(filepath)
# plotter.add_mesh(part3, color='white', opacity=0.5)
#
# # Plot points
# points_filtered = pv.PolyData(points_filtered)
# plotter.add_mesh(points_filtered, color='red', point_size=10, render_points_as_spheres=True)
#
# # Plot the max min point
# max_min_point = pv.PolyData(max_min_point)
# plotter.add_mesh(max_min_point, color='blue', point_size=20, render_points_as_spheres=True)
#
# # Plot the sphere
# sphere = pv.Sphere(center=res.x[:3], radius=res.x[3])
# plotter.add_mesh(sphere, color='green', opacity=0.75)
#
# plotter.show()