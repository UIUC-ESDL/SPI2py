"""

Ideas
-----
1. Seed Many, many, points.
2. Filter out ones that are not inside the object.
3. Filter out ones that are inside existing spheres.
4. Calculate the minimum distance from each remaining point to each surface
5. Calculate the maximum minimum distance --> use this as the starting point and radius (minus a small tolerance)
6. Remove that sphere from the mesh, repeat for maxiter or a minimum maximum distance toelrance

Cannot choose a point on the surface

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

    min_distance = trimesh.proximity.signed_distance(mesh_trimesh, point)

    return ri - min_distance

def constraint_2(d_i):
    """The distance from the sphere to other spheres must be greater than the sum of the radii"""

    xi, yi, zi, ri = d_i

    point = np.array([[xi, yi, zi]])
    radius = np.array([ri])

    distances = np.linalg.norm(sphere_points - point, axis=1)
    gaps = -1 * ( distances - (sphere_radii.reshape(-1) + radius) )

    return gaps

# Load file

filepath = 'C:/Users/cpgui/PycharmProjects/SPI2py/examples/prototyping/files/part2.stl'

mesh_pyvista = pv.read(filepath)
mesh_trimesh = trimesh.exchange.load.load(filepath)

mesh_pv = pv.wrap(mesh_trimesh)


# Define variable bounds

x_min = mesh_trimesh.vertices[:, 0].min()
x_max = mesh_trimesh.vertices[:, 0].max()
y_min = mesh_trimesh.vertices[:, 1].min()
y_max = mesh_trimesh.vertices[:, 1].max()
z_min = mesh_trimesh.vertices[:, 2].min()
z_max = mesh_trimesh.vertices[:, 2].max()
r_min = 0.01
r_max = min([x_max - x_min, y_max - y_min, z_max - z_min]) / 2

# Create an initial population of sample points
nx = 15
ny = 15
nz = 15

# Create a 3d meshgrid
x = np.linspace(x_min, x_max, nx)
y = np.linspace(y_min, y_max, ny)
z = np.linspace(z_min, z_max, nz)
xx, yy, zz = np.meshgrid(x, y, z)

# Flatten the meshgrids and conert them to a list of points
points_a = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T


sphere_points = np.empty((0, 3))
sphere_radii = np.empty((0, 1))


# Filter out points that are not inside the mesh
signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, points_a)
points_filtered = points_a[signed_distances > 0]

# Find the point with the greatest minimum distance to the surface
min_distances = []
for point in points_filtered:
    point_i = point.reshape(1, 3)
    min_distance = trimesh.proximity.signed_distance(mesh_trimesh, point_i)
    min_distances.append(min_distance)

max_min_distance = max(min_distances)
max_min_point = points_filtered[np.argmax(min_distances)]




bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
nlc = NonlinearConstraint(constraint, -np.inf, 0)

x_0, y_0, z_0 = max_min_point
r_0 = 0.5 * max_min_distance[0]

d_0 = np.array([x_0, y_0, z_0, r_0])

res = minimize(objective, d_0,
               method='trust-constr',
               constraints=nlc,
               bounds=bounds,
               tol=1e-3)

sphere_points = np.vstack((sphere_points, res.x[:3]))
sphere_radii = np.vstack((sphere_radii, res.x[3]))

# Remove any points that are within the max-min sphere
points_filtered = points_filtered[np.linalg.norm(points_filtered - res.x[:3], axis=1) > res.x[3]]

min_distances = []
for point in points_filtered:
    point_i = point.reshape(1, 3)
    min_distance = trimesh.proximity.signed_distance(mesh_trimesh, point_i)
    min_distances.append(min_distance)

max_min_distance = max(min_distances)
max_min_point = points_filtered[np.argmax(min_distances)]



bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
nlc = NonlinearConstraint(constraint, -np.inf, 0)
nlc2 = NonlinearConstraint(constraint_2, -np.inf, 0)

x_0, y_0, z_0 = max_min_point
r_0 = 0.5 * max_min_distance[0]

d_0 = np.array([x_0, y_0, z_0, r_0])

res2 = minimize(objective, d_0,
               method='trust-constr',
               constraints=[nlc, nlc2],
               bounds=bounds,
               tol=1e-3)

sphere_points = np.vstack((sphere_points, res2.x[:3]))
sphere_radii = np.vstack((sphere_radii, res2.x[3]))



points_filtered = points_filtered[np.linalg.norm(points_filtered - res2.x[:3], axis=1) > res2.x[3]]

min_distances = []
for point in points_filtered:
    point_i = point.reshape(1, 3)
    min_distance = trimesh.proximity.signed_distance(mesh_trimesh, point_i)
    min_distances.append(min_distance)

max_min_distance = max(min_distances)
max_min_point = points_filtered[np.argmax(min_distances)]


bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
nlc = NonlinearConstraint(constraint, -np.inf, 0)
nlc2 = NonlinearConstraint(constraint_2, -np.inf, 0)

x_0, y_0, z_0 = max_min_point
r_0 = 0.5 * max_min_distance[0]

d_0 = np.array([x_0, y_0, z_0, r_0])

res3 = minimize(objective, d_0,
               method='trust-constr',
               constraints=[nlc, nlc2],
               bounds=bounds,
               tol=1e-3)



# sphere_points = np.vstack((sphere_points, max_min_point_1))
# sphere_radii = np.vstack((sphere_radii, max_min_distance_1))
#
#
# # Remove any points that are within the max-min sphere
# points_filtered = points_filtered[np.linalg.norm(points_filtered - max_min_point_1, axis=1) > max_min_distance_1]
#
#
#
# min_distances = []
# for point in points_filtered:
#     point_i = point.reshape(1, 3)
#     min_distance = trimesh.proximity.signed_distance(mesh_trimesh, point_i)
#
#     # Ignore points who min_distance will overlap with existing spheres
#     if np.all(np.linalg.norm(sphere_points - point_i, axis=1) > (sphere_radii + min_distance)):
#         min_distances.append(min_distance)
#
#
# max_min_distance_2 = max(min_distances)
# max_min_point_2 = points_filtered[np.argmax(min_distances)]
#
#
#
#
# # Filter out points that are not inside the mesh
# points_filtered = points_filtered[np.linalg.norm(points_filtered - max_min_point_2, axis=1) > max_min_distance_2]
#
#

# min_distances = []
# for point in points_filtered:
#     point_i = point.reshape(1, 3)
#     min_distance = trimesh.proximity.signed_distance(mesh_trimesh, point_i)
#     min_distances.append(min_distance)
#
# max_min_distance_3 = max(min_distances)
# max_min_point_3 = points_filtered[np.argmax(min_distances)]


# Plot object with PyVista
plotter = pv.Plotter()

part2 = pv.read(filepath)
plotter.add_mesh(part2, color='white', opacity=0.5)

# Plot points
points = pv.PolyData(points_filtered)
plotter.add_mesh(points, color='red', point_size=10, render_points_as_spheres=True)

# Plot the max min point
# max_min_point = pv.PolyData(max_min_point_1)
# plotter.add_mesh(max_min_point, color='blue', point_size=20, render_points_as_spheres=True)



# Plot the sphere

# Plot the sphere
sphere = pv.Sphere(center=res.x[:3], radius=res.x[3])
plotter.add_mesh(sphere, color='green', opacity=0.75)

sphere = pv.Sphere(center=res2.x[:3], radius=res2.x[3])
plotter.add_mesh(sphere, color='purple', opacity=0.75)

sphere = pv.Sphere(center=res3.x[:3], radius=res3.x[3])
plotter.add_mesh(sphere, color='purple', opacity=0.75)

# sphere = pv.Sphere(center=res.x[:3], radius=res.x[3])
# sphere = pv.Sphere(center=max_min_point_1, radius=max_min_distance_1)
# plotter.add_mesh(sphere, color='green', opacity=0.75)
#
# sphere = pv.Sphere(center=max_min_point_2, radius=max_min_distance_2)
# plotter.add_mesh(sphere, color='purple', opacity=0.75)

# sphere = pv.Sphere(center=max_min_point_3, radius=max_min_distance_3)
# plotter.add_mesh(sphere, color='teal', opacity=0.75)

plotter.show()






# sphere = pv.Sphere(center=res.x[:3], radius=res.x[3])
# # sphere = pv.Sphere(radius=50, center=(15, 15, 15))
# # result = cube.boolean_difference(sphere)
# result = part2.boolean_difference(sphere)
# result.plot(color='tan')

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
#
# plotter.show()