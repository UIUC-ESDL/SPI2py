"""Pseudo MDBD

This script takes a CAD model (as an STL file) and generates a (pseudo) MDBD representation of it.

Please note that this script is not polished and contains many inefficiencies (both coding-wise and
optimization-theory-wise). It is meant to be a proof-of-concept and a starting point for further development.

Also note that the Trimesh library may raise some errors when you first run the script. When I pip installed trimesh,
it seems to have missed a few necessary libraries. Read the error message and see if there is an import error. If so,
pip install the missing library.

I had to pip install 'rtree'

The algorithm is as follows:

    1. Create a meshgrid of points, using the bounding box of the object as the bounds.
    2. Filter out points that are outside the object as these are not applicable.
    3. Find the point furthest from the surface. Use this point and distance as an initial sphere (position and radius).
    4. Use an optimizer to further tweak the position to maximize the radius (constraint: stay within object).
    5. Repeat the previous steps, however track the spheres that have been created and use them as additional
    non-overlap constraints. Also remove meshgrid points within the spheres, and recalulate the furthest point from the
    surface AND spheres.

"""

import numpy as np
import pyvista as pv
import trimesh
from scipy.optimize import minimize, NonlinearConstraint, Bounds


def objective(d_i):
    """Maximize radius of the sphere by minimizing the negative radius"""

    _, _, _, ri = d_i

    return -ri


def constraint_stay_within_object(d_i):
    """The distance from the sphere to the surface must be greater than the radius"""

    xi, yi, zi, ri = d_i

    point = np.array([[xi, yi, zi]])

    min_distance = trimesh.proximity.signed_distance(mesh_trimesh, point)

    return ri - min_distance

def constraint_nonoverlap(d_i):
    """The distance from the sphere to other spheres must be greater than the sum of the radii"""

    xi, yi, zi, ri = d_i

    point_i = np.array([[xi, yi, zi]])
    radius = np.array([ri])

    distances = np.linalg.norm(sphere_points - point_i, axis=1)

    # Use -1 to flip the convention (a negative gap means the spheres are overlapping, which violates the constraint
    # but the optimizer follows negative-null form so a negative constraint value means the constraint is satisfied).
    gaps = -1 * (distances - (sphere_radii.reshape(-1) + radius))

    return gaps


# USER INPUT: Filepath
filepath = '../../../../../examples/scratch/files/part2.stl'

# Create the pyvista and trimesh objects. Both are required.
mesh_pyvista = pv.read(filepath)

# If the object has a ridiculously high fidelity then the naive algorthm will run too slow
# The decimate method (commented out below) reduces the resolution of the mesh
# model_engine = pv.read('C:/Users/cpgui/PycharmProjects/SPI2py/examples/prototyping/files/Motor_final_solid_reducedv4.stl')
# model_engine_decimated = model_engine.decimate_boundary(target_reduction=0.99)
# model_engine_decimated.plot()
# model_engine_decimated.save('C:/Users/cpgui/PycharmProjects/SPI2py/examples/prototyping/files/model_engine_decimated.stl')


mesh_trimesh = trimesh.exchange.load.load(filepath)


# Define variable bounds based on the object's bounding box
x_min = mesh_trimesh.vertices[:, 0].min()
x_max = mesh_trimesh.vertices[:, 0].max()
y_min = mesh_trimesh.vertices[:, 1].min()
y_max = mesh_trimesh.vertices[:, 1].max()
z_min = mesh_trimesh.vertices[:, 2].min()
z_max = mesh_trimesh.vertices[:, 2].max()
r_min = 0.01  # Lower bound to prevent the nonphysical radius of 0
r_max = min([x_max - x_min, y_max - y_min, z_max - z_min]) / 2  # The max radius must still fit inside the bounding box

# USER INPUT: The number of increments for each dimension of the meshgrid.
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

# Initialize the empty arrays that will store the MDBD spheres
sphere_points = np.empty((0, 3))
sphere_radii = np.empty((0, 1))


# PACKAGE THE FIRST SPHERE (No sphere-sphere overlap constraint)


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


# Optimize the sphere position and radius
bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
nlc = NonlinearConstraint(constraint_stay_within_object, -np.inf, 0)

x_0, y_0, z_0 = max_min_point
r_0 = 0.5 * max_min_distance[0] # Use half the distance as the initial radius

d_0 = np.array([x_0, y_0, z_0, r_0])

res = minimize(objective, d_0,
               method='trust-constr',
               constraints=nlc,
               bounds=bounds,
               tol=1e-1)

# Add the sphere to the list of spheres
sphere_points = np.vstack((sphere_points, res.x[:3]))
sphere_radii = np.vstack((sphere_radii, res.x[3]))

# Remove any points that are within the max-min sphere
points_filtered = points_filtered[np.linalg.norm(points_filtered - res.x[:3], axis=1) > res.x[3]]


# FURTHER SPHERES

# USER INPUT: How many additional MDBD spheres to package?
num_spheres = 20

for i in range(num_spheres):

    # Find the point furthest from the surface and existing spheres
    min_distances = []
    for point in points_filtered:
        point_i = point.reshape(1, 3)

        # Check the minimum distance to a surface on the mesh
        min_distance_surface = trimesh.proximity.signed_distance(mesh_trimesh, point_i)

        # Check the minimum distance to other spheres
        min_distance_spheres = np.linalg.norm(sphere_points - point_i, axis=1) - sphere_radii.reshape(-1)
        min_distance_sphere = np.array([np.min(min_distance_spheres)])

        min_distance = np.min([min_distance_surface, min_distance_sphere])

        min_distances.append(min_distance)

    max_min_distance = max(min_distances)
    max_min_point = points_filtered[np.argmax(min_distances)]

    # Optimize the sphere position and radius
    bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
    nlc = NonlinearConstraint(constraint_stay_within_object, -np.inf, 0)
    nlc2 = NonlinearConstraint(constraint_nonoverlap, -np.inf, 0)

    x_0, y_0, z_0 = max_min_point
    r_0 = 0.5 * max_min_distance

    d_0 = np.array([x_0, y_0, z_0, r_0])

    res = minimize(objective, d_0,
                   method='trust-constr',
                   constraints=[nlc, nlc2],
                   bounds=bounds,
                   tol=1e-1)

    sphere_points = np.vstack((sphere_points, res.x[:3]))
    sphere_radii = np.vstack((sphere_radii, res.x[3]))

    # Remove any points that are within the max-min sphere
    points_filtered = points_filtered[np.linalg.norm(points_filtered - res.x[:3], axis=1) > res.x[3]]



# Plot object with PyVista
plotter = pv.Plotter()

part2 = pv.read(filepath)
plotter.add_mesh(part2, color='white', opacity=0.5)

# Plot points
# points = pv.PolyData(points_filtered)
# plotter.add_mesh(points, color='red', point_size=10, render_points_as_spheres=True)


# Plot the sphere
for i in range(len(sphere_points)):
    sphere = pv.Sphere(center=sphere_points[i], radius=sphere_radii[i])
    plotter.add_mesh(sphere, color='green', opacity=0.75)
plotter.show()


# OUTPUT

# Combine the points and radii into a single array
spheres = np.hstack((sphere_points, sphere_radii))

# Write the spheres to a text file
np.savetxt('../../../../../examples/scratch/spheres.txt', spheres, delimiter=' ')
