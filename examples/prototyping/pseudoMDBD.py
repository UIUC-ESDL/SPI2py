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


# Initialize empty arrays
points = np.empty((0, 3))
radii = np.empty(0)


def objective(d_i):
    """Maximize radius of the sphere by minimizing the negative radius"""

    x, y, z, r = d_i

    return -r


def constraint_1(d_i):
    """Signed distance must be negative to indicate that the sphere is inside the mesh

    Note: The sign of dist is flipped to make it a constraint since the trimesh convention is that the sign is positive
    if the point is inside the mesh.
    """

    x, y, z, _ = d_i

    point = np.array([[x, y, z]])

    dist = trimesh.proximity.signed_distance(mesh, point)
    dist = -float(dist)

    return dist

def constraint_2(di):
    """The sphere must not overlap with existing spheres"""

    x, y, z, r = di

    point = np.array([[x, y, z]])
    radius = np.array([r])

    if points.shape[0] == 0:
        overlap = -10 # Arbitrary negative number

    else:

        gaps = []

        for k in range(len(points)):
            gap = np.linalg.norm(point - points[k]) - (radius + radii[k])
            gaps.append(gap)

        overlap = float(min(gaps))

    return overlap



bounds = Bounds([x_min, y_min, z_min, r_min], [x_max, y_max, z_max, r_max])
nlc_1 = NonlinearConstraint(constraint_1, -np.inf, 0)
nlc_2 = NonlinearConstraint(constraint_2, -np.inf, 0)


for i in range(1):

    d = []
    f = []
    for j in range(5):
        d0 = np.random.rand(4)
        res = minimize(objective, d0,
                       method='trust-constr',
                       constraints=nlc_1,
                       bounds=bounds,
                       tol=1e-1)
        d.append(res.x)
        f.append(res.fun)

    d_opt = d[np.argmax(f)]

    points = np.vstack((points, d_opt[:3]))
    radii = np.append(radii, d_opt[3])
    print('point', d_opt[:3])
    print('radius', d_opt[3])


# Plot object with PyVista
plotter = pv.Plotter()

part2 = pv.read(filepath)
plotter.add_mesh(part2, color='white', opacity=0.5)

for i in range(len(points)):
    sphere = pv.Sphere(center=points[i], radius=radii[i])
    plotter.add_mesh(sphere, color='red', opacity=0.75)

plotter.show()
