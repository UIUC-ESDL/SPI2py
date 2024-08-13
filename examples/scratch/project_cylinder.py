import numpy as np
import pyvista as pv
from matplotlib import cm


# Define the grid cells as spheres
def create_grid_spheres(n, m, o, spacing):
    spheres = []
    for i in range(n):
        for j in range(m):
            for k in range(o):
                center = np.array([i, j, k]) * spacing
                sphere = pv.Sphere(radius=spacing / 2, center=center)
                spheres.append(sphere)
    return spheres

def create_pipe(segments, radius):
    cylinders = []
    for start, end in segments:
        length = np.linalg.norm(end - start)
        direction = (end - start) / length
        center = (start + end) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
        cylinders.append(cylinder)
    return cylinders


def H_tilde(x, dimension):
    if dimension == 2:
        return 1 - (np.arccos(x) + x * np.sqrt(1 - x ** 2)) / np.pi
    elif dimension == 3:
        return 0.5 + (3 * x) / 4 - (x ** 3) / 4


def projected_density(self, phi, r):
    x = phi / r
    return np.where(x > 1, 1, np.where(x < -1, 0, self.H_tilde(x, self.options['dimension'])))

def signed_distance_s(x, x1, x2, r_b):
    """..."""

    # EQ 9
    x21 = x2 - x1
    xe_1b = x - x1

    # EQs 10, 11, 12, 13
    l_b = np.linalg.norm(x21)
    a_b = x21 / l_b
    l_be = np.dot(xe_1b, a_b)
    r_be = np.linalg.norm(xe_1b - l_be * a_b)

    # EQ 14
    d_be_1 = np.linalg.norm(xe_1b)
    d_be_2 = np.linalg.norm(x - x2)
    d_be_3 = r_be

    if l_be <= 0:
        d_be = d_be_1
    elif l_be < l_b:
        d_be = d_be_2
    else:
        d_be = d_be_3

    # EQ 8
    phi_b = d_be - r_b

    return phi_b


def signed_distance(x, x1, x2, r_b):
    """
    Calculate the signed distance from each point x to the corresponding line segment defined by x1, x2, and radii r_b.

    Parameters:
    - x: (-1, 3) array, points from which the distances are calculated.
    - x1: (-1, 3) array, start points of the line segments.
    - x2: (-1, 3) array, end points of the line segments.
    - r_b: (-1, 1) array, radii of the line segments.

    Returns:
    - phi_b: (-1, 1) array, signed distances from each point x to the corresponding line segment.
   """

    # EQ 9
    x21 = x2 - x1
    xe_1b = x - x1

    # EQs 10, 11, 12, 13
    l_b = np.linalg.norm(x21, axis=1, keepdims=True)
    a_b = x21 / l_b
    l_be = np.sum(xe_1b * a_b, axis=1, keepdims=True)
    r_be = np.linalg.norm(xe_1b - l_be * a_b, axis=1, keepdims=True)

    # EQ 14
    d_be_1 = np.linalg.norm(xe_1b, axis=1, keepdims=True)
    d_be_2 = np.linalg.norm(x - x2, axis=1, keepdims=True)
    d_be_3 = r_be

    d_be = np.where(l_be <= 0, d_be_1,  # if l_be <= 0
                    np.where(l_be < l_b, d_be_2,  # if l_be < l_b
                    d_be_3))  # otherwise

    # EQ 8
    phi_b = d_be - r_b

    return phi_b


def penalized_density(rho, alpha, q):
    return (alpha * rho) ** q

# Define the 3D grid
n, m, o = 10, 10, 10  # Grid dimensions
grid_spacing = 1.0  # Distance between grid cells

grid_spheres = create_grid_spheres(n, m, o, grid_spacing)

# Define the pipe as a series of linear spline segments
pipe_radius = 0.5
pipe_segments = [
    (np.array([2, 2, 2]), np.array([4, 4, 2])),
    (np.array([4, 4, 2]), np.array([6, 4, 4])),
    (np.array([6, 4, 4]), np.array([8, 6, 6]))
]

pipe_cylinders = create_pipe(pipe_segments, pipe_radius)

# Visualize using PyVista
plotter = pv.Plotter(shape=(1, 3))

# Plot the grid spheres
plotter.subplot(0, 0)
for sphere in grid_spheres:
    plotter.add_mesh(sphere, color='white', opacity=0.5)
plotter.add_text("Grid Spheres", position='upper_edge')

# Plot the pipe
plotter.subplot(0, 1)
for cylinder in pipe_cylinders:
    plotter.add_mesh(cylinder, color='blue', opacity=0.7)
plotter.add_text("Pipe Segments", position='upper_edge')

# Calculate and plot the projected densities
plotter.subplot(0, 2)

# Assuming rho_penalized is calculated for each grid cell (you would need to apply your functions)
# Here, we'll use random values as a placeholder
density_values = np.random.rand(n, m, o)

# Example data
x = np.array([[1, 2, 3], [4, 5, 6]])
x1 = np.array([[0, 0, 0], [1, 1, 1]])
x2 = np.array([[3, 3, 3], [7, 7, 7]])
r_b = np.array([[1], [2]])

# Calculate signed distances
distances = signed_distance(x, x1, x2, r_b)
print("Distances (vectorized): \n", distances)

distances0 = signed_distance_s(x[0], x1[0], x2[0], r_b[0])
distances1 = signed_distance_s(x[1], x1[1], x2[1], r_b[1])
print("Distances: \n", distances0, '\n', distances1)

# phi_v = signed_distance(x, x1, x2, r_b)

# phi = signed_distance(x, x1, x2, r_b)
# rho = projected_density(phi, r)
# rho_penalized = penalized_density(rho, alpha, q)

# for i in range(n):
#     for j in range(m):
#         for k in range(o):
#             center = np.array([i, j, k]) * grid_spacing
#             sphere = pv.Sphere(radius=grid_spacing / 2, center=center)
#             color_value = density_values[i, j, k]
#             color = cm.coolwarm(color_value)[:3]  # Convert to RGB
#             plotter.add_mesh(sphere, color=color, opacity=0.5)
#
# plotter.add_text("Projected Densities", position='upper_edge')
#
# # Link all three views together
# plotter.link_views()
#
# # Show the plots
# plotter.show()






