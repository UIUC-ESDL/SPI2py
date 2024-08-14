import numpy as np
import pyvista as pv
from SPI2py.models.kinematics.distance_calculations import minimum_distance_segment_segment


def plot_density_grid(sphere_positions, sphere_radii,
                      cylinder_start_positions, cylinder_stop_positions, cylinder_radii,
                      densities):

    # n, m, o = sphere_positions.shape[0], sphere_positions.shape[1], sphere_positions.shape[2]

    plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

    # Plot 1: The grid of spheres with uniform color
    plotter.subplot(0, 0)

    for pos, radius in zip(sphere_positions, sphere_radii):
        sphere = pv.Sphere(center=pos, radius=radius)
        plotter.add_mesh(sphere, color='lightgray', opacity=0.25)

    plotter.add_text("Grid Spheres", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)


    # Plot 2: The pipe segments
    plotter.subplot(0, 1)
    for cylinder_start_position, cylinder_stop_position, cylinder_radius in zip(cylinder_start_positions, cylinder_stop_positions, cylinder_radii):
        length = np.linalg.norm(cylinder_stop_position - cylinder_start_position)
        direction = (cylinder_stop_position - cylinder_start_position) / length
        center = (cylinder_start_position + cylinder_stop_position) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=cylinder_radius, height=length)
        plotter.add_mesh(cylinder, color='blue', opacity=0.7)
    plotter.add_text("Pipe Segments", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)


    # Plot 3: The combined density with colored spheres
    plotter.subplot(0, 2)

    for pos, radius, density in zip(sphere_positions, sphere_radii, densities):
        sphere = pv.Sphere(center=pos, radius=radius)
        opacity = max(0.03, min(density, 1))
        plotter.add_mesh(sphere, color='black', opacity=opacity)

    for cylinder_start_position, cylinder_stop_position, cylinder_radius in zip(cylinder_start_positions, cylinder_stop_positions, cylinder_radii):
        length = np.linalg.norm(cylinder_stop_position - cylinder_start_position)
        direction = (cylinder_stop_position - cylinder_start_position) / length
        center = (cylinder_start_position + cylinder_stop_position) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=cylinder_radius, height=length)
        plotter.add_mesh(cylinder, color='blue', opacity=0.1)

    plotter.add_text("Combined Density", position='upper_edge', font_size=14)
    plotter.show_bounds(all_edges=True)

    plotter.link_views()
    plotter.view_isometric()
    # plotter.view_xy()

    plotter.show()


def create_grid(n, m, o, spacing=1.0):
    x = np.linspace(0, (n-1) * spacing, n)
    y = np.linspace(0, (m-1) * spacing, m)
    z = np.linspace(0, (o-1) * spacing, o)
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    positions = np.stack((xv, yv, zv), axis=-1)  # Shape (n, m, o, 3)
    radii = np.ones((n, m, o)) * spacing / 2  # Uniform radii
    return positions[..., np.newaxis, :], radii[..., np.newaxis, :]

def create_cylinders(points, radius):
    x1 = np.array(points[:-1])  # Start positions (-1, 3)
    x2 = np.array(points[1:])   # Stop positions (-1, 3)
    r  = np.full((x1.shape[0], 1), radius)
    return x1, x2, r


def signed_distance(x, x1, x2, r_b):

    # Expand dimensions to allow broadcasting
    x1 = x1[:, np.newaxis, np.newaxis, np.newaxis, :]  # Shape (-1, 1, 1, 1, 3)
    x2 = x2[:, np.newaxis, np.newaxis, np.newaxis, :]  # Shape (-1, 1, 1, 1, 3)

    # Convert output from JAX.numpy to numpy
    d_be = np.array(minimum_distance_segment_segment(x, x, x1, x2))

    phi_b = r_b - d_be
    return phi_b


def regularized_Heaviside(x):
    H_tilde = 0.5 + 0.75 * x - 0.25 * x ** 3  # EQ 3 in 3D
    return H_tilde


def density(phi_b, r):
    ratio = phi_b / r
    rho = np.where(ratio < -1, 0,
                   np.where(ratio > 1, 1,
                            regularized_Heaviside(ratio)))
    return rho


def calculate_densities(positions, radii, x1, x2, r):

    # Vectorized signed distance and density calculations using your distance function
    phi = signed_distance(positions, x1, x2, r)
    rho = density(phi, radii.T)

    # Sum densities across all cylinders
    combined_density = np.clip(np.sum(rho, axis=0), 0, 1)

    # # Sum densities across all spheres within each grid cell and all cylinders
    # combined_density = np.clip(np.sum(rho, axis=(0, 4)), 0, 1)  # Sum over the first and last axes

    return combined_density


# # Create grid
# n, m, o = 5, 5, 2
# positions, radii = create_grid(n, m, o, spacing=1)
#
# # Reshape the positions and radii
# positions = positions.reshape(-1, 3)
# radii = radii.reshape(-1, 1)
#
# # Create line segment arrays
# line_segment_points = [(0, 0, 0), (2, 2, 0), (2, 4, 0)]
# line_segment_radius = 0.25
# X1, X2, R = create_cylinders(line_segment_points, line_segment_radius)
#
# densities = calculate_densities(positions, radii, X1, X2, R)
#
#
# plot_density_grid(positions, radii, X1.reshape(-1, 3), X2.reshape(-1, 3), R.reshape(-1, 1), densities.reshape(-1, 1))

# a = np.array([[0., 0., 0.]])
# b = np.array([[1., 0., 0.]])
# c = np.array([[0., 0., 2.]])
# d = np.array([[1., 0., 1.]])
# dist = minimum_distance_segment_segment(a, b, c, d)

n, m, o, p, q = 2, 2, 2, 4, 5  # example dimensions
points = np.random.rand(n, m, o, p, 3)
start = np.random.rand(q, 3)
stop = np.random.rand(q, 3)

# Reshape arrays to allow broadcasting
points_expanded = points[..., np.newaxis, :]  # shape becomes (n, m, o, p, 1, 3)
start_expanded = start[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, 1, q, 3)
stop_expanded = stop[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :, :]  # shape becomes (1, 1, 1, 1, q, 3)

distances = minimum_distance_segment_segment(points_expanded, points_expanded, start_expanded, stop_expanded)






