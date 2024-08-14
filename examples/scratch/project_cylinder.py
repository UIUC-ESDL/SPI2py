import numpy as np
import pyvista as pv
from matplotlib import cm
import matplotlib.pyplot as plt
from SPI2py.models.kinematics.distance_calculations import minimum_distance_segment_segment


def plot_density_grid(sphere_positions, sphere_radii,
                      cylinder_start_positions, cylinder_stop_positions, cylinder_radii,
                      densities):

    n, m, o = sphere_positions.shape[0], sphere_positions.shape[1], sphere_positions.shape[2]

    plotter = pv.Plotter(shape=(1, 3), window_size=(1500, 500))

    # Plot 1: The grid of spheres with uniform color
    plotter.subplot(0, 0)

    for i in range(n):
        for j in range(m):
            for k in range(o):
                pos = sphere_positions[i, j, k]
                radius = sphere_radii[i, j, k]
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
    cmap = plt.get_cmap("gray")  # Get the grayscale colormap

    for i in range(n):
        for j in range(m):
            for k in range(o):
                pos = sphere_positions[i, j, k]
                radius = sphere_radii[i, j, k]
                density = densities[i, j, k]
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
    return positions, radii

def create_cylinders(points, radius):
    x1 = np.array(points[:-1])  # Start positions (-1, 3)
    x2 = np.array(points[1:])   # Stop positions (-1, 3)
    r  = np.full((x1.shape[0], 1), line_segment_radius)
    return x1, x2, r


def phi_b(x, x1, x2, r_b):

    d_be = float(minimum_distance_segment_segment(x, x, x1, x2))

    # EQ 8 (order reversed)
    phi_b = r_b - d_be

    return phi_b


def H_tilde(x):
    # EQ 3 in 3D
    return 0.5 + 0.75 * x - 0.25 * x ** 3


def rho_b(phi_b, r):
    # EQ 2
    if phi_b / r < -1:
        return 0

    elif -1 <= phi_b / r <= 1:
        return H_tilde(phi_b / r)

    elif phi_b / r > 1:
        return 1

    else:
        raise ValueError('Something went wrong')


def calculate_density(position, radius, x1, x2, r):
    phi = phi_b(position, x1, x2, r)
    rho = rho_b(phi, radius)
    return rho

def calculate_combined_density(position, radius, X1, X2, R):

    combined_density = 0
    for x1, x2, r in zip(X1, X2, R):
        density = calculate_density(position, radius, x1, x2, r)
        combined_density += density

    # Clip the combined densities to be between 0 and 1
    combined_density = max(0, min(combined_density, 1))

    return combined_density


# Create grid
n, m, o = 5, 5, 2
positions, radii = create_grid(n, m, o, spacing=1)

# Create line segment arrays
line_segment_points = [(0, 0, 0), (2, 2, 0), (2, 4, 0)]
line_segment_radius = 0.25
X1, X2, R = create_cylinders(line_segment_points, line_segment_radius)

# Calculate the projected densities
densities = np.zeros((n, m, o))
for i in range(n):
    for j in range(m):
        for k in range(o):
            position = positions[i, j, k]
            radius = radii[i, j, k]
            combined_density = calculate_combined_density(position, radius, X1, X2, R)
            densities[i, j, k] += combined_density

plot_density_grid(positions, radii, X1.reshape(-1, 3), X2.reshape(-1, 3), R.reshape(-1, 1), densities)







# def calculate_densities(positions, radii, x1, x2, r):
#
#     densities = np.zeros_like(radii)
#
#     for i, (position, radius) in enumerate(zip(positions, radii)):
#         phi = phi_b(position, x1, x2, r)
#         rho = rho_b(phi, radius)
#         densities[i] = rho
#
#     return densities


# def calculate_combined_densities(positions, radii, X1, X2, R):
#
#     combined_densities = np.zeros_like(radii)
#
#     for x1, x2, r in zip(positions, radii, X1, X2, R):
#         densities = calculate_densities(positions, radii, x1, x2, r)
#         combined_densities += densities
#
#     # Clip the combined densities to be between 0 and 1
#     combined_densities = np.clip(combined_densities, 0, 1)
#
#     return combined_densities






