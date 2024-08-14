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

    # for i in range(n):
    #     for j in range(m):
    #         for k in range(o):
    #             pos = sphere_positions[i, j, k]
    #             radius = sphere_radii[i, j, k]
    #             sphere = pv.Sphere(center=pos, radius=radius)
    #             plotter.add_mesh(sphere, color='lightgray', opacity=0.25)

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

    # for i in range(n):
    #     for j in range(m):
    #         for k in range(o):
    #             pos = sphere_positions[i, j, k]
    #             radius = sphere_radii[i, j, k]
    #             density = densities[i, j, k]
    #             sphere = pv.Sphere(center=pos, radius=radius)
    #             opacity = max(0.03, min(density, 1))
    #             plotter.add_mesh(sphere, color='black', opacity=opacity)

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
    return positions, radii

def create_cylinders(points, radius):
    x1 = np.array(points[:-1])  # Start positions (-1, 3)
    x2 = np.array(points[1:])   # Stop positions (-1, 3)
    r  = np.full((x1.shape[0], 1), line_segment_radius)
    return x1, x2, r


def signed_distance(x, x1, x2, r_b):
    d_be = np.array(minimum_distance_segment_segment(x, x, x1, x2))
    phi_b = r_b.reshape(-1,1) - d_be  # Adjust the shape of r_b for broadcasting
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


# def calculate_densities(positions, radii, x1, x2, r):
#     phi = signed_distance(positions, x1, x2, r)
#     rho = density(phi, radii)
#     return rho

def calculate_densities(positions, radii, x1, x2, r):

    # Expand dimensions to allow broadcasting
    positions = positions[np.newaxis, :, :]  # Shape (1, -1, 3)
    radii = radii[np.newaxis, :, :]  # Shape (1, -1, 1)
    x1 = x1[:, np.newaxis, :]  # Shape (-1, 1, 3)
    x2 = x2[:, np.newaxis, :]  # Shape (-1, 1, 3)
    r = r[:, np.newaxis, :]  # Shape (-1, 1, 1)

    # Vectorized signed distance and density calculations using your distance function
    phi = signed_distance(positions, x1, x2, r)
    rho = density(phi, radii.reshape(1,-1))

    # Sum densities across all cylinders
    combined_density = np.clip(np.sum(rho, axis=0), 0, 1)

    return combined_density


# Create grid
n, m, o = 5, 5, 2
positions, radii = create_grid(n, m, o, spacing=1)

# Reshape the positions and radii
positions = positions.reshape(-1, 3)
radii = radii.reshape(-1, 1)

# Create line segment arrays
line_segment_points = [(0, 0, 0), (2, 2, 0), (2, 4, 0)]
line_segment_radius = 0.25
X1, X2, R = create_cylinders(line_segment_points, line_segment_radius)

densities = calculate_densities(positions, radii, X1, X2, R)


# x1, x2, r = X1[0], X2[0], R[0]
# pos = positions.reshape(-1, 3)
# rad = radii.reshape(-1, 1)


# # Calculate the projected densities
# densities = np.zeros((n, m, o))
# for i in range(n):
#     for j in range(m):
#         for k in range(o):
#             position = positions[i, j, k]
#             radius = radii[i, j, k]
#             combined_density = calculate_combined_density(position, radius, X1, X2, R)
#             densities[i, j, k] += combined_density

plot_density_grid(positions, radii, X1.reshape(-1, 3), X2.reshape(-1, 3), R.reshape(-1, 1), densities)









# def calculate_density(position, radius, x1, x2, r):
#     phi = signed_distance(position, x1, x2, r)
#     rho = density(phi, radius)
#     return rho

# def calculate_combined_density(position, radius, X1, X2, R):
#
#     combined_density = 0
#     for x1, x2, r in zip(X1, X2, R):
#         density = calculate_density(position, radius, x1, x2, r)
#         combined_density += density
#
#     # Clip the combined densities to be between 0 and 1
#     combined_density = max(0, min(combined_density, 1))
#
#     return combined_density

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






