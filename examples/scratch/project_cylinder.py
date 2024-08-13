import numpy as np
import pyvista as pv
from matplotlib import cm
import matplotlib.pyplot as plt


def plot_density_grid(sphere_positions, sphere_radii,
                      cylinder_start_positions, cylinder_stop_positions, cylinder_radii,
                      densities):

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
    cmap = plt.get_cmap("gray")  # Get the grayscale colormap
    for pos, radius, density in zip(sphere_positions, sphere_radii, densities):
        sphere = pv.Sphere(center=pos, radius=radius)
        color = cmap(density[0])[:3]  # Use the "gray" colormap for a white-to-black scale
        plotter.add_mesh(sphere, color=color, opacity=0.9)
    plotter.add_text("Combined Density", position='upper_edge', font_size=14)


    plotter.link_views()
    # plotter.view_isometric()
    plotter.view_xy()

    # Set the same bounds for all plots
    # plotter.set_scale(xscale=1, yscale=1, zscale=1)

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


def H_tilde(x):
    # EQ 3 in 3D
    return 0.5 + 0.75 * x - 0.25 * x ** 3


def rho_b(phi_b, r):
    """
    Vectorized calculation of rho_b for each row in phi_b and r.

    Parameters:
    - phi_b: (-1, 1) array, input values representing signed distances.
    - r: (-1, 1) array, input values representing the radii.

    Returns:
    - rho_b_values: (-1, 1) array, calculated rho_b values for each row.
    """

    # Calculate phi_b/r
    ratio = phi_b / r

    # Vectorized conditions
    rho_b_values = np.where(ratio < -1, 0,  # If phi_b/r < -1
                            np.where(ratio > 1, 1,  # If phi_b/r > 1
                                     H_tilde(ratio)))  # If -1 <= phi_b/r <= 1

    return rho_b_values


def calculate_densities(positions_flat, radii_flat, x1, x2, r):

    densities = []

    for position, radius in zip(positions_flat, radii_flat):
        phi = phi_b(position, x1, x2, r)
        rho = rho_b(phi, radius)
        densities.append(rho)

    densities_array = np.vstack(densities)  # Shape (-1, n_segments)

    return densities_array


# def calculate_densities(positions_flat, radii_flat, x1, x2, r):
#     # Calculate signed distances for each segment
#     densities = []
#     for start, stop, radius in zip(x1, x2, r):
#         start_arr = np.tile(start, (positions_flat.shape[0], 1))  # Match shape (-1, 3)
#         stop_arr = np.tile(stop, (positions_flat.shape[0], 1))  # Match shape (-1, 3)
#         r_arr = np.full((positions_flat.shape[0], 1), radius)  # Match shape (-1, 1) with the correct radius for the segment
#         phi = phi_b(positions_flat, start_arr, stop_arr, r_arr)
#         rho = rho_b(phi, radii_flat)  # radii_flat should be shaped correctly from the grid
#         densities.append(rho)
#
#     densities_array = np.hstack(densities)  # Shape (-1, n_segments)
#     return densities_array

# Step 4: Combine densities and reshape
# def combine_and_reshape_densities(densities_array, n, m, o):
#     combined_density = np.sum(densities_array, axis=1, keepdims=True)
#     combined_density = np.clip(combined_density, 0, 1)
#     return combined_density.reshape(n, m, o)

# line_segment_points = [(0, 0, 0), (1, 1, 1), (1, 1, 3)]
# line_segment_radius = 0.2


# Example usage
n, m, o = 5, 5, 1
line_segment_points = [(0, 0, 0), (2.5, 2.5, 0)]
line_segment_radius = 0.5

# Create grid
positions, radii = create_grid(n, m, o)
positions_flat = positions.reshape(-1, 3)  # Flattened positions (-1, 3)
radii_flat = radii.reshape(-1, 1)  # Flattened radii (-1, 1)

# Create line segment arrays
x1, x2, r = create_cylinders(line_segment_points, line_segment_radius)






# Calculate densities
# densities_array = calculate_densities(positions_flat, radii_flat, x1, x2, r)
#
# plot_density_grid(positions.reshape(-1, 3), radii.reshape(-1, 1), x1, x2, r, densities_array)




# sphere_positions, sphere_radii,
#                       cylinder_start_positions, cylinder_stop_positions, cylinder_radii,
#                       densities


# plot_density_grid(positions, radii, combined_density, line_segment_points)




# # def penalized_density(rho, alpha, q):
# #     return (alpha * rho) ** q
#
# # Define the 3D gri
#
# grid_spheres = create_grid_spheres(n, m, o, grid_spacing)
#
# # Define the pipe as a series of linear spline segments
# pipe_radius = 0.5
# pipe_segments = [
#     (np.array([2, 2, 2]), np.array([4, 4, 2])),
#     (np.array([4, 4, 2]), np.array([6, 4, 4])),
#     (np.array([6, 4, 4]), np.array([8, 6, 6]))
# ]
#
# pipe_cylinders = create_pipe(pipe_segments, pipe_radius)
#
# # Visualize using PyVista
# plotter = pv.Plotter(shape=(1, 3))
#
# # Plot the grid spheres
# plotter.subplot(0, 0)
# for sphere in grid_spheres:
#     plotter.add_mesh(sphere, color='white', opacity=0.5)
# plotter.add_text("Grid Spheres", position='upper_edge')
#
# # Plot the pipe
# plotter.subplot(0, 1)
# for cylinder in pipe_cylinders:
#     plotter.add_mesh(cylinder, color='blue', opacity=0.7)
# plotter.add_text("Pipe Segments", position='upper_edge')
#
# # Calculate and plot the projected densities
# plotter.subplot(0, 2)
#
# # Assuming rho_penalized is calculated for each grid cell (you would need to apply your functions)
# # Here, we'll use random values as a placeholder
# density_values = np.random.rand(n, m, o)
#
# # Example data
# x = np.array([[1, 2, 3], [4, 5, 6]])
# x1 = np.array([[0, 0, 0], [1, 1, 1]])
# x2 = np.array([[3, 3, 3], [7, 7, 7]])
# r_b = np.array([[1], [2]])
#
# # Calculate signed distances
# distances = signed_distance(x, x1, x2, r_b)
# print("Distances (vectorized): \n", distances)
#
# distances0 = signed_distance_s(x[0], x1[0], x2[0], r_b[0])
# distances1 = signed_distance_s(x[1], x1[1], x2[1], r_b[1])
# print("Distances: \n", distances0, '\n', distances1)
#
# # phi_v = signed_distance(x, x1, x2, r_b)
#
# # phi = signed_distance(x, x1, x2, r_b)
# # rho = projected_density(phi, r)
# # rho_penalized = penalized_density(rho, alpha, q)
#
# # for i in range(n):
# #     for j in range(m):
# #         for k in range(o):
# #             center = np.array([i, j, k]) * grid_spacing
# #             sphere = pv.Sphere(radius=grid_spacing / 2, center=center)
# #             color_value = density_values[i, j, k]
# #             color = cm.coolwarm(color_value)[:3]  # Convert to RGB
# #             plotter.add_mesh(sphere, color=color, opacity=0.5)
# #
# # plotter.add_text("Projected Densities", position='upper_edge')
# #
# # # Link all three views together
# # plotter.link_views()
# #
# # # Show the plots
# # plotter.show()






