import numpy as np
import pyvista as pv
import trimesh

def read_xyz_file(filepath, n_points=100):
    """
    Reads a point cloud .xyz file and returns the points.

    TODO Remove num spheres

    :param filepath:
    :param n_points:
    :return: positions
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if n_points is not None and n_points > len(lines):
        raise ValueError('n_points must be less than the number of points in the file')

    # Set the random seed
    np.random.seed(0)

    # Randomly select n_points from the file
    lines = np.random.choice(lines, n_points)

    positions = []

    for line in lines:
        x, y, z = line.split()

        positions.append([float(x), float(y), float(z)])

    return positions


def generate_point_cloud(filepath, point_cloud_resolution=0.1, plot=False):

    axis_increments = int(1 / point_cloud_resolution)

    if axis_increments > 30:
        raise ValueError('Warning: Large values for axis increments may result in a large number of points.')

    # Create the pyvista and trimesh objects. Both are required.
    mesh_trimesh = trimesh.exchange.load.load(filepath)

    # Define variable bounds based on the object's bounding box
    x_min = mesh_trimesh.vertices[:, 0].min()
    x_max = mesh_trimesh.vertices[:, 0].max()
    y_min = mesh_trimesh.vertices[:, 1].min()
    y_max = mesh_trimesh.vertices[:, 1].max()
    z_min = mesh_trimesh.vertices[:, 2].min()
    z_max = mesh_trimesh.vertices[:, 2].max()

    # USER INPUT: The number of increments for each dimension of the meshgrid.
    nx = axis_increments
    ny = axis_increments
    nz = axis_increments

    # Create a 3d meshgrid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    xx, yy, zz = np.meshgrid(x, y, z)

    # All points inside and outside the object
    points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    # Remove points outside the objects
    signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, points)
    points_filtered = points[signed_distances > 0]

    # Calculate the relative density
    n_points_per_nxnynz_cube = len(points)
    unscaled_length = x_max - x_min
    scaled_length = 1
    scale_factor = scaled_length / unscaled_length
    n_points_per_1x1x1_cube = n_points_per_nxnynz_cube * scale_factor**3

    if plot:
        # Plot object with PyVista
        plotter = pv.Plotter()

        # Plot the filtered points
        plotter.add_points(points_filtered, color='red', point_size=5)

        # plotter.view_isometric()
        plotter.view_xz()
        plotter.background_color = 'white'
        plotter.show()

    return list(points_filtered), n_points_per_1x1x1_cube

# def generate_point_cloud(directory, input_filename, output_filename, meshgrid_increment=25, plot=True):
#
#     # Create the pyvista and trimesh objects. Both are required.
#     mesh_trimesh = trimesh.exchange.load.load(directory + input_filename)
#
#     # Define variable bounds based on the object's bounding box
#     x_min = mesh_trimesh.vertices[:, 0].min()
#     x_max = mesh_trimesh.vertices[:, 0].max()
#     y_min = mesh_trimesh.vertices[:, 1].min()
#     y_max = mesh_trimesh.vertices[:, 1].max()
#     z_min = mesh_trimesh.vertices[:, 2].min()
#     z_max = mesh_trimesh.vertices[:, 2].max()
#
#     # USER INPUT: The number of increments for each dimension of the meshgrid.
#     nx = meshgrid_increment
#     ny = meshgrid_increment
#     nz = meshgrid_increment
#
#     # Create a 3d meshgrid
#     x = np.linspace(x_min, x_max, nx)
#     y = np.linspace(y_min, y_max, ny)
#     z = np.linspace(z_min, z_max, nz)
#     xx, yy, zz = np.meshgrid(x, y, z)
#
#     # All points inside and outside the object
#     points_a = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
#
#     # All points inside the object
#     signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, points_a)
#     points_filtered = points_a[signed_distances > 0]
#
#     if plot:
#         # Plot object with PyVista
#         plotter = pv.Plotter()
#
#         # Plot the filtered points
#         plotter.add_points(points_filtered, color='red', point_size=5)
#
#         # plotter.view_isometric()
#         plotter.view_xz()
#         plotter.background_color = 'white'
#         plotter.show()
#
#
#     # OUTPUT
#
#     # Write the spheres to a text file
#     np.savetxt(directory+output_filename, points_filtered, delimiter=' ')
