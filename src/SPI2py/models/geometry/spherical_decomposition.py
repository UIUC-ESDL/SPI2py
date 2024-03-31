"""Pseudo MDBD

TODO fix for small increments
"""

import numpy as np
import pyvista as pv
import trimesh


def mdbd(directory, input_filename, output_filename, num_spheres=1000, min_radius=0.0001, meshgrid_increment=25, plot=True, color='green'):

    # Load the mesh using trimesh
    mesh_trimesh = trimesh.load(directory + input_filename)

    # Define variable bounds based on the object's bounding box
    bounds = mesh_trimesh.bounds
    x_min, y_min, z_min = bounds[0]  # Min bounds
    x_max, y_max, z_max = bounds[1]  # Max bounds

    # Create a 3D meshgrid within the bounds
    x = np.linspace(x_min, x_max, meshgrid_increment)
    y = np.linspace(y_min, y_max, meshgrid_increment)
    z = np.linspace(z_min, z_max, meshgrid_increment)
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    all_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    step_size = x[1] - x[0]

    # Calculate signed distances for all points
    signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, all_points)

    # Filter points inside the mesh
    interior_points = all_points[signed_distances > 0]
    interior_distances = signed_distances[signed_distances > 0]

    # Sort points by their distance from the surface, descending
    sorted_indices = np.argsort(interior_distances)[::-1]
    points_filtered_sorted = interior_points[sorted_indices]
    distances_filtered_sorted = interior_distances[sorted_indices]

    sphere_points = np.empty((0, 3))
    sphere_radii = np.empty((0, 1))

    # Iterate to pack spheres until reaching the limit or the smallest sphere is smaller than min_radius
    while len(sphere_points) < num_spheres and distances_filtered_sorted.size > 0 and distances_filtered_sorted[0] > min_radius:
        # Choose the point with the maximum distance from any surface or existing sphere
        sphere_center = points_filtered_sorted[0]
        sphere_radius = distances_filtered_sorted[0]

        # Update lists of points and distances
        sphere_points = np.vstack([sphere_points, sphere_center])
        sphere_radii = np.vstack([sphere_radii, sphere_radius])

        # Update distances considering the newly added sphere
        point_distances_to_new_sphere = np.linalg.norm(points_filtered_sorted - sphere_center, axis=1)
        # within_new_sphere = point_distances_to_new_sphere < sphere_radius + (distances_filtered_sorted-2*step_size)
        within_new_sphere = point_distances_to_new_sphere < sphere_radius + distances_filtered_sorted #+ step_size/2  # Step size is a fudge factor
        points_filtered_sorted = points_filtered_sorted[~within_new_sphere]
        distances_filtered_sorted = distances_filtered_sorted[~within_new_sphere]

    if plot:
        plotter = pv.Plotter()

        spheres = []
        for i in range(len(sphere_points)):
            sphere = pv.Sphere(center=sphere_points[i], radius=sphere_radii[i])
            spheres.append(sphere)

        merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
        plotter.add_mesh(merged, color=color, opacity=0.95)

        plotter.view_xz()
        plotter.background_color = 'white'
        plotter.show()

    # OUTPUT: Save the spheres to a file
    spheres = np.hstack((sphere_points, sphere_radii))
    np.savetxt(directory + output_filename, spheres, delimiter=' ')


def read_xyzr_file(filepath, num_spheres=100):
    """
    Reads a .xyzr file and returns the positions and radii of the spheres.

    TODO Remove num spheres

    :param filepath:
    :param num_spheres:
    :return: positions, radii
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if num_spheres is not None and num_spheres > len(lines):
        raise ValueError('num_spheres must be less than the number of spheres in the file')

    # Truncate the number of spheres as specified
    lines = lines[0:num_spheres]

    positions = []
    radii = []

    for line in lines:
        x, y, z, r = line.split()

        positions.append([float(x), float(y), float(z)])
        radii.append(float(r))

    return positions, radii
