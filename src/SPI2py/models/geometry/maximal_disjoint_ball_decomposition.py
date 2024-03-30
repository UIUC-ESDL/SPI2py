"""Pseudo MDBD

TODO fix for small increments
"""

import numpy as np
import pyvista as pv
import trimesh


# def mdbd_2(directory, input_filename, output_filename, num_spheres=1000, min_radius=0.0001, meshgrid_increment=25, plot=True, color='green'):
#
#     # Create the pyvista and trimesh objects. Both are required.
#     mesh_trimesh = trimesh.exchange.load.load(directory+input_filename)
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
#     # Get all the points inside and outside the object
#     all_points = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T
#
#     # Get all the points inside the object
#     signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, all_points)
#     interior_points = all_points[signed_distances > 0]
#     interior_distances = signed_distances[signed_distances > 0]
#
#     # Sort the points by distance from the surface in descending order
#     points_filtered_sorted = interior_points[np.argsort(interior_distances)[::-1]]
#     distances_filtered_sorted = interior_distances[np.argsort(interior_distances)[::-1]]
#
#     # Initialize the empty arrays that will store the largest disjoint spheres in descending order
#     sphere_points = np.empty((0, 3))
#     sphere_radii = np.empty((0, 1))
#
#     maximin_distance = distances_filtered_sorted[0]
#
#     # Package spheres
#     while maximin_distance > min_radius:
#
#         # Find the point furthest from the surface and existing spheres
#         maximin_distance = distances_filtered_sorted[0]
#         min_distance_point = points_filtered_sorted[0]
#
#         if maximin_distance < min_radius:
#             "Breaking"
#             break
#
#         # Remove the point from the list of points
#         points_filtered_sorted = points_filtered_sorted[1:]
#         distances_filtered_sorted = distances_filtered_sorted[1:]
#
#         # Add the point to the list of spheres
#         sphere_points = np.vstack((sphere_points, min_distance_point))
#         sphere_radii = np.vstack((sphere_radii, maximin_distance))
#
#         # Remove any points that are within the maximin sphere
#         distances_filtered_sorted = distances_filtered_sorted[
#             np.linalg.norm(points_filtered_sorted - min_distance_point, axis=1) > maximin_distance]
#         points_filtered_sorted = points_filtered_sorted[np.linalg.norm(points_filtered_sorted - min_distance_point, axis=1) > maximin_distance]
#
#         if len(sphere_points) >= num_spheres:
#             print('Max number of spheres reached')
#             break
#
#     if plot:
#         plotter = pv.Plotter()
#
#         spheres = []
#         for i in range(len(sphere_points)):
#             sphere = pv.Sphere(center=sphere_points[i], radius=sphere_radii[i])
#             spheres.append(sphere)
#
#         merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
#         plotter.add_mesh(merged, color=color, opacity=0.95)
#
#         plotter.view_xz()
#         plotter.background_color = 'white'
#         plotter.show()
#
#     # OUTPUT
#
#     # Combine the points and radii into a single array
#     spheres = np.hstack((sphere_points, sphere_radii))
#
#     # Write the spheres to a text file
#     np.savetxt(directory+output_filename, spheres, delimiter=' ')



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
