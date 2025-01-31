"""Pseudo MDBD

TODO fix for small increments
"""

import numpy as np
import pyvista as pv
import vtk


def compute_signed_distance(mesh, points, invert=True):

    # Convert PyVista mesh to VTK polydata
    mesh_vtk = mesh

    # Create the vtkImplicitPolyDataDistance object
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(mesh_vtk)

    # Calculate the signed distance for each point
    signed_distances = np.array([implicit_distance.EvaluateFunction(point) for point in points])

    # Invert the distances if needed
    if invert:
        signed_distances *= -1

    return signed_distances

def convert_stl_to_mdbd(directory,
                        filename,
                        n_spheres=1000,
                        n_steps=25,
                        scale=1,
                        save=False):

    # Read the mesh
    mesh = pv.read(directory+filename)

    # Create a meshgrid of points
    x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
    x = np.linspace(x_min, x_max, n_steps)
    y = np.linspace(y_min, y_max, n_steps)
    z = np.linspace(z_min, z_max, n_steps)
    all_points = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T

    # Calculate inverted signed distances for all points
    signed_distances = compute_signed_distance(mesh, all_points, invert=True)

    # Remove points outside the mesh
    mask_interior = signed_distances > 0
    points_interior = all_points[mask_interior]
    distances_interior = signed_distances[mask_interior]

    # Sort points by their distance from the surface, descending
    sorted_indices = np.argsort(distances_interior)[::-1]
    points_filtered_sorted = points_interior[sorted_indices]
    distances_filtered_sorted = distances_interior[sorted_indices]

    # Scale the distances
    points_filtered_sorted *= scale
    distances_filtered_sorted *= scale

    # Preallocate arrays for sphere centers and radii
    sphere_points = np.zeros((n_spheres, 3))
    sphere_radii = np.zeros((n_spheres, 1))

    # Iterate to pack spheres until reaching the limit or the smallest sphere is smaller than min_radius
    for i in range(n_spheres):

        if distances_filtered_sorted.size == 0:
            break

        # Choose the point with the maximum distance from any surface or existing sphere
        sphere_center = points_filtered_sorted[0]
        sphere_radius = distances_filtered_sorted[0]

        # Update lists of points and distances
        sphere_points[i] = sphere_center
        sphere_radii[i] = sphere_radius

        # Update distances considering the newly added sphere
        point_distances_to_new_sphere = np.linalg.norm(points_filtered_sorted - sphere_center, axis=1)
        within_new_sphere = point_distances_to_new_sphere < sphere_radius + distances_filtered_sorted
        points_filtered_sorted = points_filtered_sorted[~within_new_sphere]
        distances_filtered_sorted = distances_filtered_sorted[~within_new_sphere]

    # Trim the arrays to remove unused entries
    sphere_points = sphere_points[:i]
    sphere_radii = sphere_radii[:i]


    # Save the spheres to a file
    if save:
        xyzr = np.hstack((sphere_points, sphere_radii))
        filename = filename.split('.')[0]
        np.savetxt(f"{directory}pseudo_mdbd_{filename}.csv", xyzr, delimiter=",")

    return sphere_points, sphere_radii


