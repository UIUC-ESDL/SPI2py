"""Pseudo MDBD

TODO fix for small increments
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.optimize import minimize
import trimesh
import pyvista as pv
from scipy.spatial import cKDTree
import open3d as o3d
from vedo import load, Points, Mesh
import vtk


def calculate_signed_distance(mesh, points):
    """
    Calculate the signed distance from a set of points to the surface of a mesh.

    Parameters:
    - mesh: pv.PolyData
      The mesh to which distances are calculated.
    - points: np.ndarray
      A (N, 3) array of points where distances are calculated.

    Returns:
    - signed_distances: np.ndarray
      An array of signed distances from each point to the mesh surface.
      Negative values indicate points inside the mesh.
    """
    # Convert PyVista mesh to VTK polydata
    mesh_vtk = mesh

    # Create the vtkImplicitPolyDataDistance object
    implicit_distance = vtk.vtkImplicitPolyDataDistance()
    implicit_distance.SetInput(mesh_vtk)

    # Calculate the signed distance for each point
    signed_distances = np.array([implicit_distance.EvaluateFunction(point) for point in points])

    return signed_distances

def pseudo_mdbd(directory, input_filename,
                num_spheres=1000,
                min_radius=0.0001,
                meshgrid_increment=25,
                scale=1):

    # Load the mesh using trimesh
    mesh_trimesh = trimesh.load(directory+input_filename)
    mesh_pyvista = pv.read(directory+input_filename)

    # Define variable bounds based on the object's bounding box
    bounds_pv = mesh_pyvista.bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds_pv

    # Create a 3D meshgrid within the bounds
    x = np.linspace(x_min, x_max, meshgrid_increment)
    y = np.linspace(y_min, y_max, meshgrid_increment)
    z = np.linspace(z_min, z_max, meshgrid_increment)
    all_points = np.array(np.meshgrid(x, y, z)).reshape(3, -1).T

    # Calculate signed distances for all points
    # signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, all_points)

    # Calculate signed distances for all points with pyvista
    # all_points_pv = pv.PolyData(all_points)
    # _ = mesh_pyvista.compute_implicit_distance(all_points_pv, inplace=True)
    # signed_distances_pv = mesh_pyvista['implicit_distance']
    signed_distances = -calculate_signed_distance(mesh_pyvista, all_points)

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
    while len(sphere_points) < num_spheres:

        if distances_filtered_sorted.size == 0:
            break

        if distances_filtered_sorted[0] < min_radius:
            break

        # Choose the point with the maximum distance from any surface or existing sphere
        sphere_center = points_filtered_sorted[0]
        sphere_radius = distances_filtered_sorted[0]

        # Update lists of points and distances
        sphere_points = np.vstack([sphere_points, sphere_center])
        sphere_radii = np.vstack([sphere_radii, sphere_radius])

        # Update distances considering the newly added sphere
        point_distances_to_new_sphere = np.linalg.norm(points_filtered_sorted - sphere_center, axis=1)
        within_new_sphere = point_distances_to_new_sphere < sphere_radius + distances_filtered_sorted
        points_filtered_sorted = points_filtered_sorted[~within_new_sphere]
        distances_filtered_sorted = distances_filtered_sorted[~within_new_sphere]

    # Scale the spheres
    sphere_points *= scale
    sphere_radii *= scale

    return sphere_points, sphere_radii


# def pseudo_mdbd(directory, input_filename,
#                 num_spheres=1000,
#                 min_radius=0.0001,
#                 meshgrid_increment=25,
#                 scale=1):
#
#     # Load the mesh using trimesh
#     mesh_trimesh = trimesh.load(directory+input_filename)
#
#     # Define variable bounds based on the object's bounding box
#     bounds = mesh_trimesh.bounds
#     x_min, y_min, z_min = bounds[0]  # Min bounds
#     x_max, y_max, z_max = bounds[1]  # Max bounds
#
#     # Create a 3D meshgrid within the bounds
#     x = np.linspace(x_min, x_max, meshgrid_increment)
#     y = np.linspace(y_min, y_max, meshgrid_increment)
#     z = np.linspace(z_min, z_max, meshgrid_increment)
#     xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
#     all_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
#
#     # Calculate signed distances for all points
#     signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, all_points)
#
#     # Filter points inside the mesh
#     interior_points = all_points[signed_distances > 0]
#     interior_distances = signed_distances[signed_distances > 0]
#
#     # Sort points by their distance from the surface, descending
#     sorted_indices = np.argsort(interior_distances)[::-1]
#     points_filtered_sorted = interior_points[sorted_indices]
#     distances_filtered_sorted = interior_distances[sorted_indices]
#
#     sphere_points = np.empty((0, 3))
#     sphere_radii = np.empty((0, 1))
#
#     # Iterate to pack spheres until reaching the limit or the smallest sphere is smaller than min_radius
#     while len(sphere_points) < num_spheres and distances_filtered_sorted.size > 0 and distances_filtered_sorted[0] > min_radius:
#
#         # Choose the point with the maximum distance from any surface or existing sphere
#         sphere_center = points_filtered_sorted[0]
#         sphere_radius = distances_filtered_sorted[0]
#
#         # Update lists of points and distances
#         sphere_points = np.vstack([sphere_points, sphere_center])
#         sphere_radii = np.vstack([sphere_radii, sphere_radius])
#
#         # Update distances considering the newly added sphere
#         point_distances_to_new_sphere = np.linalg.norm(points_filtered_sorted - sphere_center, axis=1)
#         within_new_sphere = point_distances_to_new_sphere < sphere_radius + distances_filtered_sorted
#         points_filtered_sorted = points_filtered_sorted[~within_new_sphere]
#         distances_filtered_sorted = distances_filtered_sorted[~within_new_sphere]
#
#     # Scale the spheres
#     sphere_points *= scale
#     sphere_radii *= scale
#
#     return sphere_points, sphere_radii


# def pseudo_mdbd(directory, input_filename,
#                 num_spheres=1000,
#                 min_radius=1e-6,
#                 meshgrid_increment=25,
#                 scale=1):
#
#     # Load the mesh using PyVista
#     mesh = pv.read(directory + input_filename)
#
#     # Ensure the mesh is manifold and clean (optional)
#     mesh.clean(inplace=True)
#
#     # Define variable bounds based on the object's bounding box
#     bounds = mesh.bounds
#     x_min, x_max, y_min, y_max, z_min, z_max = bounds
#
#     # Create a 3D meshgrid within the bounds
#     x = np.linspace(x_min, x_max, meshgrid_increment)
#     y = np.linspace(y_min, y_max, meshgrid_increment)
#     z = np.linspace(z_min, z_max, meshgrid_increment)
#     xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
#     all_points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T
#
#     # Create a PyVista PolyData object from the points
#     points = pv.PolyData(all_points)
#
#     # Use PyVista's select_enclosed_points method
#     enclosed = points.select_enclosed_points(mesh, tolerance=0.0)
#
#     # Extract the mask of interior points
#     interior_mask = enclosed['SelectedPoints'].astype(bool)
#     interior_points = all_points[interior_mask]
#
#     # Build a KD-tree of the mesh surface points for efficient distance computation
#     surface_points = mesh.points
#     kdtree = cKDTree(surface_points)
#
#     # Calculate signed distances from interior points to the mesh surface
#     distances, _ = kdtree.query(interior_points)
#     distances = distances.reshape(-1)
#
#     # Sort points by their distance from the surface, descending
#     sorted_indices = np.argsort(distances)[::-1]
#     points_filtered_sorted = interior_points[sorted_indices]
#     distances_filtered_sorted = distances[sorted_indices]
#
#     sphere_centers = []
#     sphere_radii = []
#
#     # Iterate to pack spheres until reaching the limit or the smallest sphere is smaller than min_radius
#     while len(sphere_centers) < num_spheres and distances_filtered_sorted.size > 0:
#         # Choose the point with the maximum distance
#         sphere_center = points_filtered_sorted[0]
#         sphere_radius = distances_filtered_sorted[0]
#
#         if sphere_radius < min_radius:
#             break
#
#         # Append the sphere's center and radius
#         sphere_centers.append(sphere_center)
#         sphere_radii.append(sphere_radius)
#
#         # Update distances to ensure no overlap with the newly added sphere
#         distances_to_new_sphere = np.linalg.norm(points_filtered_sorted - sphere_center, axis=1)
#         within_new_sphere = distances_to_new_sphere < sphere_radius
#
#         # Update the list of candidate points
#         points_filtered_sorted = points_filtered_sorted[~within_new_sphere]
#         distances_filtered_sorted = distances_filtered_sorted[~within_new_sphere]
#
#     # Convert lists to arrays and scale
#     sphere_points = np.array(sphere_centers) * scale
#     sphere_radii = np.array(sphere_radii) * scale
#
#     return sphere_points, sphere_radii

def save_mdbd(directory, output_filename, sphere_points, sphere_radii):

    # OUTPUT: Save the spheres to a file
    spheres = np.hstack((sphere_points, sphere_radii))
    np.savetxt(directory+output_filename, spheres, delimiter=' ')


def refine_mdbd(xyzr_0):


    def objective(x):
        """Volume fraction of a 1x1x1 cube filled with spheres"""
        x = x.reshape(-1, 4)
        radii = x[:, 3]
        volumes = 4/3 * jnp.pi * radii**3
        volume = jnp.sum(volumes)

        return volume

    def constraint_1(x):
        """No overlap between spheres"""
        x = x.reshape(-1, 4)
        positions = x[:, :3]
        radii = x[:, 3]

        # Calculate the center-to-center distances
        diff = positions[:, jnp.newaxis, :] - positions[jnp.newaxis, :, :]
        center_to_center_distances = jnp.sqrt(jnp.sum(diff ** 2, axis=2))

        # Calculate the surface-to-surface distances
        radii_sum = radii[:, jnp.newaxis] + radii[jnp.newaxis, :]
        surface_distances = center_to_center_distances - radii_sum

        # Ensure the diagonal is zero since we don't calculate distance from a sphere to itself
        # Use 1 to indicate constraint is not tight
        surface_distances = jnp.where(jnp.eye(len(surface_distances)), 1, surface_distances)

        # Negate values so that overlap is positive
        surface_distances = -surface_distances

        surface_distances_flat = surface_distances.flatten()

        constraint = jnp.max(surface_distances_flat)

        return constraint

    def constraint_2(x):
        """Stay within the bounds of the object"""
        x = x.reshape(-1, 4)
        positions = x[:, :3]
        radii = x[:, 3]
        lower_bound = -0.5
        upper_bound = 0.5

        lower_violations = -(positions - radii[:, jnp.newaxis] - lower_bound)
        upper_violations = -(upper_bound - (positions + radii[:, jnp.newaxis]))

        # Combine the violations into a single array; positive values indicate violations
        all_violations = jnp.concatenate((lower_violations.flatten(), upper_violations.flatten()))

        constraint = jnp.max(all_violations)

        return constraint

    grad_f = grad(objective)
    grad_c1 = grad(constraint_1)
    grad_c2 = grad(constraint_2)


    # Analyze initial guess
    f_0 = objective(xyzr_0)
    c1_0 = constraint_1(xyzr_0)
    c2_0 = constraint_2(xyzr_0)

    print(f'Initial guess: Volume Fraction = {f_0}')
    print(f'Max overlap = {c1_0}')
    print(f'Bounds violation = {c2_0}')

    # Run the optimization
    res = minimize(objective, xyzr_0,
                   constraints=[{'type': 'ineq', 'fun': constraint_1}, {'type': 'ineq', 'fun': constraint_2}])

    # Analyze the result
    f_res = objective(res.x)
    c1_res = constraint_1(res.x)
    c2_res = constraint_2(res.x)

    print(f'Result: Volume Fraction = {f_res}')
    print(f'Max overlap = {c1_res}')
    print(f'Bounds violation = {c2_res}')

    return res

