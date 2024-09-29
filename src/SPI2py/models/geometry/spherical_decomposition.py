"""Pseudo MDBD

TODO fix for small increments
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.optimize import minimize
import pyvista as pv
import vtk


def compute_signed_distance(mesh, points, invert=True):
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

    # Invert the distances if needed
    if invert:
        signed_distances *= -1

    return signed_distances

def pseudo_mdbd(directory,
                filename,
                n_spheres=1000,
                n_steps=25,
                scale=1):

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
        # sphere_points = np.vstack([sphere_points, sphere_center])
        # sphere_radii = np.vstack([sphere_radii, sphere_radius])
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

    # Scale the spheres
    sphere_points *= scale
    sphere_radii *= scale

    return sphere_points, sphere_radii



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

