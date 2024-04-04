"""Pseudo MDBD

TODO fix for small increments
"""

import numpy as np
import jax.numpy as jnp
from jax import grad
from scipy.optimize import minimize
import pyvista as pv
import trimesh
from ..utilities.aggregation import kreisselmeier_steinhauser_max


def pseudo_mdbd(directory, input_filename, output_filename, num_spheres=1000, min_radius=0.0001, meshgrid_increment=25, plot=True, color='green'):

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

def refine_mdbd(xyzr_0):


    def objective(x):
        """Volume fraction of a 1x1x1 cube filled with spheres"""
        x = x.reshape(-1, 4)
        radii = x[:, 3]
        volumes = 4/3 * jnp.pi * radii**3
        volume = jnp.sum(volumes)

        volume_cube = 1

        # Fraction inverted to minimize (increasing volume of spheres decreases the fraction)
        volume_fraction = volume_cube / volume

        return volume_fraction

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
        surface_distances = jnp.where(jnp.eye(len(surface_distances)), 0, surface_distances)

        # Negate values so that overlap is positive
        surface_distances = -surface_distances

        # Set negative values to zero
        surface_distances = jnp.maximum(surface_distances, 0)

        surface_distances_flat = surface_distances.flatten()

        constraint = jnp.sum(surface_distances_flat)

        return constraint

    def constraint_2(x):
        """Stay within the bounds of the object"""
        x = x.reshape(-1, 4)
        positions = x[:, :3]
        radii = x[:, 3]
        lower_bound = -0.5
        upper_bound = 0.5

        lower = -(positions - radii[:, jnp.newaxis] - lower_bound)
        upper = -(upper_bound - (positions + radii[:, jnp.newaxis]))

        lower_violations = jnp.maximum(lower, 0)
        upper_violations = jnp.maximum(upper, 0)

        # Combine the violations into a single array; positive values indicate violations
        all_violations = jnp.concatenate((lower_violations.flatten(), upper_violations.flatten()))

        constraint = jnp.sum(all_violations)

        return constraint

    # grad_f = grad(objective)
    # grad_c1 = grad(constraint_1)
    # grad_c2 = grad(constraint_2)


    # Analyze initial guess
    f_0 = objective(xyzr_0)
    c1_0 = constraint_1(xyzr_0)
    c2_0 = constraint_2(xyzr_0)

    print(f'Initial guess: Volume Fraction = {f_0}')
    print(f'Max overlap = {c1_0}')
    print(f'Bounds violation = {c2_0}')

    # Run the optimization
    res = minimize(objective, xyzr_0, constraints=[{'type': 'ineq', 'fun': constraint_1}, {'type': 'ineq', 'fun': constraint_2}])

    # Analyze the result
    f_res = objective(res.x)
    c1_res = constraint_1(res.x)
    c2_res = constraint_2(res.x)

    print(f'Result: Volume Fraction = {f_res}')
    print(f'Max overlap = {c1_res}')
    print(f'Bounds violation = {c2_res}')

    return res



