"""Pseudo MDBD

TODO fix for small increments
"""

import numpy as np
import pyvista as pv
import trimesh


def mdbd(directory, input_filename, output_filename, num_spheres=1000, min_radius=0.0001, meshgrid_increment=25, plot=True, color='green'):

    # Create the pyvista and trimesh objects. Both are required.
    mesh_trimesh = trimesh.exchange.load.load(directory+input_filename)

    # Define variable bounds based on the object's bounding box
    x_min = mesh_trimesh.vertices[:, 0].min()
    x_max = mesh_trimesh.vertices[:, 0].max()
    y_min = mesh_trimesh.vertices[:, 1].min()
    y_max = mesh_trimesh.vertices[:, 1].max()
    z_min = mesh_trimesh.vertices[:, 2].min()
    z_max = mesh_trimesh.vertices[:, 2].max()

    # USER INPUT: The number of increments for each dimension of the meshgrid.
    nx = meshgrid_increment
    ny = meshgrid_increment
    nz = meshgrid_increment

    # Create a 3d meshgrid
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    z = np.linspace(z_min, z_max, nz)
    xx, yy, zz = np.meshgrid(x, y, z)

    # All points inside and outside the object
    points_a = np.vstack((xx.flatten(), yy.flatten(), zz.flatten())).T

    # All points inside the object
    signed_distances = trimesh.proximity.signed_distance(mesh_trimesh, points_a)
    points_filtered = points_a[signed_distances > 0]
    distances_filtered = signed_distances[signed_distances > 0]

    # Sort the points by distance from the surface in descending order
    points_filtered_sorted = points_filtered[np.argsort(distances_filtered)[::-1]]
    distances_filtered_sorted = distances_filtered[np.argsort(distances_filtered)[::-1]]

    # Initialize the empty arrays that will store the MDBD spheres
    sphere_points = np.empty((0, 3))
    sphere_radii = np.empty((0, 1))

    # Package spheres
    for i in range(num_spheres):

        # Find the point furthest from the surface and existing spheres
        # min_distances = []
        min_distance = distances_filtered_sorted[0]
        min_distance_point = points_filtered_sorted[0]

        if min_distance < min_radius:
            "Breaking"
            break

        # Remove the point from the list of points
        points_filtered_sorted = points_filtered_sorted[1:]
        distances_filtered_sorted = distances_filtered_sorted[1:]

        sphere_points = np.vstack((sphere_points, min_distance_point))
        sphere_radii = np.vstack((sphere_radii, min_distance))

        # Remove any points that are within the max-min sphere
        distances_filtered_sorted = distances_filtered_sorted[
            np.linalg.norm(points_filtered_sorted - min_distance_point, axis=1) > min_distance]
        points_filtered_sorted = points_filtered_sorted[np.linalg.norm(points_filtered_sorted - min_distance_point, axis=1) > min_distance]


    if plot:
        # Plot object with PyVista
        plotter = pv.Plotter()

        part2 = pv.read(directory+input_filename)
        part2_slice = part2.slice(normal=[0,1,0])
        # bounds = [0, 1, 0, 1, 0, 1]
        # clipped = dataset.clip_box(bounds)
        # clipped = part2.clip_box(bounds)
        clipped = part2.clip(normal='y')
        plotter.add_mesh(part2, color='gray', opacity=0.95, silhouette=True)
        # plotter.add_mesh(clipped, color='gray', opacity=0.95, silhouette={'line_width': 2}, roughness=0)

        # Plot the sphere
        spheres = []
        for i in range(len(sphere_points)):
            sphere = pv.Sphere(center=sphere_points[i], radius=sphere_radii[i])
            # plotter.add_mesh(sphere, color='green', opacity=0.75)
            # plotter.add_mesh(sphere, color='green', opacity=0.25)
            spheres.append(sphere)

        merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
        merged_clipped = merged.clip(normal='y')
        # merged_sliced = merged.slice(normal=[0,1,0])
        # plotter.add_mesh(merged_sliced, color='green', opacity=0.25)
        # plotter.add_mesh(merged_sliced, color='green')
        plotter.add_mesh(merged, color=color, opacity=0.95)
        # plotter.add_mesh(merged_clipped, color='green', opacity=0.25)

        # plotter.view_isometric()
        plotter.view_xz()
        # plotter.enable_shadows()
        plotter.background_color = 'white'
        plotter.show()




        # slice = plotter.mesh.slice(normal=[1,1,0])
        # slice.plot()

    # OUTPUT

    # Combine the points and radii into a single array
    spheres = np.hstack((sphere_points, sphere_radii))

    # Write the spheres to a text file
    np.savetxt(directory+output_filename, spheres, delimiter=' ')
    

def mdbd_rectangular_prism(dims, directory, input_filename, output_filename, num_spheres=1000, meshgrid_increment=25, plot=True):

    l, w, h = dims

    box = pv.Cube(x_length=l, y_length=w, z_length=h)

    box.save(directory+input_filename)

    mdbd(directory, input_filename, output_filename, num_spheres=num_spheres, meshgrid_increment=meshgrid_increment, plot=plot)