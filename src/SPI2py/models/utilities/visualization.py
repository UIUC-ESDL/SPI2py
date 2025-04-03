import numpy as np
import pyvista as pv
from ..geometry.cylinders import create_cylinders
from ..geometry.spheres import get_aabb_bounds


def plot_grid(plotter,
              subplot_index,
              centers, size,
              densities=None,
              min_opacity=5e-3):

    # Create the subplot
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    # Plot the bounding box (unchanged)
    flat_centers = centers.reshape(-1, 3)
    x_min, y_min, z_min = np.min(flat_centers - size / 2, axis=0)
    x_max, y_max, z_max = np.max(flat_centers + size / 2, axis=0)
    aabb = pv.Box(bounds=(x_min, x_max, y_min, y_max, z_min, z_max))
    plotter.add_mesh(aabb, color='black', style='wireframe')

    # Create a PolyData point cloud from the centers
    points = pv.PolyData(flat_centers)

    # Create a cube template: a unit cube centered at (0,0,0)
    cube_template = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)
    # Scale the cube so that its edge lengths equal the desired size.
    cube_template.points *= size

    # Use the glyph filter to replicate the cube at each point.
    # Since each cube is already scaled to 'size', we disable further scaling.
    glyphs = points.glyph(orient=False, scale=False, geom=cube_template)

    # If densities are provided, create an RGBA array for per-glyph opacity.
    if densities is not None:
        flat_densities = densities.flatten()
        # Clamp densities to the range [min_opacity, 1.0]
        clamped = np.maximum(min_opacity, np.minimum(flat_densities, 1.0))
        # Convert clamped densities to alpha values in 0-255.
        alphas = (clamped * 255).astype(np.uint8)
        # Determine the number of points per glyph.
        n_input = flat_centers.shape[0]
        npts_per_glyph = glyphs.n_points // n_input
        # Repeat each alpha value so that each glyph's points get the same alpha.
        new_alphas = np.repeat(alphas, npts_per_glyph)

        # Create an RGBA array for all points in the glyph mesh.
        rgba = np.zeros((glyphs.n_points, 4), dtype=np.uint8)
        rgba[:, 3] = new_alphas  # Set the alpha channel.
        # R, G, B remain zero for black.
        glyphs.point_data["RGBA"] = rgba  # Use point_data here

        # Add the glyph mesh using the RGBA values.
        plotter.add_mesh(glyphs, rgba=True, lighting=False)
    else:
        plotter.add_mesh(glyphs, color='black', opacity=min_opacity, lighting=False)


def plot_spheres(plotter, subplot_index, positions, radii, color, opacity=0.15):

    # Create the subplot
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    # Create a sphere template.
    sphere_template = pv.Sphere(radius=1.0, theta_resolution=8, phi_resolution=8)

    # Flatten positions and radii
    flat_positions = positions.reshape(-1, 3)
    flat_radii = radii.flatten()

    # Create a point cloud from positions.
    points = pv.PolyData(np.array(flat_positions))
    # Add radii as a scalar field that the glyph filter will use for scaling.
    points["scale"] = flat_radii

    # Generate glyphs:
    # The 'scale' argument here tells the glyph filter to use the "scale" array for scaling.
    glyphs = points.glyph(orient=False, scale="scale", geom=sphere_template, factor=1.0)

    # Add the combined glyph mesh to the plotter
    plotter.add_mesh(glyphs, color=color, opacity=opacity)


def plot_capsules(plotter, subplot_index, cyl_control_points, cyl_radius, color, opacity=0.875):
    cyl_starts, cyl_stops, cyl_radii = create_cylinders(cyl_control_points, cyl_radius)

    # Create the subplot
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    for cyl_start, cyl_stop, cyl_radius in zip(cyl_starts, cyl_stops, cyl_radii):
        # Plot the spheres
        plot_spheres(plotter, subplot_index, cyl_start, cyl_radius, color, opacity)
        plot_spheres(plotter, subplot_index, cyl_stop, cyl_radius, color, opacity)

        # Plot the cylinders
        length = np.linalg.norm(cyl_stop - cyl_start)
        direction = (cyl_stop - cyl_start) / length
        center = (cyl_start + cyl_stop) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=cyl_radius[0], height=length)
        plotter.add_mesh(cylinder, color=color, opacity=opacity, lighting=False)

    # plotter.add_text("Pipe Segments", position='upper_edge', font_size=14)
    # plotter.show_bounds(all_edges=True)


def plot_capsules2(plotter, subplot_index, start_points, end_points, radii, color, opacity=0.875):

    # Create the subplot
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    for cyl_start, cyl_stop, cyl_radius in zip(start_points, end_points, radii):
        # Plot the spheres
        plot_spheres(plotter, subplot_index, cyl_start, cyl_radius, color, opacity)
        plot_spheres(plotter, subplot_index, cyl_stop, cyl_radius, color, opacity)

        # Plot the cylinders
        length = np.linalg.norm(cyl_stop - cyl_start)
        direction = (cyl_stop - cyl_start) / length
        center = (cyl_start + cyl_stop) / 2
        cylinder = pv.Cylinder(center=center, direction=direction, radius=cyl_radius, height=length)
        plotter.add_mesh(cylinder, color=color, opacity=opacity, lighting=False)

    # plotter.add_text("Pipe Segments", position='upper_edge', font_size=14)
    # plotter.show_bounds(all_edges=True)


def plot_AABB_spheres(plotter, subplot_index, centers, radii, color, opacity=0.25):
    # Create the subplot
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    # Get the AABB bounds
    x_min, x_max, y_min, y_max, z_min, z_max = get_aabb_bounds(centers, radii)

    # Create the AABB
    aabb = pv.Box([x_min, x_max, y_min, y_max, z_min, z_max])

    # Add the AABB to the plot
    plotter.add_mesh(aabb, color='black', style='wireframe', lighting=False)

    # Add the AABB to the plot
    plotter.add_mesh(aabb, color=color, opacity=opacity, lighting=False)


def plot_AABB(plotter, subplot_index, bounds, color, opacity=0.25):
    # Create the subplot
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    # Get the AABB bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    # Create the AABB
    aabb = pv.Box([x_min, x_max, y_min, y_max, z_min, z_max])

    # Add the AABB to the plot
    plotter.add_mesh(aabb, color='black', style='wireframe', lighting=False)

    # Add the AABB to the plot
    plotter.add_mesh(aabb, color=color, opacity=opacity, lighting=False)


def plot_stl_file(plotter, subplot_index, stl_file_path, translation=(0, 0, 0), rotation=(0, 0, 0), scaling=1,
                  opacity=0.5, color='lightgray'):
    """
    Plots an STL file with an optional translation.

    Parameters:
    - plotter: The PyVista plotter instance.
    - subplot_index: Tuple specifying the subplot location.
    - stl_file_path: Path to the STL file.
    - translation: Tuple (x, y, z) to shift the STL mesh.
    """
    # Create the subplot
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    # Load the STL file
    mesh = pv.read(stl_file_path)

    # Apply scaling if needed
    if scaling != 1:
        mesh.scale(scaling, inplace=True)

    # Apply translation if needed
    if translation != (0, 0, 0):
        mesh.translate(translation, inplace=True)

    # Apply rotation if needed
    if rotation != (0, 0, 0):
        rx, ry, rz = rotation
        # Get the center of the mesh
        center = mesh.center
        # Rotate around the mesh's center
        if rx != 0:
            mesh.rotate_x(rx, point=center, inplace=True)
        if ry != 0:
            mesh.rotate_y(ry, point=center, inplace=True)
        if rz != 0:
            mesh.rotate_z(rz, point=center, inplace=True)

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, color=color, opacity=opacity, lighting=False)


def plot_temperature_distribution(plotter,
                                  subplot_index,
                                  nodes,
                                  T,
                                  heat_load_nodes,
                                  robin_nodes,
                                  dirichlet_nodes,
                                  dims=None,
                                  cmap="rainbow",
                                  opacity=0.10,
                                  climits=(0, 300)):
    """
    Visualize the 3D temperature distribution on a structured grid using PyVista.

    Parameters:
        plotter          : PyVista plotter instance.
        subplot_index    : Tuple (i, j) specifying the subplot location.
      nodes            : NumPy array of shape (n_nodes,3) with node coordinates.
      T                : NumPy array of nodal temperatures.
      robin_nodes : 1D NumPy array of indices for nodes on the convection boundary.
      dirichlet_nodes      : 1D NumPy array of indices for nodes with Dirichlet conditions.
      dims             : Tuple (nx+1, ny+1, nz+1) defining grid dimensions. If None, it is inferred.
      origin           : Grid origin (default (0,0,0)).
      cmap             : Colormap for temperature (default "inferno").
      opacity          : Mesh opacity.
    """

    # Create the subplot
    plotter.subplot(*subplot_index)

    # sort_idx = np.lexsort((nodes[:, 2], nodes[:, 1], nodes[:, 0]))
    # nodes_sorted = nodes[sort_idx]
    # T_sorted = T[sort_idx]
    #
    # # Reshape nodes to a (nx, ny, nz, 3) array.
    # # Use Fortran order because VTK expects the x-index to vary fastest.
    # nx, ny, nz = dims
    # grid_points = nodes_sorted.reshape((nx, ny, nz, 3), order='F')
    #
    # # Similarly, reshape the temperature array to (nx, ny, nz)
    # T_grid = T_sorted.reshape((nx, ny, nz), order='F')
    #
    # # Create the StructuredGrid.
    # grid = pv.StructuredGrid()
    # grid.points = grid_points.reshape(-1, 3)  # Flatten back to (n_nodes,3)
    # grid.dimensions = (nx, ny, nz)
    #
    # # Attach the temperature as a point data array.
    # # (Flatten it in the same order.)
    # grid["Temperature"] = T_grid.flatten(order='F')

    # # Infer grid dimensions if not provided.
    # if dims is None:
    #     unique_x = np.unique(nodes[:, 0])
    #     unique_y = np.unique(nodes[:, 1])
    #     unique_z = np.unique(nodes[:, 2])
    #     dims = (len(unique_x), len(unique_y), len(unique_z))
    #
    # # Reshape nodes into (nx+1, ny+1, nz+1, 3) array.
    # try:
    #     grid_points = nodes.reshape(dims + (3,))
    # except ValueError:
    #     raise ValueError("Nodes cannot be reshaped to the provided dimensions.")
    #
    #
    # # Create the StructuredGrid by setting points (flattened) and dimensions.
    # grid = pv.StructuredGrid()
    # grid.points = grid_points.reshape(-1, 3)
    # grid.dimensions = dims
    # grid["Temperature"] = T
    #
    # # Setup PyVista plotter.
    # plotter.add_volume(grid, scalars="Temperature", cmap=cmap, opacity=opacity, show_scalar_bar=True, scalar_bar_args={'title': 'Temperature'})

    # Create a point cloud from the nodal positions.
    points = pv.PolyData(nodes)
    points["Temperature"] = T

    # Plot the point cloud as spheres with a color mapping based on temperature.
    # Adjust point_size for better visibility.
    actor = plotter.add_mesh(
        points,
        render_points_as_spheres=True,
        point_size=10,
        scalars="Temperature",
        cmap=cmap,
        opacity=opacity,
        clim=climits,
        show_scalar_bar=True,
        scalar_bar_args={'title': 'Temperature'}
    )


    # plot_nodes(plotter, subplot_index, nodes, heat_load_nodes, label='Heat Load', color='red', point_size=20, opacity=0.10)
    plot_nodes(plotter, subplot_index, nodes, robin_nodes, label='Robin BC', color='green')
    plot_nodes(plotter, subplot_index, nodes, dirichlet_nodes, label='Dirichlet BC', color='blue')


    # Plot the hottest and coldest points
    hottest_point = nodes[np.argmax(T)]
    coldest_point = nodes[np.argmin(T)]
    plotter.add_mesh(pv.Sphere(radius=0.5, center=hottest_point), color='red', opacity=0.25)
    plotter.add_mesh(pv.Sphere(radius=0.5, center=coldest_point), color='blue', opacity=0.25)

    plotter.add_legend()




def plot_nodes(plotter, subplot_index, nodes, selected_nodes, label, color="blue",point_size=5,opacity=1):

    plotter.subplot(*subplot_index)

    # Create PolyData for convection boundary and fixed (Dirichlet) nodes.
    node_points = nodes[selected_nodes]
    nodes_poly = pv.PolyData(node_points.reshape(-1, 3))


    # Setup PyVista plotter.
    plotter.add_mesh(nodes_poly, color=color, point_size=point_size, render_points_as_spheres=True, label=label,opacity=opacity)


def plot_translation_sensitivities(plotter, subplot_index, centers, sensitivities, color='red', factor=1.0):
    """
    Plot normalized 3D arrows representing translation sensitivities.

    Parameters:
      plotter       : PyVista Plotter object.
      subplot_index : Tuple (row, col) for subplot placement.
      centers       : (N, 3) array of 3D positions (starting points of arrows).
      sens_x, sens_y, sens_z : (N,) arrays for the translation sensitivities.
      color         : Color for the arrows.
      factor        : Scaling factor for arrow size.

    Each arrow starts at centers[i] and points in the direction given by the normalized vector
    (sens_x[i], sens_y[i], sens_z[i]).
    """
    # Set the desired subplot.
    plotter.subplot(*subplot_index)
    # plotter.render_window.SetMultiSamples(0)

    # TODO Reverse direction?
    sensitivities = -sensitivities.reshape(-1, 3)


    # Normalize each direction vector (avoiding division by zero).
    norms = np.linalg.norm(sensitivities, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions_normalized = sensitivities / norms

    # Create a point cloud from the provided centers.
    points = pv.PolyData(centers)
    points["direction"] = directions_normalized

    # Create an arrow template.
    arrow_template = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0),
                              tip_length=0.35, tip_radius=0.1, shaft_radius=0.03)

    # Generate arrow glyphs oriented by the "direction" field and scaled by factor.
    glyphs = points.glyph(orient="direction", geom=arrow_template, factor=factor)

    # Add the arrow glyph mesh to the plotter.
    plotter.add_mesh(glyphs, color=color)
    # sphere = pv.Sphere(radius=0.1, theta_resolution=8, phi_resolution=8)
    # plotter.add_mesh(sphere, color=color)