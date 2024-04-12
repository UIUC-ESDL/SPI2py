import numpy as np
import pyvista as pv

# TODO MOVE TO API

def plot_problem(prob, plot_grid=True, plot_grid_points=True, plot_bounding_box=True, plot_projection=True):
    """
    Plot the model at a given state.
    """

    # Create the plotter
    plotter = pv.Plotter(window_size=[1000, 1000])

    # Plot the components
    components = []
    component_colors = []
    for subsystem in prob.model.system.components._subsystems_myproc:
        positions = prob.get_val('system.components.' + subsystem.name + '.transformed_sphere_positions')
        radii = prob.get_val('system.components.' + subsystem.name + '.transformed_sphere_radii')
        color = subsystem.options['color']

        spheres = []
        for position, radius in zip(positions, radii):
            spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

        merged = pv.MultiBlock(spheres).combine().extract_surface().clean()

        components.append(merged)
        component_colors.append(color)

    for comp, color in zip(components, component_colors):
        plotter.add_mesh(comp, color=color, opacity=0.5)

    # Plot the interconnects
    if 'interconnects' in prob.model.system._subsystems_allprocs:

        interconnects = []
        interconnect_colors = []
        for subsystem in prob.model.system.interconnects._subsystems_myproc:

            positions = prob.get_val('system.interconnects.' + subsystem.name + '.transformed_sphere_positions')
            radii = prob.get_val('system.interconnects.' + subsystem.name + '.transformed_sphere_radii')
            color = subsystem.options['color']

            spheres = []
            for position, radius in zip(positions, radii):
                spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

            merged = pv.MultiBlock(spheres).combine().extract_surface().clean()

            interconnects.append(merged)
            interconnect_colors.append(color)

        for inter, color in zip(interconnects, interconnect_colors):
            plotter.add_mesh(inter, color=color, lighting=False)


    if plot_bounding_box:
        bounds = prob.get_val('bbv.bounding_box_bounds')
        bounding_box = pv.Box(bounds=bounds)
        bounding_box_color = 'black'
        plotter.add_mesh(bounding_box, color=bounding_box_color, opacity=0.2)


    if 'projections' in prob.model._subsystems_allprocs:

        # Get the options
        n_comp_projections = prob.model.projections.options['n_comp_projections']
        n_int_projections = prob.model.projections.options['n_int_projections']
        n_projections = n_comp_projections + n_int_projections

        bounds = prob.model.mesh.options['bounds']
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        nx = int(prob.get_val('mesh.n_el_x'))
        ny = int(prob.get_val('mesh.n_el_y'))
        nz = int(prob.get_val('mesh.n_el_z'))
        spacing = float(prob.get_val('mesh.element_length'))

        # Create an empty uniform grid
        grid = pv.ImageData()

        # Set the grid dimensions
        grid.dimensions = np.array([nx + 1, ny + 1, nz + 1])

        # Set the spacing
        grid.spacing = (spacing, spacing, spacing)

        # Set the origin of the grid to the minimum XYZ coordinates
        grid.origin = (x_min, y_min, z_min)

        plotter.add_mesh(grid, color='lightgrey', show_edges=True, opacity=0.15, lighting=False)

        if plot_grid_points:
            # Plot the mdmd unit cube all spheres
            all_points = prob.get_val('mesh.sample_points')
            all_points = np.array(all_points).reshape(-1,3)
            all_radii = prob.get_val('mesh.sample_radii')
            all_radii = np.array(all_radii).reshape(-1,1)
            spheres = []
            for position, radius in zip(all_points, all_radii):
                spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=10, phi_resolution=10))

            merged = pv.MultiBlock(spheres).combine().extract_surface().clean()

            plotter.add_mesh(merged, color='blue', opacity=0.1)

            plotter.add_points(all_points, color='black', point_size=0.1)


        # Plot projections
        if plot_projection:
            for i in range(n_projections):

                # Get the object color
                color = prob.model.projections._subsystems_myproc[i].options['color']

                # Get the pseudo-densities
                density_values = prob.get_val(f'projections.projection_{i}.pseudo_densities').flatten(order='F')

                # Get the grid coordinates
                mesh_grid = prob.get_val('mesh.grid')
                x_grid = mesh_grid[:, :, :, 0]
                y_grid = mesh_grid[:, :, :, 1]
                z_grid = mesh_grid[:, :, :, 2]

                grid = pv.StructuredGrid(x_grid, y_grid, z_grid)
                grid["density"] = density_values

                centers = prob.get_val(f'mesh.centers')

                # Plot the projected pseudo-densities of each element (speed up by skipping near-zero densities)
                pseudo_densities = prob.get_val(f'projections.projection_{i}.pseudo_densities')
                density_threshold = 0.15
                above_threshold_indices = np.argwhere(pseudo_densities > density_threshold)
                for idx in above_threshold_indices:
                    n_i, n_j, n_k = idx

                    # Calculate the center of the current box
                    center = centers[n_i, n_j, n_k]

                    # Create the box
                    box = pv.Cube(center=center, x_length=spacing, y_length=spacing, z_length=spacing)

                    # Add the box to the plotter with the corresponding opacity
                    opacity = pseudo_densities[n_i, n_j, n_k]  # /2 is to make the boxes more transparent
                    plotter.add_mesh(box, color=color, opacity=opacity)


                # Define the opacity transfer function
                # n_points = 127
                # opacity_transfer_function = np.zeros((n_points, 2))
                #
                # # Scalar values from 0 to 1
                # opacity_transfer_function[:, 0] = np.linspace(0, 1, n_points)  # Scalar values from 0 to 1
                #
                # # Opacity values from 0 to 1
                # opacity_transfer_function[:, 1] = np.linspace(0, 1, n_points)**2

                # opacity = np.linspace(0, 1, 256).tolist()
                # tf = pv.opacity_transfer_function(opacity, 256).astype(float) / 255.0

                # # Opacity remains at 0 from scalar 0 to 0.5
                # opacity_transfer_function[:int(n_points/2), 1] = 0
                #
                # # Opacity increases from 0 to 0.5 from scalar 0.5 to 1.0
                # opacity_transfer_function[int(n_points/2):, 1] = np.linspace(0, 0.5, int(n_points/2))

                # tf = np.linspace(0, 255, 256, dtype=np.uint8)


                # color = "blue"
                # p.add_volume(grid, cmap='coolwarm', clim=[0, 1], opacity=0.5)
                # https://matplotlib.org/stable/users/explain/colors/colormaps.html
                # p.add_volume(grid, scalars="density", opacity="sigmoid", cmap='Purples', clim=[0, 1])


    # plotter.view_isometric()
    plotter.view_xy()
    plotter.show_axes()
    plotter.show_bounds(color='black')
    # p.background_color = 'white'


    plotter.show()