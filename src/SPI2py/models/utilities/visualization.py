import numpy as np
import pyvista as pv

# TODO MOVE TO API

def plot_problem(prob):
    """
    Plot the model at a given state.
    """

    # Create the plot objects

    # Plot the objects
    p = pv.Plotter(window_size=[1000, 1000])

    # # Check if the problem contains components
    # if 'components' in prob.model.system._subsystems_allprocs:
    #
    #     components = []
    #     component_colors = []
    #
    #     for subsystem in prob.model.system.components._subsystems_myproc:
    #         positions = prob.get_val('system.components.' + subsystem.name + '.transformed_points')
    #         color = subsystem.options['color']
    #
    #         # Create a point cloud using the positions
    #         point_cloud = pv.PolyData(positions)
    #
    #         components.append(point_cloud)
    #         component_colors.append(color)
    #
    #     for comp, color in zip(components, component_colors):
    #         p.add_mesh(comp, color=color, point_size=10, render_points_as_spheres=True, lighting=False)

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
        # merged_clipped = merged.clip(normal='z')
        # merged_slice = merged.slice(normal=[0, 0, 1])

        components.append(merged)
        component_colors.append(color)

    for comp, color in zip(components, component_colors):
        p.add_mesh(comp, color=color, opacity=0.5)

    # Check if the problem contains interconnects
    if 'interconnects' in prob.model.system._subsystems_allprocs:

        interconnects = []
        interconnect_colors = []
        for subsystem in prob.model.system.interconnects._subsystems_myproc:

            positions = prob.get_val('system.interconnects.' + subsystem.name + '.transformed_positions')
            radii = prob.get_val('system.interconnects.' + subsystem.name + '.transformed_radii')
            color = subsystem.options['color']

            spheres = []
            for position, radius in zip(positions, radii):
                spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

            merged = pv.MultiBlock(spheres).combine().extract_surface().clean()

            interconnects.append(merged)
            interconnect_colors.append(color)

        for inter, color in zip(interconnects, interconnect_colors):
            p.add_mesh(inter, color=color, lighting=False)

    # Check if the problem contains a bounding box
    if 'bbv' in prob.model._outputs:

        # Plot the bounding box
        bounds = prob.get_val('bbv.bounding_box_bounds')
        bounding_box = pv.Box(bounds=bounds)
        bounding_box_color = 'black'

        # Plot the bounding box
        p.add_mesh(bounding_box, color=bounding_box_color, opacity=0.2)



    # Check if the problem contains a projection
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
        grid = pv.UniformGrid()

        # Set the grid dimensions
        grid.dimensions = np.array([nx + 1, ny + 1, nz + 1])

        # Set the spacing
        grid.spacing = (spacing, spacing, spacing)

        # Set the origin of the grid to the minimum XYZ coordinates
        grid.origin = (x_min, y_min, z_min)

        p.add_mesh(grid, color='lightgrey', show_edges=True, opacity=0.15, lighting=False)

        # Plot the mdmd unit cube all spheres
        all_points = prob.get_val('mesh.all_points')
        all_points = np.array(all_points).reshape(-1,3)
        all_radii = prob.get_val('mesh.all_radii')
        all_radii = np.array(all_radii).reshape(-1,1)
        #
        # spheres = []
        # for position, radius in zip(all_points, all_radii):
        #     spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
        #
        # merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
        #
        # p.add_mesh(merged, color='black', opacity=0.1)


        # p.add_points(all_points, color='black', point_size=0.1)

        # Create a point cloud using the positions
        #         point_cloud = pv.PolyData(positions)
        #
        #         components.append(point_cloud)
        #         component_colors.append(color)
        #
        #     for comp, color in zip(components, component_colors):
        #         p.add_mesh(comp, color=color, point_size=10, render_points_as_spheres=True, lighting=False)



        # Plot projections
        for i in range(n_projections):

            # Get the object color
            color = prob.model.projections._subsystems_myproc[i].options['color']

            # Get the density values
            density_values = prob.get_val(f'projections.projection_{i}.element_pseudo_densities').flatten(order='F')
            # print("Density values range:", density_values.min(), density_values.max())
            # Create the grid

            x_grid = prob.get_val('mesh.x_grid')
            y_grid = prob.get_val('mesh.y_grid')
            z_grid = prob.get_val('mesh.z_grid')
            grid = pv.StructuredGrid(x_grid, y_grid, z_grid)
            grid["density"] = density_values



            pseudo_densities = prob.get_val(f'projections.projection_{i}.element_pseudo_densities')
            # Loop over each element in the mesh
            for n_i in range(nx):
                for n_j in range(ny):
                    for n_k in range(nz):
                        # Calculate the center of the current box
                        center = [x_min + n_i * spacing + spacing / 2,
                                  y_min + n_j * spacing + spacing / 2,
                                  z_min + n_k * spacing + spacing / 2]

                        # Create the box
                        box = pv.Box(bounds=(center[0] - spacing / 2, center[0] + spacing / 2,
                                             center[1] - spacing / 2, center[1] + spacing / 2,
                                             center[2] - spacing / 2, center[2] + spacing / 2))

                        # Add the box to the plotter with the corresponding opacity
                        p.add_mesh(box, color=color, opacity=pseudo_densities[n_i, n_j, n_k])

            # Highlight the "Highlight" Element
            ix, iy, iz = prob.get_val(f'projections.projection_{i}.highlight_element_index')

            # Calculate the center coordinates of the highlighted element
            center_x = x_min + ix * spacing + spacing / 2.0
            center_y = y_min + iy * spacing + spacing / 2.0
            center_z = z_min + iz * spacing + spacing / 2.0
            center = [center_x, center_y, center_z]

            # Option 1: Plot a sphere as a highlight marker
            highlight_marker = pv.Box(bounds=[center_x - spacing / 2.0, center_x + spacing / 2.0,
                                               center_y - spacing / 2.0, center_y + spacing / 2.0,
                                               center_z - spacing / 2.0, center_z + spacing / 2.0])
            p.add_mesh(highlight_marker, color='orange', opacity=0.2)





    p.view_isometric()
    # p.view_xy()
    p.show_axes()
    p.show_bounds(color='black')
    # p.background_color = 'white'


    p.show()