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

    # Check if the problem contains components
    if 'components' in prob.model.system._subsystems_allprocs:

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
            p.add_mesh(comp, color=color)

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
            p.add_mesh(inter, color=color)

    # Check if the problem contains a bounding box
    if 'bbv' in prob.model._outputs:

        # Plot the bounding box
        bounds = prob.get_val('bbv.bounding_box_bounds')
        bounding_box = pv.Box(bounds=bounds)
        bounding_box_color = 'black'

        # Plot the bounding box
        p.add_mesh(bounding_box, color=bounding_box_color, opacity=0.2)



    # Check if the problem contains a projection
    if 'projection' in prob.model._subsystems_allprocs:
        # Assuming 'prob' is your problem instance with necessary data
        nxyz = prob.model.projection.options['n_el_xyz']
        nx, ny, nz = nxyz, nxyz, nxyz

        min_xyz = prob.model.projection.options['min_xyz']
        max_xyz = prob.model.projection.options['max_xyz']

        # Calculate the uniform spacing based on the min and max coordinates and the number of elements
        spacing = (max_xyz - min_xyz) / nxyz

        # Create an empty uniform grid
        grid = pv.UniformGrid()

        # Set the grid dimensions
        grid.dimensions = np.array([nx + 1, ny + 1, nz + 1])

        # Set the spacing
        grid.spacing = (spacing, spacing, spacing)

        # Set the origin of the grid to the minimum XYZ coordinates
        grid.origin = (min_xyz, min_xyz, min_xyz)

        p.add_mesh(grid, color='lightgrey', show_edges=True, opacity=0.5)


    p.view_isometric()
    # p.view_xy()
    p.show_axes()
    p.show_bounds(color='black')
    p.background_color = 'white'
    p.show()