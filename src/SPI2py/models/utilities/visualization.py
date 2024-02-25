import pyvista as pv

def plot_problem(prob):
    """
    Plot the model at a given state.
    """

    # Create the plot objects

    components = []
    component_colors = []

    for subsystem in prob.model.components._subsystems_myproc:

        positions = prob.get_val('components.' + subsystem.name + '.transformed_sphere_positions')
        radii = prob.get_val('components.' + subsystem.name + '.transformed_sphere_radii')
        color = subsystem.options['color']

        spheres = []
        for position, radius in zip(positions, radii):
            spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

        merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
        # merged_clipped = merged.clip(normal='z')
        # merged_slice = merged.slice(normal=[0, 0, 1])

        components.append(merged)
        component_colors.append(color)

    # interconnects = []
    # interconnect_colors = []
    # for subsystem in prob.model.interconnects._subsystems_myproc:
    #
    #     positions = prob.get_val('interconnects.' + subsystem.name + '.positions')
    #     radii = prob.get_val('interconnects.' + subsystem.name + '.radii')
    #     color = subsystem.options['color']
    #
    #     spheres = []
    #     for position, radius in zip(positions, radii):
    #         spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
    #
    #     merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
    #
    #     interconnects.append(merged)
    #     interconnect_colors.append(color)

    # Plot the bounding box
    bounds = prob.get_val('bbv.bounding_box_bounds')
    bounding_box = pv.Box(bounds=bounds)
    bounding_box_color = 'black'


    # Plot the objects
    p = pv.Plotter(window_size=[1000, 1000])

    for comp, color in zip(components, component_colors):
        p.add_mesh(comp, color=color)

    # for inter, color in zip(interconnects, interconnect_colors):
    #     p.add_mesh(inter, color=color)

    # Plot the bounding box
    p.add_mesh(bounding_box, color=bounding_box_color, opacity=0.2)

    p.view_isometric()
    # p.view_xy()
    p.show_axes()
    p.show_bounds(color='black')
    p.background_color = 'white'
    p.show()