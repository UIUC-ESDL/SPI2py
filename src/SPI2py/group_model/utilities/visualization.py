import pyvista as pv

def plot(components, component_colors, bounding_box, bounding_box_color):
    """
    Plot the model at a given state.
    """

    # Plot the objects
    p = pv.Plotter(window_size=[1000, 1000])

    for comp, color in zip(components, component_colors):
        p.add_mesh(comp, color=color)

    # Plot the bounding box
    p.add_mesh(bounding_box, color=bounding_box_color, opacity=0.2)

    p.view_isometric()
    # p.view_xy()
    p.show_axes()
    p.show_bounds(color='black')
    p.background_color = 'white'
    p.show()