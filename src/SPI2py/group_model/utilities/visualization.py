import pyvista as pv

def plot(objects, colors):
    """
    Plot the model at a given state.
    """

    # Plot the objects
    p = pv.Plotter(window_size=[1000, 1000])

    for obj, color in zip(objects, colors):
        p.add_mesh(obj, color=color)

    p.view_isometric()
    # p.view_xy()
    p.show_axes()
    p.show_bounds(color='black')
    p.background_color = 'white'
    p.show()