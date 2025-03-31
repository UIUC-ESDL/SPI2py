import numpy as np
import pyvista as pv


def plot_temperature_distribution(plotter,
                                  subplot_index,
                                  node_positions,
                                  node_temperatures,
                                  dims,
                                  cmap="rainbow",
                                  opacity=0.5):
    """
    Plots a 3D nodal temperature distribution on a structured grid using PyVista.

    Parameters
    ----------
    plotter : pv.Plotter
        The PyVista Plotter instance where the volume is to be added.
    subplot_index : tuple of (row, col)
        Subplot location on the plotter (e.g., (0, 0)).
    node_positions : np.ndarray, shape (N, 3)
        Array of node coordinates, where N = dims[0]*dims[1]*dims[2].
    node_temperatures : np.ndarray, shape (N,)
        Nodal temperature values.
    dims : tuple of int (nx, ny, nz)
        Structured grid dimensions: must satisfy nx*ny*nz = N.
    cmap : str, optional
        Name of the colormap to use, by default "rainbow".
    opacity : float, optional
        Opacity level for the volume rendering, by default 0.5.
    """

    # 1) Reshape the node positions into a structured (nx, ny, nz, 3) grid.
    try:
        grid_points = node_positions.reshape(dims + (3,))
    except ValueError:
        raise ValueError("Unable to reshape node_positions to dims = {}. "
                         "Expected len(node_positions) = nx*ny*nz.".format(dims))

    # 2) Create a PyVista StructuredGrid.
    grid = pv.StructuredGrid()
    # Flatten grid_points back to (N, 3) for PyVista, but the dimension metadata is stored in grid.dimensions.
    grid.points = grid_points.reshape(-1, 3)
    grid.dimensions = dims

    # 3) Attach the temperature data as a scalar field.
    grid["Temperature"] = node_temperatures

    # 4) Select the desired subplot and add the volume.
    plotter.subplot(*subplot_index)
    volume_actor = plotter.add_volume(
        grid,
        scalars="Temperature",
        cmap=cmap,
        opacity=opacity,
        show_scalar_bar=True,
        scalar_bar_args={"title": "Temperature"}
    )

    # # Optional: set linear interpolation for smoother rendering.
    # volume_actor.prop.interpolation_type = 'linear'
    #
    # # Force the volume actor's scalar range to match the data range (for consistent coloring).
    # t_min, t_max = node_temperatures.min(), node_temperatures.max()
    # volume_actor.mapper.scalar_range = (t_min, t_max)




# Define grid dimensions.
nx, ny, nz = 50, 40, 30
dims = (nx, ny, nz)

# Create a structured grid in [0,1] for each axis.
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
z = np.linspace(0, 1, nz)
# Generate a structured grid with 'ij' indexing.
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Create nodal positions: shape (nx*ny*nz, 3).
nodes = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))

# Create a smooth temperature field.
# For example, a smooth function that varies with sine and cosine.
T = 150 + 50 * np.sin(np.pi * X.ravel()) * np.cos(np.pi * Y.ravel()) * np.sin(np.pi * Z.ravel())
# Alternatively, you could try a simpler function:
# T = 300 * (X.ravel() + Y.ravel() + Z.ravel()) / 3.0

plotter = pv.Plotter(shape=(1,1))
vol = plot_temperature_distribution(plotter, (0,0), nodes, T, dims)
plotter.show()

