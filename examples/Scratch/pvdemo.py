import numpy as np
import pyvista as pv

grid = pv.ImageData(dimensions=(9, 9, 9))

vals = -grid.x

grid['scalars'] = vals

pl = pv.Plotter()

_ = pl.add_volume(grid, opacity='linear')

pl.show()
