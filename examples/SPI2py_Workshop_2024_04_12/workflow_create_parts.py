from SPI2py.models.geometry.spherical_decomposition import pseudo_mdbd, save_mdbd
from SPI2py.models.utilities.visualization import plot_mdbd
from time import time_ns

directory = 'models/'
filename_bot_eye = 'Bot_Eye.stl'
filename_cog_driven_gear = 'CogDrivenGear.stl'
filename_cross_head_pin = 'CrossHead_Pin.stl'

import pyvista as pv
cog_driven_gear = pv.read(directory+filename_cog_driven_gear)
# cog_driven_gear.plot()


start = time_ns()
pos, rad = pseudo_mdbd(directory, filename_bot_eye, num_spheres=300, meshgrid_increment=25, scale=0.1)
stop = time_ns()
print(f"Elapsed time: {(stop-start)/1e9} seconds")

plot_mdbd(pos, rad)

# pos, rad = pseudo_mdbd(directory, filename_bot_eye, 'Bot_Eye.xyzr', num_spheres=1000, min_radius=0.01, meshgrid_increment=12, scale=0.1, plot=False)
# pseudo_mdbd(directory, filename_cog_driven_gear, 'CogDrivenGear.xyzr', num_spheres=1000, min_radius=0.01, meshgrid_increment=50, scale=0.1,  plot=True)
# pseudo_mdbd(directory, filename_cross_head_pin, 'CrossHead_Pin.xyzr', num_spheres=1000, min_radius=0.01, meshgrid_increment=20, scale=0.1,  plot=True)