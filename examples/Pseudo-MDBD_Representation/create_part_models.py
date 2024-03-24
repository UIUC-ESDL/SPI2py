"""

"""
import pyvista as pv
from SPI2py.models.geometry.maximal_disjoint_ball_decomposition import mdbd

# filenames = os.listdir(path='Thingi10k_models/')

num_spheres = 1000
min_radius = 0.01
increments = 30
#
# mdbd_unit_cube = pv.Cube(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5))
# mdbd_unit_cube.save('mdbd_unit_cube.stl')
#
# mdbd('', 'mdbd_unit_cube.stl','mdbd_unit_cube.xyzr', num_spheres=4000, min_radius=0.01, meshgrid_increment=100)

# for filename in filenames:
#     mdbd('', 'cad_models/'+filename, 'mdbds/'+filename+'.xyzr', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)
# mdbd('', 'Compressor_reduced.stl', 'Compressor_reduced.xyzr', color='#1873BA', num_spheres=num_spheres, meshgrid_increment=increments)