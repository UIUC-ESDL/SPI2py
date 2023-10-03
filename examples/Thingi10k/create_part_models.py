"""

"""
import os
from SPI2py.group_model.component_geometry.spherical_decomposition_methods.maximal_disjoint_ball_decomposition import mdbd

filenames = os.listdir(path='Thingi10k_models/')

num_spheres = 1000
min_radius = 0.01
increments = 40

# for filename in filenames:
#     mdbd('', 'cad_models/'+filename, 'mdbds/'+filename+'.xyzr', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)



mdbd('', 'poets/MarqueeAlphabet_P_fixed.stl', 'poets/MarqueeAlphabet_P_fixed.xyzr', color='#1873BA', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)
mdbd('', 'poets/MarqueeAlphabet_O_fixed.stl', 'poets/MarqueeAlphabet_O_fixed.xyzr', color='#F3D316', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)
mdbd('', 'poets/MarqueeAlphabet_E_fixed.stl', 'poets/MarqueeAlphabet_E_fixed.xyzr', color='#F28223', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)
mdbd('', 'poets/MarqueeAlphabet_T_fixed.stl', 'poets/MarqueeAlphabet_T_fixed.xyzr', color='#D02029', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)
mdbd('', 'poets/MarqueeAlphabet_S_fixed.stl', 'poets/MarqueeAlphabet_S_fixed.xyzr', color='#363636', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)