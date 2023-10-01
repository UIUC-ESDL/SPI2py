"""

"""

from SPI2py.group_model.component_geometry.spherical_decomposition_methods.maximal_disjoint_ball_decomposition import mdbd

# filename = 'ShowerHandleHolder.STL'
filename = 'engine_3'
# filename = 'stepper_gear'
num_spheres = 1000
min_radius = 0.01
increments = 30

mdbd('', 'cad_models/'+filename+'.STL', 'mdbds/'+filename+'.xyzr', num_spheres=num_spheres, min_radius=min_radius, meshgrid_increment=increments)
