"""

"""

from SPI2py.group_model.component_geometry import generate_rectangular_prisms
from SPI2py.group_model.component_geometry.spherical_decomposition_methods.maximal_disjoint_ball_decomposition import mdbd, mdbd_rectangular_prism

# a = pack_spheres('part_models/', 'engine_1.stl', 'engine_1.txt')

num_spheres = 150

mdbd_rectangular_prism([2.850, 0.830, 0.830], 'part_models/', 'radiator_and_ion_exchanger.stl', 'radiator_and_ion_exchanger.xyzr', num_spheres=num_spheres)

mdbd_rectangular_prism([0.450, 0.450, 0.450], 'part_models/', 'pump.stl', 'pump.xyzr', num_spheres=num_spheres)

mdbd_rectangular_prism([0.489, 0.330, 0.330], 'part_models/', 'particle_filter.stl', 'particle_filter.xyzr', num_spheres=num_spheres)

mdbd_rectangular_prism([1.800, 1.600, 1.600], 'part_models/', 'fuel_cell_stack.stl', 'fuel_cell_stack.xyzr', num_spheres=num_spheres)

mdbd_rectangular_prism([1.655, 1.833, 1.833], 'part_models/', 'WEG_heater_and_pump.stl', 'WEG_heater_and_pump.xyzr', num_spheres=num_spheres)

mdbd_rectangular_prism([1.345, 0.460, 0.460], 'part_models/', 'heater_core.stl', 'heater_core.xyzr', num_spheres=num_spheres)
