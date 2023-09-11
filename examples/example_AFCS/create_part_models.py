"""

"""

from SPI2py.group_model.component_geometry import generate_rectangular_prisms
from SPI2py.group_model.component_geometry.spherical_decomposition_methods.other import pack_spheres

a = pack_spheres('part_models/part2.stl')

# generate_rectangular_prisms('part_models/radiator_and_ion_exchanger',
#                             [[0, 0, 0]],
#                             [[6, 2, 2]])
#
# generate_rectangular_prisms('part_models/pump',
#                             [[0, 0, 0], [0, 0, 1.5], [0, 0, 3], [1, 1, 3.5]],
#                             [[3, 3, 1.5], [3, 3, 1.5], [3, 3, 1.5], [1, 1, 5]])
#
# generate_rectangular_prisms('part_models/particle_filter',
#                             [[0, 0, 0]],
#                             [[1, 1, 1]])
#
# generate_rectangular_prisms('part_models/fuel_cell_stack',
#                             [[0, 0, 0], [1, 0, 0], [1, 1, 0.5], [1, 1, 3]],
#                             [[1, 1, 1], [1, 2, 1], [1, 1, 3], [2, 1, 1]])
#
# generate_rectangular_prisms('part_models/WEG_heater_and_pump',
#                             [[0, 0, 0]],
#                             [[2, 2, 0.5]])
#
# generate_rectangular_prisms('part_models/heater_core',
#                             [[0, 0, 0]],
#                             [[2, 2, 0.5]])
