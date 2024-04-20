"""

"""

import pyvista as pv
from SPI2py.models.geometry.spherical_decomposition import pseudo_mdbd

# Perform MDBD on each part
num_spheres = 1000
min_radius = 0.001
meshgrid_increment = 100

pseudo_mdbd('models/',
            'stl/radiator_and_ion_exchanger.stl',
            'mdbd/radiator_and_ion_exchanger.xyzr',
            num_spheres=num_spheres,
            min_radius=min_radius,
            meshgrid_increment=meshgrid_increment,
            plot=False)

# pseudo_mdbd('models/',
#             'stl/pump.stl',
#             'mdbd/pump.xyzr',
#             num_spheres=num_spheres,
#             min_radius=min_radius,
#             meshgrid_increment=meshgrid_increment,
#             plot=False)
#
# pseudo_mdbd('models/',
#             'stl/particle_filter.stl',
#             'mdbd/particle_filter.xyzr',
#             num_spheres=num_spheres,
#             min_radius=min_radius,
#             meshgrid_increment=meshgrid_increment,
#             plot=False)
#
# pseudo_mdbd('models/',
#             'stl/fuel_cell_stack.stl',
#             'mdbd/fuel_cell_stack.xyzr',
#             num_spheres=num_spheres,
#             min_radius=min_radius,
#             meshgrid_increment=meshgrid_increment,
#             plot=False)
#
# pseudo_mdbd('models/',
#             'stl/WEG_heater_and_pump.stl',
#             'mdbd/WEG_heater_and_pump.xyzr',
#             num_spheres=num_spheres,
#             min_radius=min_radius,
#             meshgrid_increment=meshgrid_increment,
#             plot=False)
#
# pseudo_mdbd('models/',
#             'stl/heater_core.stl',
#             'mdbd/heater_core.xyzr',
#             num_spheres=num_spheres,
#             min_radius=min_radius,
#             meshgrid_increment=meshgrid_increment,
#             plot=False)
