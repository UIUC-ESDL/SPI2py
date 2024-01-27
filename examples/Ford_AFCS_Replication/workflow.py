"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import openmdao.api as om
# from SPI2py import Component, Interconnect, System
# from SPI2py.group_model.utilities.visualization import plot
from SPI2py.API.Components import Component
from SPI2py.API.Interconnects import Interconnect
from SPI2py.API.Systems import System

from SPI2py.models.utilities.visualization import plot_problem

# Initialize the problem
prob = om.Problem()
model = prob.model
model.add_subsystem('components', om.Group())
model.add_subsystem('interconnects', om.Group())

# Initialize the System
system = System(num_components=10)
model.add_subsystem('system', system)

# Initialize components
# TODO Add names, number components...
radiator_and_ion_exchanger_1a = Component(spheres_filepath='components/radiator_and_ion_exchanger.xyzr',
                                          ports={'left': [-1.475, 0, 0], 'right': [1.475, 0, 0]},
                                          color='red')

radiator_and_ion_exchanger_1b = Component(spheres_filepath='components/radiator_and_ion_exchanger.xyzr',
                                          ports={'left': [-1.475, 0, 0], 'right': [1.475, 0, 0]},
                                          color='red')

pump_2a = Component(spheres_filepath='components/pump.xyzr',
                    ports={'top': [0, 0.275, 0], 'bottom': [0, -0.275, 0]},
                    color='blue')

pump_2b = Component(spheres_filepath='components/pump.xyzr',
                    ports={'top': [0, 0.275, 0], 'bottom': [0, -0.275, 0]},
                    color='blue')

particle_filter_3a = Component(spheres_filepath='components/particle_filter.xyzr',
                               ports={'top': [0, 0.215, 0], 'bottom': [0, -0.215, 0]},
                               color='yellow')

particle_filter_3b = Component(spheres_filepath='components/particle_filter.xyzr',
                               ports={'top': [0, 0.215, 0], 'bottom': [0, -0.215, 0]},
                               color='yellow')

fuel_cell_Stack_4a = Component(spheres_filepath='components/fuel_cell_Stack.xyzr',
                               ports={'top': [0, 0.85, 0], 'bottom': [0, -0.85, 0]},
                               color='green')

fuel_cell_Stack_4b = Component(spheres_filepath='components/fuel_cell_Stack.xyzr',
                               ports={'top': [0, 0.85, 0], 'bottom': [0, -0.85, 0]},
                               color='green')

WEG_heater_and_pump_5 = Component(spheres_filepath='components/WEG_heater_and_pump.xyzr',
                                  ports={'top': [0, 0.9665, 0], 'left': [-0.8775, 0, 0]},
                                  color='gray')

heater_core_6 = Component(spheres_filepath='components/heater_core.xyzr',
                          ports={'left': [-0.7225, 0, 0], 'right': [0.7225, 0, 0]},
                          color='red')

# Add components to the system
model.components.add_subsystem('radiator_and_ion_exchanger_1a', radiator_and_ion_exchanger_1a)
model.components.add_subsystem('radiator_and_ion_exchanger_1b', radiator_and_ion_exchanger_1b)
model.components.add_subsystem('pump_2a', pump_2a)
model.components.add_subsystem('pump_2b', pump_2b)
model.components.add_subsystem('particle_filter_3a', particle_filter_3a)
model.components.add_subsystem('particle_filter_3b', particle_filter_3b)
model.components.add_subsystem('fuel_cell_Stack_4a', fuel_cell_Stack_4a)
model.components.add_subsystem('fuel_cell_Stack_4b', fuel_cell_Stack_4b)
model.components.add_subsystem('WEG_heater_and_pump_5', WEG_heater_and_pump_5)
model.components.add_subsystem('heater_core_6', heater_core_6)

# # Initialize interconnects
c1a_c1b = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c1b_c2b = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c2b_c3b = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c3b_c4b = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c4b_c5 = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c5_c6 = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c6_c4a = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c4a_c3a = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c3a_c2a = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')
c2a_c1a = Interconnect(num_segments=3, num_spheres_per_segment=20, radius=0.1, color='black')

# Add interconnects to the system
model.interconnects.add_subsystem('c1a_c1b', c1a_c1b)
model.interconnects.add_subsystem('c1b_c2b', c1b_c2b)
model.interconnects.add_subsystem('c2b_c3b', c2b_c3b)
model.interconnects.add_subsystem('c3b_c4b', c3b_c4b)
model.interconnects.add_subsystem('c4b_c5', c4b_c5)
model.interconnects.add_subsystem('c5_c6', c5_c6)
model.interconnects.add_subsystem('c6_c4a', c6_c4a)
model.interconnects.add_subsystem('c4a_c3a', c4a_c3a)
model.interconnects.add_subsystem('c3a_c2a', c3a_c2a)
model.interconnects.add_subsystem('c2a_c1a', c2a_c1a)





# Connect the components to the system
model.connect('components.radiator_and_ion_exchanger_1a.sphere_positions', 'system.comp_0_sphere_positions')
model.connect('components.radiator_and_ion_exchanger_1a.sphere_radii', 'system.comp_0_sphere_radii')
model.connect('components.radiator_and_ion_exchanger_1b.sphere_positions', 'system.comp_1_sphere_positions')
model.connect('components.radiator_and_ion_exchanger_1b.sphere_radii', 'system.comp_1_sphere_radii')
model.connect('components.pump_2a.sphere_positions', 'system.comp_2_sphere_positions')
model.connect('components.pump_2a.sphere_radii', 'system.comp_2_sphere_radii')
model.connect('components.pump_2b.sphere_positions', 'system.comp_3_sphere_positions')
model.connect('components.pump_2b.sphere_radii', 'system.comp_3_sphere_radii')
model.connect('components.particle_filter_3a.sphere_positions', 'system.comp_4_sphere_positions')
model.connect('components.particle_filter_3a.sphere_radii', 'system.comp_4_sphere_radii')
model.connect('components.particle_filter_3b.sphere_positions', 'system.comp_5_sphere_positions')
model.connect('components.particle_filter_3b.sphere_radii', 'system.comp_5_sphere_radii')
model.connect('components.fuel_cell_Stack_4a.sphere_positions', 'system.comp_6_sphere_positions')
model.connect('components.fuel_cell_Stack_4a.sphere_radii', 'system.comp_6_sphere_radii')
model.connect('components.fuel_cell_Stack_4b.sphere_positions', 'system.comp_7_sphere_positions')
model.connect('components.fuel_cell_Stack_4b.sphere_radii', 'system.comp_7_sphere_radii')
model.connect('components.WEG_heater_and_pump_5.sphere_positions', 'system.comp_8_sphere_positions')
model.connect('components.WEG_heater_and_pump_5.sphere_radii', 'system.comp_8_sphere_radii')
model.connect('components.heater_core_6.sphere_positions', 'system.comp_9_sphere_positions')
model.connect('components.heater_core_6.sphere_radii', 'system.comp_9_sphere_radii')

# Connect the interconnects to the system
# Make specific ports outputs...
model.connect('components.radiator_and_ion_exchanger_1a.port_positions', 'interconnects.c1a_c1b.start_point', src_indices=om.slicer[1, :])
model.connect('components.radiator_and_ion_exchanger_1b.port_positions', 'interconnects.c1a_c1b.end_point', src_indices=om.slicer[0, :])
model.connect('components.radiator_and_ion_exchanger_1b.port_positions', 'interconnects.c1b_c2b.start_point', src_indices=om.slicer[1, :])
model.connect('components.pump_2b.port_positions', 'interconnects.c1b_c2b.end_point', src_indices=om.slicer[0, :])
model.connect('components.pump_2b.port_positions', 'interconnects.c2b_c3b.start_point', src_indices=om.slicer[1, :])
model.connect('components.particle_filter_3b.port_positions', 'interconnects.c2b_c3b.end_point', src_indices=om.slicer[0, :])
model.connect('components.particle_filter_3b.port_positions', 'interconnects.c3b_c4b.start_point', src_indices=om.slicer[1, :])
model.connect('components.fuel_cell_Stack_4b.port_positions', 'interconnects.c3b_c4b.end_point', src_indices=om.slicer[0, :])
model.connect('components.fuel_cell_Stack_4b.port_positions', 'interconnects.c4b_c5.start_point', src_indices=om.slicer[1, :])
model.connect('components.WEG_heater_and_pump_5.port_positions', 'interconnects.c4b_c5.end_point', src_indices=om.slicer[0, :])
model.connect('components.WEG_heater_and_pump_5.port_positions', 'interconnects.c5_c6.start_point', src_indices=om.slicer[1, :])
model.connect('components.heater_core_6.port_positions', 'interconnects.c5_c6.end_point', src_indices=om.slicer[0, :])
model.connect('components.heater_core_6.port_positions', 'interconnects.c6_c4a.start_point', src_indices=om.slicer[1, :])
model.connect('components.fuel_cell_Stack_4a.port_positions', 'interconnects.c6_c4a.end_point', src_indices=om.slicer[0, :])
model.connect('components.fuel_cell_Stack_4a.port_positions', 'interconnects.c4a_c3a.start_point', src_indices=om.slicer[1, :])
model.connect('components.particle_filter_3a.port_positions', 'interconnects.c4a_c3a.end_point', src_indices=om.slicer[0, :])
model.connect('components.particle_filter_3a.port_positions', 'interconnects.c3a_c2a.start_point', src_indices=om.slicer[1, :])
model.connect('components.pump_2a.port_positions', 'interconnects.c3a_c2a.end_point', src_indices=om.slicer[0, :])
model.connect('components.pump_2a.port_positions', 'interconnects.c2a_c1a.start_point', src_indices=om.slicer[1, :])
model.connect('components.radiator_and_ion_exchanger_1a.port_positions', 'interconnects.c2a_c1a.end_point', src_indices=om.slicer[0, :])

# Set the initial state
prob.setup()

# Configure the system
prob.set_val('components.radiator_and_ion_exchanger_1a.translation', [2, 7, 0])
prob.set_val('components.radiator_and_ion_exchanger_1b.translation', [5.5, 7, 0])
prob.set_val('components.pump_2a.translation', [0.5, 6, 0])
prob.set_val('components.pump_2b.translation', [7.5, 6, 0])
prob.set_val('components.particle_filter_3a.translation', [0.5, 4.5, 0])
prob.set_val('components.particle_filter_3b.translation', [7.5, 5, 0])
prob.set_val('components.fuel_cell_Stack_4a.translation', [0.5, 2, 0])
prob.set_val('components.fuel_cell_Stack_4b.translation', [7.5, 3, 0])
prob.set_val('components.WEG_heater_and_pump_5.translation', [6, 1, 0])
prob.set_val('components.heater_core_6.translation', [2, 0, 0])

prob.set_val('interconnects.c1a_c1b.control_points', [[3.75, 7, 0]])
prob.set_val('interconnects.c1b_c2b.control_points', [[7.5, 7, 0]])
prob.set_val('interconnects.c2b_c3b.control_points', [[7.5, 5.47, 0]])
prob.set_val('interconnects.c3b_c4b.control_points', [[7.5, 4.3175, 0]])
prob.set_val('interconnects.c4b_c5.control_points', [[6.5, 2, 0]])
prob.set_val('interconnects.c5_c6.control_points', [[3, 0, 0]])
prob.set_val('interconnects.c6_c4a.control_points', [[0.5, 0, 0]])
prob.set_val('interconnects.c4a_c3a.control_points', [[0.5, 3, 0]])
prob.set_val('interconnects.c3a_c2a.control_points', [[0.5, 5, 0]])
prob.set_val('interconnects.c2a_c1a.control_points', [[0.5, 6.5, 0]])



# prob.model.add_design_var('translations', ref=10)
# prob.model.add_design_var('rotations', ref=2*torch.pi)
#
# # TODO Unbound routing
# prob.model.add_design_var('routings', ref=10, lower=routings_0, upper=routings_0)
# prob.model.add_objective('f')
# prob.model.add_constraint('g', upper=0)
# # prob.model.add_constraint('g_c', upper=0)
# # prob.model.add_constraint('g_i', upper=0)
# # prob.model.add_constraint('g_ci', upper=0)
#


prob.run_model()

# Plot initial spatial configuration
plot_problem(prob)



# # Run the optimization
# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['maxiter'] = 50
#
# prob.run_driver()


# # Plot optimized spatial
# plot_problem(prob)







