"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""
import numpy as np
import openmdao.api as om
from SPI2py.API.Components import Component, Components
from SPI2py.API.Interconnects import Interconnect
from SPI2py.API.Systems import System
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file
from openmdao.api import MuxComp

# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model


model.add_subsystem('components', Components(input_dict=input_file))
# model.add_subsystem('interconnects', om.Group())
# model.add_subsystem('system', System(num_components=2))

mux = MuxComp()
model.add_subsystem('mux', mux)
n = 2
# for i in range(n):
mux.add_var('transformed_sphere_positions', shape=(100, 3))

for i in range(n):
    model.connect(f'components.comp_{i}.transformed_sphere_positions', f'mux.transformed_sphere_positions_{i}')


# Initialize interconnects

# # Add interconnects to the system
# # model.interconnects.add_subsystem('c1a_c1b', c1a_c1b)
# # model.interconnects.add_subsystem('c1b_c2b', c1b_c2b)
# # model.interconnects.add_subsystem('c2b_c3b', c2b_c3b)
# # model.interconnects.add_subsystem('c3b_c4b', c3b_c4b)
# # model.interconnects.add_subsystem('c4b_c5', c4b_c5)
# # model.interconnects.add_subsystem('c5_c6', c5_c6)
# # model.interconnects.add_subsystem('c6_c4a', c6_c4a)
# # model.interconnects.add_subsystem('c4a_c3a', c4a_c3a)
# # model.interconnects.add_subsystem('c3a_c2a', c3a_c2a)
# # model.interconnects.add_subsystem('c2a_c1a', c2a_c1a)

# Connect the components to the system


# model.connect('components.comp_0.transformed_sphere_positions', 'system.comp_0_sphere_positions')
# model.connect('components.comp_0.transformed_sphere_radii', 'system.comp_0_sphere_radii')
# model.connect('components.comp_1.transformed_sphere_positions', 'system.comp_1_sphere_positions')
# model.connect('components.comp_1.transformed_sphere_radii', 'system.comp_1_sphere_radii')


# # Connect the interconnects to the system
# # model.connect('components.radiator_and_ion_exchanger_1a.port_positions', 'interconnects.c1a_c1b.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.radiator_and_ion_exchanger_1b.port_positions', 'interconnects.c1a_c1b.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.radiator_and_ion_exchanger_1b.port_positions', 'interconnects.c1b_c2b.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.pump_2b.port_positions', 'interconnects.c1b_c2b.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.pump_2b.port_positions', 'interconnects.c2b_c3b.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.particle_filter_3b.port_positions', 'interconnects.c2b_c3b.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.particle_filter_3b.port_positions', 'interconnects.c3b_c4b.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.fuel_cell_Stack_4b.port_positions', 'interconnects.c3b_c4b.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.fuel_cell_Stack_4b.port_positions', 'interconnects.c4b_c5.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.WEG_heater_and_pump_5.port_positions', 'interconnects.c4b_c5.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.WEG_heater_and_pump_5.port_positions', 'interconnects.c5_c6.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.heater_core_6.port_positions', 'interconnects.c5_c6.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.heater_core_6.port_positions', 'interconnects.c6_c4a.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.fuel_cell_Stack_4a.port_positions', 'interconnects.c6_c4a.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.fuel_cell_Stack_4a.port_positions', 'interconnects.c4a_c3a.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.particle_filter_3a.port_positions', 'interconnects.c4a_c3a.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.particle_filter_3a.port_positions', 'interconnects.c3a_c2a.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.pump_2a.port_positions', 'interconnects.c3a_c2a.end_point', src_indices=om.slicer[0, :])
# # model.connect('components.pump_2a.port_positions', 'interconnects.c2a_c1a.start_point', src_indices=om.slicer[1, :])
# # model.connect('components.radiator_and_ion_exchanger_1a.port_positions', 'interconnects.c2a_c1a.end_point', src_indices=om.slicer[0, :])
#
# # Define the variables, objective, and constraints
#
# model.add_design_var('components.comp_0.translation', ref=10)
# model.add_design_var('components.radiator_and_ion_exchanger_1a.rotation', ref=2*3.14159)
# model.add_design_var('components.comp_1.translation', ref=10)
# model.add_design_var('components.radiator_and_ion_exchanger_1b.rotation', ref=2*3.14159)

# # prob.model.add_design_var('components.pump_2a.translation', ref=10)
# # prob.model.add_design_var('components.pump_2b.translation', ref=10)
# # prob.model.add_design_var('components.particle_filter_3a.translation', ref=10)
# # prob.model.add_design_var('components.particle_filter_3b.translation', ref=10)
# # prob.model.add_design_var('components.fuel_cell_Stack_4a.translation', ref=10)
# # prob.model.add_design_var('components.fuel_cell_Stack_4b.translation', ref=10)
# # prob.model.add_design_var('components.WEG_heater_and_pump_5.translation', ref=10)
# # prob.model.add_design_var('components.heater_core_6.translation', ref=10)

# prob.model.add_objective('system.bounding_box_volume', ref=50)
# # prob.model.add_constraint('g', upper=0)
# # # prob.model.add_constraint('g_c', upper=0)
# # # prob.model.add_constraint('g_i', upper=0)
# # # prob.model.add_constraint('g_ci', upper=0)


# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['maxiter'] = 15

# Set the initial state
prob.setup()

# Configure the system
prob.set_val('components.comp_0.translation', [2, 7, 0])
prob.set_val('components.comp_1.translation', [5.5, 7, 0])
# # prob.set_val('components.comp_2.translation', [0.5, 6, 0])
# # prob.set_val('components.comp_3.translation', [7.5, 6, 0])
# # prob.set_val('components.comp_4.translation', [0.5, 4.5, 0])
# # prob.set_val('components.comp_5.translation', [7.5, 5, 0])
# # prob.set_val('components.comp_6.translation', [0.5, 2, 0])
# # prob.set_val('components.comp_7.translation', [7.5, 3, 0])
# # prob.set_val('components.comp_8.translation', [6, 1, 0])
# # prob.set_val('components.comp_9.translation', [2, 0, 0])
#
# # prob.set_val('interconnects.c1a_c1b.control_points', [[3.75, 7, 0]])
# # prob.set_val('interconnects.c1b_c2b.control_points', [[7.5, 7, 0]])
# # prob.set_val('interconnects.c2b_c3b.control_points', [[7.5, 5.47, 0]])
# # prob.set_val('interconnects.c3b_c4b.control_points', [[7.5, 4.3175, 0]])
# # prob.set_val('interconnects.c4b_c5.control_points', [[6.5, 2, 0]])
# # prob.set_val('interconnects.c5_c6.control_points', [[3, 0, 0]])
# # prob.set_val('interconnects.c6_c4a.control_points', [[0.5, 0, 0]])
# # prob.set_val('interconnects.c4a_c3a.control_points', [[0.5, 3, 0]])
# # prob.set_val('interconnects.c3a_c2a.control_points', [[0.5, 5, 0]])
# # prob.set_val('interconnects.c2a_c1a.control_points', [[0.5, 6.5, 0]])


prob.run_model()
# Plot initial spatial configuration
# plot_problem(prob)


# Run the optimization

# print(prob.model.list_inputs(is_design_var=True))
# prob.run_driver()


# # # Plot optimized spatial
# # plot_problem(prob)
#
