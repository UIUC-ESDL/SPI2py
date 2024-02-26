"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""
import numpy as np
import openmdao.api as om
from SPI2py.API.system import Component, Components, Interconnect
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.constraints import PairwiseCollisionDetection
from SPI2py.API.utilities import Multiplexer, MaxAggregator
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file


# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Parameters
n_components = 3
n_spheres = 10
n_spheres_per_object = [n_spheres, n_spheres, n_spheres]

# Initialize the components and interconnects
model.add_subsystem('components', Components(input_dict=input_file))
# model.add_subsystem('interconnects', om.Group())

# Add a Multiplexer for component sphere positions
mux_comp_sphere_positions = Multiplexer(n_i=n_spheres_per_object, m=3)
model.add_subsystem('mux_comp_sphere_positions', mux_comp_sphere_positions)
for i in range(n_components):
    model.connect(f'components.comp_{i}.transformed_sphere_positions', f'mux_comp_sphere_positions.input_{i}')

# Add a Multiplexer for component sphere radii
mux_comp_sphere_radii = Multiplexer(n_i=n_spheres_per_object, m=1)
model.add_subsystem('mux_comp_sphere_radii', mux_comp_sphere_radii)
for i in range(n_components):
    model.connect(f'components.comp_{i}.transformed_sphere_radii', f'mux_comp_sphere_radii.input_{i}')


model.add_subsystem('bbv', BoundingBoxVolume(n_spheres_per_object=n_spheres_per_object))

model.connect('mux_comp_sphere_positions.stacked_output', 'bbv.positions')
model.connect('mux_comp_sphere_radii.stacked_output', 'bbv.radii')


# Component-Component Collision Detection
collision = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=n_spheres)
model.add_subsystem('collision1', collision)
model.connect('components.comp_0.transformed_sphere_positions', 'collision1.positions_a')
model.connect('components.comp_0.transformed_sphere_radii', 'collision1.radii_a')
model.connect('components.comp_1.transformed_sphere_positions', 'collision1.positions_b')
model.connect('components.comp_1.transformed_sphere_radii', 'collision1.radii_b')

collision2 = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=n_spheres)
model.add_subsystem('collision2', collision2)
model.connect('components.comp_0.transformed_sphere_positions', 'collision2.positions_a')
model.connect('components.comp_0.transformed_sphere_radii', 'collision2.radii_a')
model.connect('components.comp_2.transformed_sphere_positions', 'collision2.positions_b')
model.connect('components.comp_2.transformed_sphere_radii', 'collision2.radii_b')

collision3 = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=n_spheres)
model.add_subsystem('collision3', collision3)
model.connect('components.comp_1.transformed_sphere_positions', 'collision3.positions_a')
model.connect('components.comp_1.transformed_sphere_radii', 'collision3.radii_a')
model.connect('components.comp_2.transformed_sphere_positions', 'collision3.positions_b')
model.connect('components.comp_2.transformed_sphere_radii', 'collision3.radii_b')

collision_aggregation1 = MaxAggregator(n=n_spheres, m=n_spheres)
model.add_subsystem('collision_aggregation1', collision_aggregation1)
model.connect('collision1.signed_distances', 'collision_aggregation1.input_vector')

collision_aggregation2 = MaxAggregator(n=n_spheres, m=n_spheres)
model.add_subsystem('collision_aggregation2', collision_aggregation2)
model.connect('collision2.signed_distances', 'collision_aggregation2.input_vector')

collision_aggregation3 = MaxAggregator(n=n_spheres, m=n_spheres)
model.add_subsystem('collision_aggregation3', collision_aggregation3)
model.connect('collision3.signed_distances', 'collision_aggregation3.input_vector')

collision_multiplexer = Multiplexer(n_i=[1, 1, 1], m=1)
model.add_subsystem('collision_multiplexer', collision_multiplexer)
model.connect('collision_aggregation1.aggregated_output', 'collision_multiplexer.input_0')
model.connect('collision_aggregation2.aggregated_output', 'collision_multiplexer.input_1')
model.connect('collision_aggregation3.aggregated_output', 'collision_multiplexer.input_2')


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
model.add_design_var('components.comp_0.translation', ref=10)
model.add_design_var('components.comp_0.rotation', ref=2*3.14159)
model.add_design_var('components.comp_1.translation', ref=10)
model.add_design_var('components.comp_1.rotation', ref=2*3.14159)

# # prob.model.add_design_var('components.pump_2a.translation', ref=10)
# # prob.model.add_design_var('components.pump_2b.translation', ref=10)
# # prob.model.add_design_var('components.particle_filter_3a.translation', ref=10)
# # prob.model.add_design_var('components.particle_filter_3b.translation', ref=10)
# # prob.model.add_design_var('components.fuel_cell_Stack_4a.translation', ref=10)
# # prob.model.add_design_var('components.fuel_cell_Stack_4b.translation', ref=10)
# # prob.model.add_design_var('components.WEG_heater_and_pump_5.translation', ref=10)
# # prob.model.add_design_var('components.heater_core_6.translation', ref=10)


prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
# prob.model.add_constraint('collision_aggregation.aggregated_output', upper=0.0)
prob.model.add_constraint('collision_multiplexer.stacked_output', upper=0.0)
# # # prob.model.add_constraint('g_c', upper=0)
# # # prob.model.add_constraint('g_i', upper=0)
# # # prob.model.add_constraint('g_ci', upper=0)


prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['maxiter'] = 50
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-12

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


# Check the initial state
# plot_problem(prob)
print('Initial Objective:', prob.get_val('bbv.bounding_box_volume'))
print('Initial Collision:', prob.get_val('collision_multiplexer.stacked_output'))


# Run the optimization
prob.run_driver()


# Check the final state
plot_problem(prob)
print('Final Objective:', prob.get_val('bbv.bounding_box_volume'))
print('Final Collision:', prob.get_val('collision_multiplexer.stacked_output'))


# Troubleshooting/Debugging
# prob.check_partials(show_only_incorrect=True, compact_print=True,includes=['bbv'])