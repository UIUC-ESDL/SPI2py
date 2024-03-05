"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import openmdao.api as om
from SPI2py.API.system import System
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.constraints import PairwiseCollisionDetection
from SPI2py.API.utilities import Multiplexer, MaxAggregator
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file
from itertools import combinations, product


# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Parameters
n_components = 10
n_spheres = 10
n_spheres_per_object = [n_spheres for _ in range(n_components)]
m_interconnects = 10
m_segments = 2
m_spheres_per_segment = 25
m_spheres = m_segments * m_spheres_per_segment
m_spheres_per_object = [m_spheres for _ in range(m_interconnects)]


# Initialize the groups
model.add_subsystem('system', System(input_dict=input_file))
model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_spheres_per_object + m_spheres_per_object, m=3))
model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_spheres_per_object + m_spheres_per_object, m=1))
model.add_subsystem('bbv', BoundingBoxVolume(n_spheres_per_object=n_spheres_per_object+m_spheres_per_object))


# Add a Multiplexer for component sphere positions
i = 0
for j in range(n_components):
    model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
    model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
    i += 1

# Add a Multiplexer for interconnect sphere positions and radii
for j in range(m_interconnects):
    model.connect(f'system.interconnects.int_{j}.transformed_positions', f'mux_all_sphere_positions.input_{i}')
    model.connect(f'system.interconnects.int_{j}.transformed_radii', f'mux_all_sphere_radii.input_{i}')
    i += 1

model.connect('mux_all_sphere_positions.stacked_output', 'bbv.positions')
model.connect('mux_all_sphere_radii.stacked_output', 'bbv.radii')


# Define the combinatorial collision detection pairs
component_component_pairs = list(combinations(range(n_components), 2))
interconnect_interconnect_pairs = list(combinations(range(m_interconnects), 2))
component_interconnect_pairs = list(product(range(n_components), range(m_interconnects)))

# Loop through the pairs and add the collision detection components
for i, pair in enumerate(component_component_pairs):
    collision = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=n_spheres)
    model.add_subsystem(f'collision_{i}', collision)
    model.connect(f'system.components.comp_{pair[0]}.transformed_sphere_positions', f'collision_{i}.positions_a')
    model.connect(f'system.components.comp_{pair[0]}.transformed_sphere_radii', f'collision_{i}.radii_a')
    model.connect(f'system.components.comp_{pair[1]}.transformed_sphere_positions', f'collision_{i}.positions_b')
    model.connect(f'system.components.comp_{pair[1]}.transformed_sphere_radii', f'collision_{i}.radii_b')

# for i, pair in enumerate(interconnect_interconnect_pairs):
#     collision = PairwiseCollisionDetection(n_spheres=m_spheres, m_spheres=m_spheres)
#     model.add_subsystem(f'collision_{i}', collision)
#     model.connect(f'system.interconnects.int_{pair[0]}.transformed_positions', f'collision_{i}.positions_a')
#     model.connect(f'system.interconnects.int_{pair[0]}.transformed_radii', f'collision_{i}.radii_a')
#     model.connect(f'system.interconnects.int_{pair[1]}.transformed_positions', f'collision_{i}.positions_b')
#     model.connect(f'system.interconnects.int_{pair[1]}.transformed_radii', f'collision_{i}.radii_b')

# for i, pair in enumerate(component_interconnect_pairs):
#     collision = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=m_spheres)
#     model.add_subsystem(f'collision_{i}', collision)
#     model.connect(f'system.components.comp_{pair[0]}.transformed_sphere_positions', f'collision_{i}.positions_a')
#     model.connect(f'system.components.comp_{pair[0]}.transformed_sphere_radii', f'collision_{i}.radii_a')
#     model.connect(f'system.interconnects.int_{pair[1]}.transformed_positions', f'collision_{i}.positions_b')
#     model.connect(f'system.interconnects.int_{pair[1]}.transformed_radii', f'collision_{i}.radii_b')

# Aggregate each collision detection output
for i in range(len(component_component_pairs)):
    collision_aggregation = MaxAggregator(n=n_spheres, m=n_spheres)
    model.add_subsystem(f'collision_aggregation_{i}', collision_aggregation)
    model.connect(f'collision_{i}.signed_distances', f'collision_aggregation_{i}.input_vector')

# Multiplex the aggregated collision detection outputs
collision_multiplexer = Multiplexer(n_i=[1]*len(component_component_pairs), m=1)
model.add_subsystem('collision_multiplexer', collision_multiplexer)
for i in range(len(component_component_pairs)):
    model.connect(f'collision_aggregation_{i}.aggregated_output', f'collision_multiplexer.input_{i}')

# Aggregate the multiplexed collision detection outputs
total_collision_aggregation = MaxAggregator(n=len(component_component_pairs), m=1)
model.add_subsystem('total_collision_aggregation', total_collision_aggregation)
model.connect('collision_multiplexer.stacked_output', 'total_collision_aggregation.input_vector')






# collision = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=n_spheres)
# model.add_subsystem('collision1', collision)
# model.connect('system.components.comp_0.transformed_sphere_positions', 'collision1.positions_a')
# model.connect('system.components.comp_0.transformed_sphere_radii', 'collision1.radii_a')
# model.connect('system.components.comp_1.transformed_sphere_positions', 'collision1.positions_b')
# model.connect('system.components.comp_1.transformed_sphere_radii', 'collision1.radii_b')
#
# collision2 = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=n_spheres)
# model.add_subsystem('collision2', collision2)
# model.connect('system.components.comp_0.transformed_sphere_positions', 'collision2.positions_a')
# model.connect('system.components.comp_0.transformed_sphere_radii', 'collision2.radii_a')
# model.connect('system.components.comp_2.transformed_sphere_positions', 'collision2.positions_b')
# model.connect('system.components.comp_2.transformed_sphere_radii', 'collision2.radii_b')
#
# collision3 = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=n_spheres)
# model.add_subsystem('collision3', collision3)
# model.connect('system.components.comp_1.transformed_sphere_positions', 'collision3.positions_a')
# model.connect('system.components.comp_1.transformed_sphere_radii', 'collision3.radii_a')
# model.connect('system.components.comp_2.transformed_sphere_positions', 'collision3.positions_b')
# model.connect('system.components.comp_2.transformed_sphere_radii', 'collision3.radii_b')
#
# # collision4 = PairwiseCollisionDetection(n_spheres=n_spheres, m_spheres=m_spheres)
# # model.add_subsystem('collision4', collision4)
# # model.connect('system.components.comp_0.transformed_sphere_positions', 'collision4.positions_a')
# # model.connect('system.components.comp_0.transformed_sphere_radii', 'collision4.radii_a')
# # model.connect('system.interconnects.int_0.transformed_positions', 'collision4.positions_b')
# # model.connect('system.interconnects.int_0.transformed_radii', 'collision4.radii_b')
#
# collision_aggregation1 = MaxAggregator(n=n_spheres, m=n_spheres)
# model.add_subsystem('collision_aggregation1', collision_aggregation1)
# model.connect('collision1.signed_distances', 'collision_aggregation1.input_vector')
#
# collision_aggregation2 = MaxAggregator(n=n_spheres, m=n_spheres)
# model.add_subsystem('collision_aggregation2', collision_aggregation2)
# model.connect('collision2.signed_distances', 'collision_aggregation2.input_vector')
#
# collision_aggregation3 = MaxAggregator(n=n_spheres, m=n_spheres)
# model.add_subsystem('collision_aggregation3', collision_aggregation3)
# model.connect('collision3.signed_distances', 'collision_aggregation3.input_vector')
# #
# # collision_aggregation4 = MaxAggregator(n=n_spheres, m=m_spheres)
# # model.add_subsystem('collision_aggregation4', collision_aggregation4)
# # model.connect('collision4.signed_distances', 'collision_aggregation4.input_vector')
#
# collision_multiplexer = Multiplexer(n_i=[1, 1, 1], m=1)
# model.add_subsystem('collision_multiplexer', collision_multiplexer)
# model.connect('collision_aggregation1.aggregated_output', 'collision_multiplexer.input_0')
# model.connect('collision_aggregation2.aggregated_output', 'collision_multiplexer.input_1')
# model.connect('collision_aggregation3.aggregated_output', 'collision_multiplexer.input_2')
# model.connect('collision_aggregation4.aggregated_output', 'collision_multiplexer.input_3')


# Define the variables, objective, and constraints

model.add_design_var('system.components.comp_0.translation', ref=10)
model.add_design_var('system.components.comp_0.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_1.translation', ref=10)
model.add_design_var('system.components.comp_1.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_2.translation', ref=10)
model.add_design_var('system.components.comp_2.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_3.translation', ref=10)
model.add_design_var('system.components.comp_3.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_4.translation', ref=10)
model.add_design_var('system.components.comp_4.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_5.translation', ref=10)
model.add_design_var('system.components.comp_5.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_6.translation', ref=10)
model.add_design_var('system.components.comp_6.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_7.translation', ref=10)
model.add_design_var('system.components.comp_7.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_8.translation', ref=10)
model.add_design_var('system.components.comp_8.rotation', ref=2*3.14159)
model.add_design_var('system.components.comp_9.translation', ref=10)
model.add_design_var('system.components.comp_9.rotation', ref=2*3.14159)

model.add_design_var('system.interconnects.int_0.control_points')
model.add_design_var('system.interconnects.int_1.control_points')
model.add_design_var('system.interconnects.int_2.control_points')
model.add_design_var('system.interconnects.int_3.control_points')
model.add_design_var('system.interconnects.int_4.control_points')
model.add_design_var('system.interconnects.int_5.control_points')
model.add_design_var('system.interconnects.int_6.control_points')
model.add_design_var('system.interconnects.int_7.control_points')
model.add_design_var('system.interconnects.int_8.control_points')
model.add_design_var('system.interconnects.int_9.control_points')

prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
prob.model.add_constraint('total_collision_aggregation.aggregated_output', upper=0.0)



prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['maxiter'] = 50
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-12


# Set the initial state
prob.setup()


# Configure the system
prob.set_val('system.components.comp_0.translation', [2, 7, 0])
prob.set_val('system.components.comp_1.translation', [5.5, 7, 0])
prob.set_val('system.components.comp_2.translation', [0.5, 6, 0])
prob.set_val('system.components.comp_3.translation', [7.5, 6, 0])
prob.set_val('system.components.comp_4.translation', [0.5, 4.5, 0])
prob.set_val('system.components.comp_5.translation', [7.5, 5, 0])
prob.set_val('system.components.comp_6.translation', [0.5, 2, 0])
prob.set_val('system.components.comp_7.translation', [7.5, 3, 0])
prob.set_val('system.components.comp_8.translation', [6, 1, 0])
prob.set_val('system.components.comp_9.translation', [2, 0, 0])

prob.set_val('system.interconnects.int_0.control_points', [[3.75, 7, 0]])
prob.set_val('system.interconnects.int_1.control_points', [[7.50, 7, 0]])
prob.set_val('system.interconnects.int_2.control_points', [[7.5, 5.47, 0]])
prob.set_val('system.interconnects.int_3.control_points', [[7.5, 4.3175, 0]])
prob.set_val('system.interconnects.int_4.control_points', [[6.5, 2, 0]])
prob.set_val('system.interconnects.int_5.control_points', [[3, 0, 0]])
prob.set_val('system.interconnects.int_6.control_points', [[0.5, 0, 0]])
prob.set_val('system.interconnects.int_7.control_points', [[0.5, 3, 0]])
prob.set_val('system.interconnects.int_8.control_points', [[0.5, 5, 0]])
prob.set_val('system.interconnects.int_9.control_points', [[0.5, 6.5, 0]])



prob.run_model()


# Check the initial state
plot_problem(prob)
print('Initial Objective:', prob.get_val('bbv.bounding_box_volume'))
# print('Initial Collision:', prob.get_val('collision_multiplexer.stacked_output'))


# # Run the optimization
# prob.run_driver()
#
#
# # Check the final state
# plot_problem(prob)
# print('Final Objective:', prob.get_val('bbv.bounding_box_volume'))
# print('Final Collision:', prob.get_val('collision_multiplexer.stacked_output'))
#
#
# # Troubleshooting/Debugging
# # prob.check_partials(show_only_incorrect=True, compact_print=True,includes=['bbv'])