"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import openmdao.api as om
from SPI2py.API.system import System
from SPI2py.API.projection import Mesh, Projections, ProjectionAggregator
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer, MaxAggregator

from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file

# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# System Parameters
n_components = 10
n_spheres = 10
n_spheres_per_object = [n_spheres for _ in range(n_components)]
m_interconnects = 10
m_segments = 2
m_spheres_per_segment = 10
m_spheres = m_segments * m_spheres_per_segment
m_spheres_per_object = [m_spheres for _ in range(m_interconnects)]

# Mesh Parameters
bounds = (0, 10, 0, 10, 0, 4)
n_elements_per_unit_length = 2.0

# Projection Parameters
n_projections = n_components + m_interconnects
n_points_per_object = n_spheres_per_object + m_spheres_per_object

# Initialize the subsystems
model.add_subsystem('system', System(input_dict=input_file, upper=7, lower=0))
model.add_subsystem('mesh', Mesh(bounds=bounds,
                                 n_elements_per_unit_length=n_elements_per_unit_length))

model.add_subsystem('projections', Projections(n_comp_projections=n_components,
                                               n_int_projections=m_interconnects))

model.add_subsystem('aggregator', ProjectionAggregator(n_projections=n_projections))


model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_points_per_object, m=3))
model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_points_per_object, m=1))
model.add_subsystem('bbv', BoundingBoxVolume())

# Connect the system to the projections
i = 0
for j in range(n_components):
    model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'projections.projection_{i}.sphere_positions')
    model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'projections.projection_{i}.sphere_radii')
    model.connect(f'system.components.comp_{j}.volume', f'projections.projection_{i}.volume')
    model.connect(f'system.components.comp_{j}.AABB', f'projections.projection_{i}.AABB')
    model.connect(f'projections.projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
    i += 1
for j in range(m_interconnects):
    model.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', f'projections.projection_{i}.sphere_positions')
    model.connect(f'system.interconnects.int_{j}.transformed_sphere_radii', f'projections.projection_{i}.sphere_radii')
    model.connect(f'system.interconnects.int_{j}.volume', f'projections.projection_{i}.volume')
    model.connect(f'system.interconnects.int_{j}.AABB', f'projections.projection_{i}.AABB')
    model.connect(f'projections.projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
    i += 1


# Connect the mesh to the projections
model.connect('mesh.element_length', 'aggregator.element_length')
for i in range(n_projections):
    model.connect('mesh.element_length', f'projections.projection_{i}.element_length')
    model.connect('mesh.centers', f'projections.projection_{i}.centers')
    model.connect('mesh.element_bounds', f'projections.projection_{i}.element_bounds')
    model.connect('mesh.sample_points', f'projections.projection_{i}.sample_points')
    model.connect('mesh.sample_radii', f'projections.projection_{i}.sample_radii')


# Connect the system to the bounding box
i = 0
for j in range(n_components):
    model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
    model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
    i += 1
for j in range(m_interconnects):
    model.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
    model.connect(f'system.interconnects.int_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
    i += 1

model.connect('mux_all_sphere_positions.stacked_output', 'bbv.sphere_positions')
model.connect('mux_all_sphere_radii.stacked_output', 'bbv.sphere_radii')


# Define the objective and constraints
prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
prob.model.add_constraint('aggregator.max_pseudo_density', upper=1.0)

prob.model.add_design_var('system.components.comp_0.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_1.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_2.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_3.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_4.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_5.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_6.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_7.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_8.translation', ref=1, lower=0, upper=10)
prob.model.add_design_var('system.components.comp_9.translation', ref=1, lower=0, upper=10)


prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 25
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-9


# Set the initial state
prob.setup()


# Configure the system
prob.set_val('system.components.comp_0.translation', [1, 8, 2])
prob.set_val('system.components.comp_1.translation', [5, 8, 2])
prob.set_val('system.components.comp_2.translation', [1, 7, 2])
prob.set_val('system.components.comp_3.translation', [8, 7, 2])
prob.set_val('system.components.comp_4.translation', [1, 5.5, 2])
prob.set_val('system.components.comp_5.translation', [8, 5.5, 2])
prob.set_val('system.components.comp_6.translation', [0.5, 3, 2])
prob.set_val('system.components.comp_7.translation', [7.5, 3, 2])
prob.set_val('system.components.comp_8.translation', [6, 1, 2])
prob.set_val('system.components.comp_9.translation', [2, 0, 2])

prob.set_val('system.interconnects.int_0.control_points', [[4.75, 8, 2]])
prob.set_val('system.interconnects.int_1.control_points', [[8, 8, 2]])
prob.set_val('system.interconnects.int_2.control_points', [[8, 6, 2]])
prob.set_val('system.interconnects.int_3.control_points', [[8, 5, 2]])
prob.set_val('system.interconnects.int_4.control_points', [[6.5, 2.75, 2]])
prob.set_val('system.interconnects.int_5.control_points', [[5, 1, 2]])
prob.set_val('system.interconnects.int_6.control_points', [[1.5, 1.5, 2]])
prob.set_val('system.interconnects.int_7.control_points', [[1, 5, 2]])
prob.set_val('system.interconnects.int_8.control_points', [[1, 6, 2]])
prob.set_val('system.interconnects.int_9.control_points', [[1, 8, 2]])



prob.run_model()


# Check the initial state

# print('Initial Objective:', prob.get_val('bbv.bounding_box_volume'))
# print('Initial Collision:', prob.get_val('collision_multiplexer.stacked_output'))


# Run the optimization
# from time import time_ns
# start = time_ns()
# prob.run_driver()
# end = time_ns()
# print('Time:', (end - start) / 1e9)


# # Check the final state
# plot_problem(prob)
# print('Final Objective:', prob.get_val('bbv.bounding_box_volume'))
# print('Final Collision:', prob.get_val('collision_multiplexer.stacked_output'))

plot_problem(prob, plot_bounding_box=True, plot_grid_points=False)
