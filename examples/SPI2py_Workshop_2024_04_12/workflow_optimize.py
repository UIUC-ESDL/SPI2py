"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""
import numpy as np
import openmdao.api as om
from time import time_ns

from SPI2py.API.system import System
from SPI2py.API.projection import Mesh, Projections, ProjectionAggregator
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer

from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file

# jax.config.update("jax_enable_x64", True)

# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Mesh Parameters
# bounds = (0, 7, 0, 7, 0, 2)
# n_elements_per_unit_length = 1.0
bounds = (0, 7, 0, 7, 0, 2)
n_elements_per_unit_length = 2.0

# System Parameters
n_components = 3
n_spheres = 200

m_interconnects = 3
m_spheres_per_segment = 10
m_segments = 2

n_projections = n_components + m_interconnects
n_points_per_object = [n_spheres for _ in range(n_components)] + [m_spheres_per_segment * m_segments for _ in range(m_interconnects)]


# Initialize the groups
model.add_subsystem('system', System(input_dict=input_file, upper=7, lower=0))
model.add_subsystem('mesh', Mesh(bounds=bounds,
                                 n_elements_per_unit_length=n_elements_per_unit_length))

model.add_subsystem('projections', Projections(n_comp_projections=n_components,
                                               n_int_projections=m_interconnects))

model.add_subsystem('aggregator', ProjectionAggregator(n_projections=n_projections))


model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_points_per_object, m=3))
model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_points_per_object, m=1))

# model.add_subsystem('collision', VolumeFractionCollision())
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
ref = bounds[1] * bounds[3] * bounds[5]  # Volume of the bounding box
prob.model.add_objective('bbv.bounding_box_volume', ref=ref)
prob.model.add_constraint('aggregator.max_pseudo_density', upper=1.1)

# prob.model.add_design_var('system.components.comp_0.translation', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)
prob.model.add_design_var('system.components.comp_1.translation', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)
prob.model.add_design_var('system.components.comp_2.translation', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)
prob.model.add_design_var('system.interconnects.int_0.control_points', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)
prob.model.add_design_var('system.interconnects.int_1.control_points', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)
prob.model.add_design_var('system.interconnects.int_2.control_points', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)


# Set the initial state
prob.setup()


# Configure the system
# prob.set_val('system.components.comp_0.translation', [1, 2.5, 0.5])
# prob.set_val('system.components.comp_1.translation', [5, 2, 0.5])
# prob.set_val('system.components.comp_2.translation', [2, 2, 0])
# prob.set_val('system.components.comp_2.rotation', [-np.pi/2, 0, 0])
# prob.set_val('system.interconnects.int_0.control_points', [[2.5, 2, 0.5]])
# prob.set_val('system.interconnects.int_1.control_points', [[5, 5, 0.5]])
# prob.set_val('system.interconnects.int_2.control_points', [[1, 4, 0.5]])

# Demo optimal
prob.set_val('system.components.comp_0.translation', [1, 1, 0.5])
prob.set_val('system.components.comp_1.translation', [3.5, 2, 0.5])
prob.set_val('system.components.comp_2.translation', [0.5, 1, 0])
prob.set_val('system.components.comp_2.rotation', [-np.pi/2, 0, 0])
prob.set_val('system.interconnects.int_0.control_points', [[1.6, 1.6, 0.5]])
prob.set_val('system.interconnects.int_1.control_points', [[2.35, 3.75, 0.5]])
prob.set_val('system.interconnects.int_2.control_points', [[0.65, 1.75, 0.5]])

# Setup the problem with complex-step
# prob.setup(force_alloc_complex=True)

# Set where to approximate derivatives
# prob.model.approx_totals(method='fd')
# prob.model.approx_totals(method='cs')



prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 10
prob.driver.options['optimizer'] = 'COBYLA'
# prob.driver.options['optimizer'] = 'trust-constr'

prob.run_model()

# print('Max Pseudo Density:', prob.get_val('aggregator.max_pseudo_density'))

# plot_problem(prob, plot_objects=True, plot_bounding_box=False, plot_grid_points=False, plot_projection=False)
# plot_problem(prob, plot_objects=False, plot_bounding_box=False, plot_grid_points=False, plot_projection=True)


# Run the optimization
# start = time_ns()
# prob.run_driver()
# end = time_ns()
# print('Elapsed Time: ', (end - start) / 1e9)

# Check the initial state
plot_problem(prob, plot_bounding_box=False, plot_grid_points=False, plot_projection=True)

print('Max Pseudo Density:', prob.get_val('aggregator.max_pseudo_density'))



print('Done')







