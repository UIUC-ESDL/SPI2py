"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import numpy as np
import openmdao.api as om
import torch

from SPI2py.API.system import System
from SPI2py.API.utilities import Multiplexer, MaxAggregator
from SPI2py.API.projection import Projection, Projections, Mesh
from SPI2py.API.constraints import VolumeFractionConstraint
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer, estimate_partial_derivative_memory, estimate_projection_error

# Set the random seed for reproducibility
np.random.seed(0)

# Set the default data type
torch.set_default_dtype(torch.float64)

# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Mesh Parameters
# bounds = (3, 11, 3, 11, 0, 3)
# bounds = (0, 8, 1, 8, 0, 3)
bounds = (0, 8, 0, 10, 0, 3) # TODO FIX nonzero bounds
n_elements_per_unit_length = 3.0  # 6.0

# System Parameters
n_components = 2
n_points = 50
n_points_per_object = [n_points for _ in range(n_components)]


# Initialize the groups
model.add_subsystem('system', System(input_dict=input_file, upper=7, lower=0))
model.add_subsystem('mesh', Mesh(bounds=bounds, n_elements_per_unit_length=n_elements_per_unit_length))
model.add_subsystem('projections', Projections(n_comp_projections=n_components, n_int_projections=0))
model.add_subsystem('volume_fraction_constraint', VolumeFractionConstraint(n_projections=n_components))

model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_points_per_object, m=3))
model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_points_per_object, m=1))
model.add_subsystem('bbv', BoundingBoxVolume(n_points_per_object=n_points_per_object))

# TODO Promote?
model.connect('mesh.element_length', 'projections.projection_0.element_length')
model.connect('mesh.element_length', 'projections.projection_1.element_length')
model.connect('mesh.x_centers', 'projections.projection_0.x_centers')
model.connect('mesh.y_centers', 'projections.projection_0.y_centers')
model.connect('mesh.z_centers', 'projections.projection_0.z_centers')
model.connect('mesh.x_centers', 'projections.projection_1.x_centers')
model.connect('mesh.y_centers', 'projections.projection_1.y_centers')
model.connect('mesh.z_centers', 'projections.projection_1.z_centers')


model.connect('mesh.element_length', 'volume_fraction_constraint.element_length')


model.connect('system.components.comp_0.transformed_sphere_positions', 'projections.projection_0.sphere_positions')
model.connect('system.components.comp_1.transformed_sphere_positions', 'projections.projection_1.sphere_positions')
model.connect('system.components.comp_0.transformed_sphere_radii', 'projections.projection_0.sphere_radii')
model.connect('system.components.comp_1.transformed_sphere_radii', 'projections.projection_1.sphere_radii')

model.connect('projections.projection_0.element_pseudo_densities', 'volume_fraction_constraint.element_pseudo_densities_0')
model.connect('projections.projection_1.element_pseudo_densities', 'volume_fraction_constraint.element_pseudo_densities_1')

# Add a Multiplexer for component sphere positions

for i in range(n_components):
    model.connect(f'system.components.comp_{i}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
    model.connect(f'system.components.comp_{i}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')

model.connect('mux_all_sphere_positions.stacked_output', 'bbv.sphere_positions')
model.connect('mux_all_sphere_radii.stacked_output', 'bbv.sphere_radii')

# Define the objective and constraints
prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
prob.model.add_constraint('volume_fraction_constraint.volume_fraction_constraint', upper=0.01)



prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 25
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-12


# Set the initial state
prob.setup()


# Configure the system
prob.set_val('system.components.comp_0.translation', [5, 5, 1.5])
prob.set_val('system.components.comp_0.rotation', [0, 0, 0])

prob.set_val('system.components.comp_1.translation', [5.5, 7, 1.5])
prob.set_val('system.components.comp_1.rotation', [0, 0, 0])

# # Collision
# prob.set_val('system.components.comp_1.translation', [5.5, 5, 1.5])
# prob.set_val('system.components.comp_1.rotation', [0, 0, 0])

prob.run_model()


estimate_projection_error(prob,
                          'system.components.comp_0.sphere_radii',
                          'system.components.comp_0.translation',
                          'projections.projection_0.volume',
                          [5, 5, 1.5],
                          5, 0.15)


# Check the initial state
plot_problem(prob)

# print('Number of elements:', prob.get_val('mesh.mesh_shape').size)

# Run the optimization
# prob.run_driver()


# Check the final state
# plot_problem(prob)


# print('Constraint violation:', prob.get_val('volume_fraction_constraint.volume_fraction_constraint'))

# # Print positions
# # print(prob.get_val('system.components.comp_0.translation'))
# # print(prob.get_val('system.components.comp_1.translation'))





