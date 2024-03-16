"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import openmdao.api as om
from SPI2py.API.system import System
from SPI2py.API.utilities import Multiplexer, MaxAggregator
from SPI2py.API.projection import Projection, Projections
from SPI2py.API.constraints import VolumeFractionConstraint
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer


# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Parameters
n_components = 1
n_points = 10
n_points_per_object = [n_points for _ in range(n_components)]




# Initialize the groups
model.add_subsystem('system', System(input_dict=input_file, upper=7, lower=0))
model.add_subsystem('projections', Projections(n_comp_projections=n_components, n_int_projections=0,min_xyz=-3, max_xyz=10, n_el_xyz=18))
model.add_subsystem('volume_fraction_constraint', VolumeFractionConstraint(n_projections=n_components, min_xyz=-3, max_xyz=10, n_el_xyz=18))

model.add_subsystem('mux_all_points', Multiplexer(n_i=n_points_per_object, m=3))
model.add_subsystem('bbv', BoundingBoxVolume(n_points_per_object=n_points_per_object))

model.connect('system.components.comp_0.transformed_points', 'projections.projection_0.points')
# model.connect('system.components.comp_1.transformed_points', 'projections.projection_1.points')
model.connect('projections.projection_0.element_pseudo_densities', 'volume_fraction_constraint.element_pseudo_densities_0')
# model.connect('projections.projection_1.element_pseudo_densities', 'volume_fraction_constraint.element_pseudo_densities_1')

# Add a Multiplexer for component sphere positions
i = 0
for j in range(n_components):
    model.connect(f'system.components.comp_{j}.transformed_points', f'mux_all_points.input_{i}')
    i += 1

model.connect('mux_all_points.stacked_output', 'bbv.points')

# Define the objective and constraints
prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
prob.model.add_constraint('volume_fraction_constraint.volume_fraction_constraint', upper=0.2)



# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['maxiter'] = 5
# prob.driver.options['optimizer'] = 'SLSQP'
# # prob.driver.options['tol'] = 1e-12


# Set the initial state
prob.setup()


# Configure the system
# prob.set_val('system.components.comp_0.translation', [2, 7, 0])

prob.set_val('system.components.comp_0.translation', [5, 5, 0])
prob.set_val('system.components.comp_0.rotation', [0, 0, 0])

# prob.set_val('system.components.comp_1.translation', [5.5, 7, 0])
# prob.set_val('system.components.comp_1.rotation', [0, 0, 0])

# # Collision
# prob.set_val('system.components.comp_1.translation', [5.5, 5, 0])
# prob.set_val('system.components.comp_1.rotation', [0, 0, 0])





prob.run_model()


# Check the initial state
plot_problem(prob)

# # Run the optimization
# prob.run_driver()
#
#
# # Check the final state
# plot_problem(prob)
#
#
# print('Constraint violation:', prob.get_val('volume_fraction_constraint.volume_fraction_constraint'))
#
# # Print positions
# print(prob.get_val('system.components.comp_0.translation'))
# print(prob.get_val('system.components.comp_1.translation'))