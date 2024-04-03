"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import openmdao.api as om
from time import time_ns

from SPI2py.API.system import System
from SPI2py.API.projection import Mesh, Projections
from SPI2py.API.constraints import VolumeFractionCollision
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer, estimate_partial_derivative_memory, estimate_projection_error


# Read the input file
input_file = read_input_file('input_single_projection.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Mesh Parameters
bounds = (0, 7, 0, 7, 0, 3)
n_elements_per_unit_length = 1.0

# System Parameters
n_components = 2
n_points = 25
n_points_per_object = [n_points for _ in range(n_components)]
n_interconnects = 0
n_projections = n_components + n_interconnects

# TODO Move true volume calculation back to objects; interconnects must consider overlapping spheres

# Initialize the groups
model.add_subsystem('system', System(input_dict=input_file, upper=7, lower=0))
model.add_subsystem('mesh', Mesh(bounds=bounds,
                                 n_elements_per_unit_length=n_elements_per_unit_length,
                                 mesh_kernel_min_radius=0.1))

model.add_subsystem('projections', Projections(n_comp_projections=n_components,
                                               n_int_projections=0))

model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_points_per_object, m=3))
model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_points_per_object, m=1))

# model.add_subsystem('collision', VolumeFractionCollision())
model.add_subsystem('bbv', BoundingBoxVolume())

# Connect the system to the projections
i = 0
for j in range(n_components):
    model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'projections.projection_{i}.sphere_positions')
    model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'projections.projection_{i}.sphere_radii')
    i += 1
# for j in range(n_interconnects):
#     pass
#     i += 1


# Connect the mesh to the projections
for i in range(n_projections):
    model.connect('mesh.element_length', f'projections.projection_{i}.element_length')
    model.connect('mesh.centers', f'projections.projection_{i}.centers')
    model.connect('mesh.sample_points', f'projections.projection_{i}.sample_points')
    model.connect('mesh.sample_radii', f'projections.projection_{i}.sample_radii')



# Connect the system to the bounding box
i = 0
for j in range(n_components):
    model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
    model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
    i += 1
# for j in range(n_interconnects):
#     pass
#     i += 1

model.connect('mux_all_sphere_positions.stacked_output', 'bbv.sphere_positions')
model.connect('mux_all_sphere_radii.stacked_output', 'bbv.sphere_radii')


# Define the objective and constraints
prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
# prob.model.add_constraint('collision.volume_fraction', lower=0, upper=0.01)

prob.model.add_design_var('system.components.comp_0.translation', ref=10, lower=0, upper=10)
# prob.model.add_design_var('rotation', ref=2*3.14159)


# Set the initial state
prob.setup()


# Configure the system
prob.set_val('system.components.comp_0.translation', [2, 2.5, 1.5])
prob.set_val('system.components.comp_0.rotation', [0, 0, 0.3])

prob.set_val('system.components.comp_1.translation', [2, 5.5, 1.5])
prob.set_val('system.components.comp_1.rotation', [0.1, 0.1, 0])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 10
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-12

prob.run_model()

# print("Constraint Value: ", prob.get_val('collision.volume_fraction'))

t1 = time_ns()

# Run the optimization
# prob.run_driver()

t2 = time_ns()
print('Runtime: ', (t2 - t1) / 1e9, 's')


# Debugging
element_index = [1, 2, 1]
pseudo_densities = prob.get_val('projections.projection_0.pseudo_densities')

print("Checking element: ", element_index)
print("Pseudo-Density: ", pseudo_densities[element_index[0], element_index[1], element_index[2]])
print("Max Pseudo-Density: ", pseudo_densities.max())
# print("Constraint Value: ", prob.get_val('collision.volume_fraction'))

# Check the initial state
plot_problem(prob, plot_bounding_box=True, plot_grid_points=False)



# estimate_projection_error(prob,
#                           'system.components.comp_0.sphere_radii',
#                           'system.components.comp_0.translation',
#                           'projections.projection_0.volume',
#                           [2, 2.5, 1.5],
#                           10, 0.02)


print('Done')







