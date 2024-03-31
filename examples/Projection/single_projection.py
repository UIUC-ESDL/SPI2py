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
from SPI2py.models.geometry.maximal_disjoint_ball_decomposition import mdbd
import pyvista as pv

# radiator_and_ion_exchanger = pv.Cube(bounds=(0, 2.850, 0, 0.830, 0, 0.830))
# radiator_and_ion_exchanger.save('radiator_and_ion_exchanger.stl')
# # Create the parts
# mdbd('components/', 'CAD_Files/radiator_and_ion_exchanger.stl','mdbds/radiator_and_ion_exchanger.xyzr',
#      num_spheres=100, min_radius=0.0001, meshgrid_increment=100, plot=True)


# Set the random seed for reproducibility
np.random.seed(0)

# Set the default data type
torch.set_default_dtype(torch.float64)

# Read the input file
input_file = read_input_file('input_single_projection.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Mesh Parameters
# bounds = (3, 11, 3, 11, 0, 3)
# bounds = (0, 8, 1, 8, 0, 3)
bounds = (0, 5, 0, 6, 0, 3)
n_elements_per_unit_length = 1.0  # 1.0  # 6.0

# System Parameters
n_components = 2
n_points = 100
n_points_per_object = [n_points for _ in range(n_components)]


# Initialize the groups
model.add_subsystem('system', System(input_dict=input_file, upper=7, lower=0))
model.add_subsystem('mesh', Mesh(bounds=bounds,
                                 n_elements_per_unit_length=n_elements_per_unit_length,
                                 mdbd_unit_cube_filepath='mdbd_unit_cube.xyzr',
                                 mdbd_unit_cube_min_radius=0.04))

model.add_subsystem('projections', Projections(n_comp_projections=n_components,
                                               n_int_projections=0))

model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_points_per_object, m=3))
model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_points_per_object, m=1))

model.add_subsystem('volume_fraction_constraint', VolumeFractionConstraint(n_projections=n_components))
model.add_subsystem('bbv', BoundingBoxVolume(n_points_per_object=n_points_per_object))


# TODO Promote?
model.connect('mesh.element_length', 'projections.projection_0.element_length')
model.connect('mesh.element_length', 'projections.projection_1.element_length')
model.connect('mesh.element_length', 'volume_fraction_constraint.element_length')


model.connect('mesh.centers', 'projections.projection_0.centers')
model.connect('mesh.centers', 'projections.projection_1.centers')

model.connect('mesh.sample_points', 'projections.projection_0.sample_points')
model.connect('mesh.sample_radii', 'projections.projection_0.sample_radii')

model.connect('mesh.sample_points', 'projections.projection_1.sample_points')
model.connect('mesh.sample_radii', 'projections.projection_1.sample_radii')

model.connect('system.components.comp_0.transformed_sphere_positions', 'projections.projection_0.sphere_positions')
model.connect('system.components.comp_0.transformed_sphere_radii', 'projections.projection_0.sphere_radii')

model.connect('system.components.comp_1.transformed_sphere_positions', 'projections.projection_1.sphere_positions')
model.connect('system.components.comp_1.transformed_sphere_radii', 'projections.projection_1.sphere_radii')

model.connect('projections.projection_0.pseudo_densities', 'volume_fraction_constraint.element_pseudo_densities_0')
model.connect('projections.projection_1.pseudo_densities', 'volume_fraction_constraint.element_pseudo_densities_1')

for i in range(n_components):
    model.connect(f'system.components.comp_{i}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
    model.connect(f'system.components.comp_{i}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')

model.connect('mux_all_sphere_positions.stacked_output', 'bbv.sphere_positions')
model.connect('mux_all_sphere_radii.stacked_output', 'bbv.sphere_radii')

# Define the objective and constraints
prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
prob.model.add_constraint('volume_fraction_constraint.volume_fraction_constraint', upper=0.01)

prob.model.add_design_var('system.components.comp_0.translation', ref=10, lower=0, upper=10)
# prob.model.add_design_var('rotation', ref=2*3.14159)


# Set the initial state
prob.setup()


# Configure the system
prob.set_val('system.components.comp_0.translation', [2, 2.5, 1.5])
prob.set_val('system.components.comp_0.rotation', [0, 0, 0.3])

prob.set_val('system.components.comp_1.translation', [2, 4.5, 1.5])
prob.set_val('system.components.comp_1.rotation', [0, 0, 0])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 5
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-12

prob.run_model()

print("Constraint Value: ", prob.get_val('volume_fraction_constraint.volume_fraction_constraint'))

# Run the optimization
# prob.run_driver()


# Debugging
element_index = [1, 2, 1]
pseudo_densities = prob.get_val('projections.projection_0.pseudo_densities')

print("Checking element: ", element_index)
print("Pseudo-Density: ", pseudo_densities[element_index[0], element_index[1], element_index[2]])
print("Max Pseudo-Density: ", pseudo_densities.max())
print("Constraint Value: ", prob.get_val('volume_fraction_constraint.volume_fraction_constraint'))

# Check the initial state
plot_problem(prob, plot_bounding_box=True)



# estimate_projection_error(prob,
#                           'system.components.comp_0.sphere_radii',
#                           'system.components.comp_0.translation',
#                           'projections.projection_0.volume',
#                           [2, 2.5, 1.5],
#                           10, 0.02)


print('Done')



# print('Number of elements:', prob.get_val('mesh.mesh_shape').size)






