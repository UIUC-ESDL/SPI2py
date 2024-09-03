"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import openmdao.api as om
from time import time_ns

from SPI2py.API.system import SpatialConfiguration
from SPI2py.API.projection import Mesh, Projections, ProjectionAggregator
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer


# Read the input file
input_file = read_input_file('input_single_projection.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Mesh Parameters
bounds = (0, 3, 0, 2, 0, 2)
n_elements_per_unit_length = 1.0

# System Parameters
n_components = 1
n_spheres = 50

m_interconnects = 0
m_spheres_per_segment = 10
m_segments = 2


n_projections = n_components + m_interconnects
n_points_per_object = [n_spheres for _ in range(n_components)] + [m_spheres_per_segment * m_segments for _ in range(m_interconnects)]

# Initialize the groups
model.add_subsystem('system', SpatialConfiguration(input_dict=input_file, upper=7, lower=0))
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
prob.model.add_objective('bbv.bounding_box_volume', ref=1, ref0=0)
prob.model.add_constraint('aggregator.max_pseudo_density', upper=1.0)

# prob.model.add_design_var('system.components.comp_0.translation', ref=10, lower=0, upper=10)
# prob.model.add_design_var('system.components.comp_1.translation', ref=5, lower=0, upper=10)
# prob.model.add_design_var('system.interconnects.int_0.control_points', ref=5, lower=0, upper=10)
# prob.model.add_design_var('rotation', ref=2*3.14159)


# Set the initial state
prob.setup()


# Configure the system
# prob.set_val('system.components.comp_0.translation', [2, 7, 2])
# prob.set_val('system.components.comp_1.translation', [5.5, 7, 2])
# prob.set_val('system.interconnects.int_0.control_points', [[3.75, 7, 2]])

# prob.set_val('system.components.comp_0.translation', [2, 7, 2])
# prob.set_val('system.components.comp_1.translation', [7, 4, 2])
# prob.set_val('system.interconnects.int_0.control_points', [[5, 5, 2]])

prob.set_val('system.components.comp_0.translation', [1.2, 1.2, 1])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 10
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['tol'] = 1e-12

prob.run_model()

# print("Constraint Value: ", prob.get_val('collision.volume_fraction'))

# plot_problem(prob, plot_bounding_box=True, plot_grid_points=False)


# Run the optimization
# prob.run_driver()



# Debugging
# element_index = [1, 2, 1]
pseudo_densities = prob.get_val('projections.projection_0.pseudo_densities')

# print("Checking element: ", element_index)
# print("Pseudo-Density: ", pseudo_densities[element_index[0], element_index[1], element_index[2]])
print("Max Pseudo-Density: ", pseudo_densities.max())
# print("Constraint Value: ", prob.get_val('collision.volume_fraction'))


print('Kernel Volume Fraction:', prob.get_val('mesh.kernel_volume_fraction'))
print('Volume Estimation Error (Component 0):', prob.get_val('projections.projection_0.volume_estimation_error'))
print('Max Pseudo Density:', prob.get_val('aggregator.max_pseudo_density'))


# Check the initial state
plot_problem(prob, plot_bounding_box=False, plot_grid_points=True)



print('Done')







