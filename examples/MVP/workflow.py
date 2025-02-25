"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""
import pyvista as pv
import openmdao.api as om
from SPI2py.API.system import System, Components, Interconnects, Component, Interconnect
from SPI2py.API.projection import Projections, ProjectionAggregator, ProjectComponent
from SPI2py.API.FEA import Mesh
from SPI2py.models.physics.distributed.mesh import generate_mesh_vec
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer, read_input_file
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_stl_file


# Read the input file
# input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model
system = System()
components = Components()
interconnects = Interconnects()
projections = Projections()

# Initialize the Mesh
x_bounds = (0, 10)
y_bounds = (0, 5)
z_bounds = (0, 10)
element_size = 1.0
kernel_steps_per_unit_length = 1
mesh = Mesh(x_bounds=x_bounds, y_bounds=y_bounds, z_bounds=z_bounds, element_size=element_size)

nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh_vec(-1, 3, -1, 3, -1, 4, element_size=element_size)
centers = centers.reshape(nx, ny, nz, 1, 3)

kernel_points, kernel_radii = create_uniform_kernel(kernel_steps_per_unit_length, mode='circumscription')
kernel_points = kernel_points.reshape(-1, 3)
kernel_radii = kernel_radii.reshape(-1, 1)


# Assemble the system
model.add_subsystem('system', system)
model.system.add_subsystem('components', components)
model.system.add_subsystem('interconnects', interconnects)
model.add_subsystem('mesh', mesh)
model.add_subsystem('projections', projections)

# Define the individual components
comp_1 = Component(description='Cross Head Pin', filepath='csvs/CrossHead_Pin_5k_300s.csv', n_spheres=50, ports=[[0.0, 0.415, 0.415], [2.850, 0.415, 0.415]], color='purple')
proj_1 = ProjectComponent(element_size=element_size, mesh_centers=centers, kernel_points=kernel_points, kernel_radii=kernel_radii)

model.system.components.add_subsystem('comp_1', comp_1)
model.projections.add_subsystem('proj_1', proj_1)

# Connect the components to the system
model.connect('system.components.comp_1.transformed_sphere_positions', 'projections.proj_1.sphere_positions')
model.connect('system.components.comp_1.transformed_sphere_radii', 'projections.proj_1.sphere_radii')

# model.connect('mesh.element_size', 'projections.proj_1.element_size')
# model.connect('mesh.mesh_centers', 'projections.proj_1.mesh_centers')



#                 interconnects.add_subsystem(f'int_{i}', interconnect)
#
#                 # Connect the interconnects to the components
#                 self.connect(f'components.comp_{component_1}.transformed_ports',
#                              f'interconnects.int_{i}.start_point',
#                              src_indices=om.slicer[port_1, :])
#                 self.connect(f'components.comp_{component_2}.transformed_ports',
#                              f'interconnects.int_{i}.end_point',
#                              src_indices=om.slicer[port_2, :])
#
#             self.add_subsystem('interconnects', interconnects)
#
#         # Create the system
#         system = System(n_projections=len(components_dict) + len(interconnects_dict), rho_min=1e-3)
#         self.add_subsystem('system', system)
#
#
#         # Connect the components to the system
#         i = 0
#         for j in range(len(components_dict)):
#             self.connect(f'components.comp_{j}.pseudo_densities',
#                           f'system.pseudo_densities_{i}')
#             i += 1
#
#         # Connect the interconnects to the system
#         for j in range(len(interconnects_dict)):
#             self.connect(f'interconnects.int_{j}.pseudo_densities',
#                           f'system.pseudo_densities_{i}')
#             i += 1

# model.add_subsystem('system', SpatialConfiguration(input_dict=input_file))
# model.add_subsystem('mesh', Mesh(bounds=bounds,
#                                  n_elements_per_unit_length=n_elements_per_unit_length))

# model.add_subsystem('projections', Projections(n_comp_projections=n_components,
#                                                n_int_projections=m_interconnects))
#
# model.add_subsystem('aggregator', ProjectionAggregator(n_projections=n_projections))
#
#
# model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_points_per_object, m=3))
# model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_points_per_object, m=1))
# model.add_subsystem('bbv', BoundingBoxVolume())
#
# # Connect the system to the projections
# i = 0
# for j in range(n_components):
#     model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'projections.projection_{i}.sphere_positions')
#     model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'projections.projection_{i}.sphere_radii')
#     model.connect(f'system.components.comp_{j}.volume', f'projections.projection_{i}.volume')
#     model.connect(f'projections.projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
#     i += 1
#
# for j in range(m_interconnects):
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', f'projections.projection_{i}.sphere_positions')
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_radii', f'projections.projection_{i}.sphere_radii')
#     # model.connect(f'system.interconnects.int_{j}.volume', f'projections.projection_{i}.volume')
#     model.connect(f'projections.projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
#     i += 1
#
# # Connect the mesh to the projections
# model.connect('mesh.element_length', 'aggregator.element_length')
# for i in range(n_projections):
#     model.connect('mesh.element_length', f'projections.projection_{i}.element_length')
#     model.connect('mesh.centers', f'projections.projection_{i}.centers')
#     model.connect('mesh.element_bounds', f'projections.projection_{i}.element_bounds')
#     model.connect('mesh.sample_points', f'projections.projection_{i}.element_sphere_positions')
#     model.connect('mesh.sample_radii', f'projections.projection_{i}.element_sphere_radii')
#
# # Connect the system to the bounding box
# i = 0
# for j in range(n_components):
#     model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
#     model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
#     i += 1
#
# for j in range(m_interconnects):
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
#     i += 1
#
# model.connect('mux_all_sphere_positions.stacked_output', 'bbv.sphere_positions')
# model.connect('mux_all_sphere_radii.stacked_output', 'bbv.sphere_radii')
#
# # Define the objective and constraints
# ref = bounds[1] * bounds[3] * bounds[5]  # Volume of the bounding box
# prob.model.add_objective('bbv.bounding_box_volume', ref=ref)
# prob.model.add_constraint('aggregator.max_pseudo_density', upper=1.1)
#
# # Define the design variables
# prob.model.add_design_var('system.components.comp_0.translation', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)
# prob.model.add_design_var('system.components.comp_1.translation', ref=5, lower=0, upper=10, indices=[0, 1], flat_indices=True)



# Set the initial state
prob.setup()


# Configure the system
# prob.set_val('system.components.comp_0.translation', [1.25, 8, 2])


# Set up the optimizer
# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['maxiter'] = 25
# prob.driver.options['optimizer'] = 'COBYLA'

# Run the model once
prob.run_model()

# Run the optimization
# prob.run_driver()


# Check the initial state
# print('Max Pseudo Density:', prob.get_val('aggregator.max_pseudo_density'))
# plot_problem(prob)

# mesh_centers = prob.get_val('mesh.mesh_centers')
# el_size = prob.get_val('mesh.element_size')
sphere_positions = prob.get_val('system.components.comp_1.transformed_sphere_positions')
sphere_radii = prob.get_val('system.components.comp_1.transformed_sphere_radii')
densities = prob.get_val('projections.proj_1.pseudo_densities')

# Plot the grid without the kernel
import numpy as np
sphere_positions = np.array(sphere_positions)
sphere_radii = np.array(sphere_radii)
plotter = pv.Plotter(shape=(1, 1), window_size=(1500, 500))
plot_grid(plotter, (0, 0), np.array(centers), element_size, densities=densities)
plot_spheres(plotter, (0, 0), sphere_positions, sphere_radii, 'purple', opacity=0.5)
plot_stl_file(plotter, (0, 0), 'models/CrossHead_Pin_scaled.stl', translation=(1, 0, 0), rotation=(0, 0, 0), opacity=0.25, color='purple')
plotter.show()

# data = prob.check_partials(includes='system.components.comp_1', step=1e-4,show_only_incorrect=True)
# data = prob.check_partials(includes='projections.proj_1')
# print(data['projections.proj_1']['pseudo_densities','sphere_positions'])
# print(data['system.components.comp_1']['transformed_sphere_positions','translation'])
print('Done')
