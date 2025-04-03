"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson

TODO
1. Why prob.check_partials(includes='system.components.comp_1')
updated end points to end points not correctly identity matrix?
Same for updated ports and radii
"""

# Standard Python libraries
from copy import copy
from time import time_ns
import pyvista as pv
import numpy as np
import jax
import jax.numpy as jnp
import openmdao.api as om

# Import SPI2py API
from SPI2py.API.system import System, Components, MDBDComponent, LinearSplineComponent, Interconnects, Interconnect
from SPI2py.API.projection import Projections, ProjectionAggregator, ProjectMDBDComponent, ProjectLinearSplineComponent, ProjectInterconnect
from SPI2py.API.FEA import ExplicitFEA
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer

# Import SPI2py supporting models
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_active_nodes, find_face_nodes
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_capsules, plot_stl_file, plot_AABB, plot_capsules2
from SPI2py.models.utilities.visualization import plot_temperature_distribution, plot_translation_sensitivities

# Set PyVista backend to use PyQt6
# from pyvistaqt import BackgroundPlotter
# import os
# os.environ["QT_API"] = "pyside6"  # Force use of PySide6

# Start the timer
t0 = time_ns()

# Set up the JAX backend
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

# Initialize the main problem elements/groups
prob = om.Problem()
model = prob.model

system = System()
components = Components()
interconnects = Interconnects()
projections = Projections()


model.add_subsystem('system', system)
model.system.add_subsystem('components', components)
model.system.add_subsystem('interconnects', interconnects)
model.add_subsystem('projections', projections)


# Initialize the Mesh
x_min, x_max = (-4, 4)
y_min, y_max = (-2, 3)
z_min, z_max = (-2, 4)
# x_min, x_max = (-5, 5)
# y_min, y_max = (-5, 5)
# z_min, z_max = (-5, 5)
# x_min, x_max = (-3, 3)
# y_min, y_max = (-3, 3)
# z_min, z_max = (-3, 3)

# element_size = 0.4
# element_size = 0.125
element_size = 0.25
# element_size = 0.5
# element_size = 1.0


nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
centers = centers.reshape(nx, ny, nz, 1, 3)



# Mult kernels cause error
kernel_steps_per_unit_length = 1
kernel_points, kernel_radii = create_uniform_kernel(kernel_steps_per_unit_length, mode='circumscription')
kernel_points = kernel_points.reshape(-1, 3)
kernel_radii = kernel_radii.reshape(-1, 1)


# Define the system elements
comp_1 = MDBDComponent(description='Cross Head Pin', filepath='csvs/CrossHead_Pin_5k_300s.csv', n_spheres=50, ports=[[0.0, 0.415, 0.415], [2.850, 0.415, 0.415]], color='blue')
# comp_2 = LinearSplineComponent(start_points=[[0, 0, 0], [0, 2, 0]], end_points=[[0, 2, 0], [0, 2, 1]], radii=[0.25, 0.5], ports=[[0, 0, 0]], color='red')
# comp_1 = LinearSplineComponent(start_points=[[0, 0, 0], [0, 2, 0]], end_points=[[0, 2, 0], [0, 2, 1]], radii=[0.5, 0.25], ports=[[0, 0, 0]], color='blue')
comp_2 = LinearSplineComponent(start_points=[[0, 0, 0], [0, 0, 0], [0, 2, 0]], end_points=[[0, 2, 0], [1, 0, 0], [1, 2, 0]], radii=[0.5, 0.5, 0.5], ports=[[1.75, 0, 0]], color='red')
int_1 = Interconnect(n_segments=3, radius=0.25, color='green')
model.system.components.add_subsystem('comp_1', comp_1)
model.system.components.add_subsystem('comp_2', comp_2)
model.system.interconnects.add_subsystem('int_1', int_1)


# Interconnect system elements
model.connect('system.components.comp_1.updated_ports', 'system.interconnects.int_1.start_point', src_indices=om.slicer[0, :])
# model.connect('system.components.comp_1.updated_ports', 'system.interconnects.int_1.start_point', src_indices=om.slicer[0, :])
model.connect('system.components.comp_2.updated_ports', 'system.interconnects.int_1.end_point', src_indices=om.slicer[0, :])


# Define the projections
proj_1 = ProjectMDBDComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
# proj_1 = ProjectLinearSplineComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_2 = ProjectLinearSplineComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_3 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
model.projections.add_subsystem('proj_1', proj_1)
model.projections.add_subsystem('proj_2', proj_2)
model.projections.add_subsystem('proj_3', proj_3)



# Connect the system elements to the projections
model.connect('system.components.comp_1.updated_sphere_positions', 'projections.proj_1.centers')
model.connect('system.components.comp_1.updated_sphere_radii', 'projections.proj_1.radii')
# model.connect('system.components.comp_1.updated_start_points', 'projections.proj_1.start_points')
# model.connect('system.components.comp_1.updated_end_points', 'projections.proj_1.end_points')
# model.connect('system.components.comp_1.updated_radii', 'projections.proj_1.radii')
model.connect('system.components.comp_2.updated_start_points', 'projections.proj_2.start_points')
model.connect('system.components.comp_2.updated_end_points', 'projections.proj_2.end_points')
model.connect('system.components.comp_2.updated_radii', 'projections.proj_2.radii')
model.connect('system.interconnects.int_1.updated_cyl_positions', 'projections.proj_3.control_points')
model.connect('system.interconnects.int_1.updated_cyl_radius', 'projections.proj_3.radius')


# Aggregate the spheres of each component
# TODO Update for interconnects
# mux_centers = Multiplexer(n_i=[2, 2, 2, 2], m=3)
mux_centers = Multiplexer(n_i=[939, 3, 3, 4], m=3)
prob.model.add_subsystem('mux_centers', mux_centers)
# mux_radii = Multiplexer(n_i=[2, 2, 2, 2], m=1)
mux_radii = Multiplexer(n_i=[939, 3, 3, 4], m=1)
prob.model.add_subsystem('mux_radii', mux_radii)

bbv = BoundingBoxVolume()
model.add_subsystem('bbv', bbv)

prob.model.connect('system.components.comp_1.updated_sphere_positions', 'mux_centers.input_0')
# prob.model.connect('system.components.comp_2.updated_sphere_positions', 'mux_centers.input_1')
# prob.model.connect('mux_centers.stacked_output', 'bbv.centers')

prob.model.connect('system.components.comp_1.updated_sphere_radii', 'mux_radii.input_0')
# prob.model.connect('system.components.comp_2.updated_sphere_radii', 'mux_radii.input_1')
# prob.model.connect('mux_radii.stacked_output', 'bbv.radii')

# prob.model.connect('system.components.comp_1.updated_start_points', 'mux_centers.input_0')
# prob.model.connect('system.components.comp_1.updated_end_points', 'mux_centers.input_1')
# prob.model.connect('system.components.comp_2.updated_start_points', 'mux_centers.input_2')
# prob.model.connect('system.components.comp_2.updated_end_points', 'mux_centers.input_3')
# prob.model.connect('mux_centers.stacked_output', 'bbv.centers')
prob.model.connect('system.components.comp_2.updated_start_points', 'mux_centers.input_1')
prob.model.connect('system.components.comp_2.updated_end_points', 'mux_centers.input_2')
prob.model.connect('system.interconnects.int_1.updated_cyl_positions', 'mux_centers.input_3')
prob.model.connect('mux_centers.stacked_output', 'bbv.centers')

# prob.model.connect('system.components.comp_1.updated_radii', 'mux_radii.input_0')
# prob.model.connect('system.components.comp_1.updated_radii', 'mux_radii.input_1')
# prob.model.connect('system.components.comp_2.updated_radii', 'mux_radii.input_2')
# prob.model.connect('system.components.comp_2.updated_radii', 'mux_radii.input_3')
# prob.model.connect('mux_radii.stacked_output', 'bbv.radii')
prob.model.connect('system.components.comp_2.updated_radii', 'mux_radii.input_1')
prob.model.connect('system.components.comp_2.updated_radii', 'mux_radii.input_2')
prob.model.connect('system.interconnects.int_1.updated_cyl_radius', 'mux_radii.input_3')
prob.model.connect('mux_radii.stacked_output', 'bbv.radii')


# Aggregate the pseudo-densities
# projection_aggregator = ProjectionAggregator(n_projections=2, rho_min=3e-3)
projection_aggregator = ProjectionAggregator(n_projections=3, rho_min=1e-3)
model.projections.add_subsystem('aggregator', projection_aggregator)

model.connect('projections.proj_1.penalized_densities', 'projections.aggregator.densities_0')
model.connect('projections.proj_2.penalized_densities', 'projections.aggregator.densities_1')
model.connect('projections.proj_3.penalized_densities', 'projections.aggregator.densities_2')

model.connect('projections.proj_1.penalized_heat_loads', 'projections.aggregator.heat_loads_0')
model.connect('projections.proj_2.penalized_heat_loads', 'projections.aggregator.heat_loads_1')
model.connect('projections.proj_3.penalized_heat_loads', 'projections.aggregator.heat_loads_2')

# FEA
dirichlet_nodes = find_face_nodes(nodes, jnp.array([0.0, 0.0, -1.0]))
dirichlet_T = 300.0 * jnp.ones(len(dirichlet_nodes))
robin_nodes = find_face_nodes(nodes, jnp.array([0.0, 0.0, 1.0]))
FEA = ExplicitFEA(nodes=nodes,
                  elements=elements,
                  el_size=element_size,
                  el_centers=centers,
                  dirichlet_nodes=dirichlet_nodes,
                  dirichlet_values=dirichlet_T,
                  robin_nodes=robin_nodes,
                  robin_h=10.0,
                  robin_T_inf=200.0,
                  fea_solution_scheme='partition')

model.add_subsystem('FEA', FEA)
model.connect('projections.aggregator.aggregated_densities', 'FEA.density')
model.connect('projections.aggregator.aggregated_heat_loads', 'FEA.heat_loads')


# Define the design variables
# prob.model.add_design_var('system.components.comp_1.translation', ref=0.25, lower=-5, upper=5)
# prob.model.add_design_var('system.components.comp_1.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=-5, upper=5)
prob.model.add_design_var('system.components.comp_2.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
# prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=-3, upper=3, indices=[0], flat_indices=True)
# prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=0, upper=3, indices=[1], flat_indices=True)
# prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=0, upper=3, indices=[2], flat_indices=True)
prob.model.add_design_var('system.interconnects.int_1.control_points', ref=0.25, lower=-5, upper=5)

# Define the objective and constraints
prob.model.add_objective('bbv.volume', ref=1)
prob.model.add_constraint('projections.aggregator.max_density', upper=1.1)


# Set the initial state
prob.setup()


# # Configure the system
# TODO Rotation?
prob.set_val('system.components.comp_1.translation', [2, 0, 0])
# prob.set_val('system.components.comp_1.rotation', [0, 0, 0])
# prob.set_val('system.components.comp_2.translation', [1.5, 0, 0])
prob.set_val('system.components.comp_2.translation', [-3, 0, 0])
prob.set_val('system.components.comp_2.rotation', [np.pi/3, 0, 0])
prob.set_val('system.interconnects.int_1.control_points', [[1.5, 0, -1], [-1, 0, -1]])

prob.set_val('projections.proj_1.heat_load', 0.1)


# Set up the optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 10


# Run the model once
t1 = time_ns()
prob.run_model()
t2 = time_ns()

print(f"Run time: {(t2 - t1) / 1e9} seconds")


# Check the initial state
print("BBV Before:", prob.get_val('bbv.volume'))
print("Max Density:", prob.get_val('projections.aggregator.max_density'))
comp_1_translation_before = copy(prob.get_val('system.components.comp_1.translation'))
comp_1_rotation_before = copy(prob.get_val('system.components.comp_1.rotation'))
# comp_1_start_points_before = copy(prob.get_val('system.components.comp_1.updated_start_points'))
# comp_1_end_points_before = copy(prob.get_val('system.components.comp_1.updated_end_points'))
# comp_1_radii_before = copy(prob.get_val('system.components.comp_1.updated_radii'))
comp_1_spheres_before = copy(prob.get_val('system.components.comp_1.updated_sphere_positions'))
comp_1_radii_before = copy(prob.get_val('system.components.comp_1.updated_sphere_radii'))
comp_2_translation_before = copy(prob.get_val('system.components.comp_2.translation'))
comp_2_rotation_before = copy(prob.get_val('system.components.comp_2.rotation'))
comp_2_start_points_before = copy(prob.get_val('system.components.comp_2.updated_start_points'))
comp_2_end_points_before = copy(prob.get_val('system.components.comp_2.updated_end_points'))
comp_2_radii_before = copy(prob.get_val('system.components.comp_2.updated_radii'))
int_1_points_before = copy(prob.get_val('system.interconnects.int_1.updated_cyl_positions'))
int_1_control_points_before = copy(prob.get_val('system.interconnects.int_1.control_points'))
# sphere_positions_before = copy(prob.get_val('mux_centers.stacked_output'))
# sphere_radii_before = copy(prob.get_val('mux_radii.stacked_output'))
bounds_before = copy(prob.get_val('bbv.bounds'))
densities_before = copy(prob.get_val('projections.aggregator.aggregated_densities'))
T_before = copy(prob.get_val('FEA.temperature'))
densities_before = copy(prob.get_val('projections.aggregator.aggregated_densities'))
heat_load_nodes_before = find_active_nodes(densities_before, elements, threshold=1e-3)

tot_before = prob.compute_totals(of=['bbv.volume'], wrt=['system.components.comp_1.translation','system.components.comp_2.translation','system.interconnects.int_1.control_points'])
tot_before_comp_1 = copy(tot_before[('bbv.volume', 'system.components.comp_1.translation')][0])
tot_before_comp_2 = copy(tot_before[('bbv.volume', 'system.components.comp_2.translation')][0])
tot_before_int_1 = copy(tot_before[('bbv.volume', 'system.interconnects.int_1.control_points')][0])





# # Run the optimization
# t3 = time_ns()
# prob.run_driver()
# t4 = time_ns()
# print(f"Optimization time: {(t4 - t3) / 1e9} seconds")






# # Check the final state
print("BBV After:", prob.get_val('bbv.volume'))
# print("BBV Bounds:", prob.get_val('bbv.bounds'))
print("Max Density:", prob.get_val('projections.aggregator.max_density'))
comp_1_translation_after = prob.get_val('system.components.comp_1.translation')
comp_1_rotation_after = prob.get_val('system.components.comp_1.rotation')
# comp_1_start_points_after = prob.get_val('system.components.comp_1.updated_start_points')
# comp_1_end_points_after = prob.get_val('system.components.comp_1.updated_end_points')
# comp_1_radii_after = prob.get_val('system.components.comp_1.updated_radii')
comp_1_spheres_after = prob.get_val('system.components.comp_1.updated_sphere_positions')
comp_1_radii_after = prob.get_val('system.components.comp_1.updated_sphere_radii')
comp_2_translation_after = prob.get_val('system.components.comp_2.translation')
comp_2_rotation_after = prob.get_val('system.components.comp_2.rotation')
comp_2_start_points_after = prob.get_val('system.components.comp_2.updated_start_points')
comp_2_end_points_after = prob.get_val('system.components.comp_2.updated_end_points')
comp_2_radii_after = prob.get_val('system.components.comp_2.updated_radii')
int_1_points_after = prob.get_val('system.interconnects.int_1.updated_cyl_positions')
int_1_control_points_after = prob.get_val('system.interconnects.int_1.control_points')
# sphere_positions_after = prob.get_val('mux_centers.stacked_output')
# sphere_radii_after = prob.get_val('mux_radii.stacked_output')
bounds_after = prob.get_val('bbv.bounds')
densities_after = prob.get_val('projections.aggregator.aggregated_densities')
T_after = prob.get_val('FEA.temperature')
densities_after = prob.get_val('projections.aggregator.aggregated_densities')
heat_load_nodes_after = find_active_nodes(densities_after, elements, threshold=1e-3)

tot_after = prob.compute_totals(of=['bbv.volume'], wrt=['system.components.comp_1.translation','system.components.comp_2.translation','system.interconnects.int_1.control_points'])
tot_after_comp_1 = copy(tot_after[('bbv.volume', 'system.components.comp_1.translation')][0])
tot_after_comp_2 = copy(tot_after[('bbv.volume', 'system.components.comp_2.translation')][0])
tot_after_int_1 = copy(tot_after[('bbv.volume', 'system.interconnects.int_1.control_points')][0])


# Convert JAX arrays to NumPy arrays
centers = np.array(centers)
comp_1_translation_before = tuple(np.array(comp_1_translation_before).tolist())
comp_1_rotation_before = tuple(np.array(comp_1_rotation_before).tolist())
# comp_1_start_points_before = np.array(comp_1_start_points_before)
# comp_1_end_points_before = np.array(comp_1_end_points_before)
# comp_1_radii_before = np.array(comp_1_radii_before)
comp_1_spheres_before = np.array(comp_1_spheres_before)
comp_1_radii_before = np.array(comp_1_radii_before)
comp_1_spheres_before = np.array(comp_1_spheres_before)
comp_2_translation_before = tuple(np.array(comp_2_translation_before).tolist())
comp_2_rotation_before = tuple(np.array(comp_2_rotation_before).tolist())
comp_2_start_points_before = np.array(comp_2_start_points_before)
comp_2_end_points_before = np.array(comp_2_end_points_before)
comp_2_radii_before = np.array(comp_2_radii_before)
comp_1_translation_after = tuple(np.array(comp_1_translation_after).tolist())
comp_1_rotation_after = tuple(np.array(comp_1_rotation_after).tolist())
# comp_1_start_points_after = np.array(comp_1_start_points_after)
# comp_1_end_points_after = np.array(comp_1_end_points_after)
# comp_1_radii_after = np.array(comp_1_radii_after)
comp_1_spheres_after = np.array(comp_1_spheres_after)
comp_1_radii_after = np.array(comp_1_radii_after)
comp_2_translation_after = tuple(np.array(comp_2_translation_after).tolist())
comp_2_rotation_after = tuple(np.array(comp_2_rotation_after).tolist())
comp_2_start_points_after = np.array(comp_2_start_points_after)
comp_2_end_points_after = np.array(comp_2_end_points_after)
comp_2_radii_after = np.array(comp_2_radii_after)
bounds_before = np.array(bounds_before)
bounds_after = np.array(bounds_after)
densities_before = np.array(densities_before)
densities_after = np.array(densities_after)
# sphere_positions_before = np.array(sphere_positions_before)
# sphere_radii_before = np.array(sphere_radii_before)
# sphere_positions_after = np.array(sphere_positions_after)
# sphere_radii_after = np.array(sphere_radii_after)
int_1_points_before = np.array(int_1_points_before)
int_1_points_after = np.array(int_1_points_after)
int_1_control_points_before = np.array(int_1_control_points_before)
int_1_control_points_after = np.array(int_1_control_points_after)
T_before = np.array(T_before)
T_after = np.array(T_after)
densities_before = np.array(densities_before)
densities_after = np.array(densities_after)
heat_load_nodes_before = np.array(heat_load_nodes_before)
heat_load_nodes_after = np.array(heat_load_nodes_after)

print('Min Temp:', np.min(T_after))
print('Max Temp:', np.max(T_after))
print('Mean Temp:', np.mean(T_after))

# Plot the results
t5 = time_ns()
plotter = pv.Plotter(shape=(2, 3), window_size=(1500, 500))
# plotter = BackgroundPlotter(shape=(2, 3), window_size=(1500, 500))

# BEFORE

# Geometry
plot_grid(plotter, (0, 0), centers, element_size, densities=None)
plot_stl_file(plotter, (0, 0), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_before, rotation=comp_1_rotation_before, opacity=0.25, color='blue')
plot_spheres(plotter, (0, 0), comp_1_spheres_before, comp_1_radii_before, 'blue', opacity=0.5)
# plot_capsules2(plotter, (0, 0), comp_1_start_points_before, comp_1_end_points_before, comp_1_radii_before, color='blue', opacity=0.5)
plot_capsules2(plotter, (0, 0), comp_2_start_points_before, comp_2_end_points_before, comp_2_radii_before, color='red', opacity=0.5)
plot_capsules(plotter, (0, 0), int_1_points_before, 0.25, color='green', opacity=0.5)


# Projection
plot_grid(plotter, (0, 1), centers, element_size, densities=None)
plot_grid(plotter, (0, 1), centers, element_size, densities=densities_before)

plot_AABB(plotter, (0, 1), bounds_before, color='blue', opacity=0.15)
plot_translation_sensitivities(plotter, (0, 1), comp_1_spheres_before[0], tot_before_comp_1, color='red', factor=3.0)
plot_translation_sensitivities(plotter, (0, 1), comp_2_start_points_before[2], tot_before_comp_2, color='red', factor=3.0)
plot_translation_sensitivities(plotter, (0, 1), int_1_control_points_before, tot_before_int_1, color='red', factor=3.0)

# FEA
plot_grid(plotter, (0, 2), centers, element_size, densities=None)
plot_stl_file(plotter, (0, 2), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_before, rotation=comp_1_rotation_before, opacity=1.0, color='black')
plot_capsules2(plotter, (0, 2), comp_2_start_points_before, comp_2_end_points_before, comp_2_radii_before, color='black', opacity=1.0)
plot_capsules(plotter, (0, 2), int_1_points_before, 0.25, color='black', opacity=1.0)
plot_temperature_distribution(plotter,
                              (0, 2),
                              np.array(nodes),
                              T_before,
                              heat_load_nodes=heat_load_nodes_before,
                              robin_nodes=robin_nodes,
                              dirichlet_nodes=dirichlet_nodes,
                              dims=(nx + 1, ny + 1, nz + 1),
                              cmap='jet')

# plot_temperature_distribution(plotter,
#                               (0, 2),
#                               np.array(nodes),
#                               T_before,
#                               dims=(nx + 1, ny + 1, nz + 1),
#                               cmap='jet')



# # Define grid dimensions.
# nx, ny, nz = 50, 40, 30
# dims = (nx, ny, nz)
#
# # Create a structured grid in [0,1] for each axis.
# x = np.linspace(0, 1, nx)
# y = np.linspace(0, 1, ny)
# z = np.linspace(0, 1, nz)
# # Generate a structured grid with 'ij' indexing.
# X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
#
# # Create nodal positions: shape (nx*ny*nz, 3).
# nodes = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
#
# # Create a smooth temperature field.
# # For example, a smooth function that varies with sine and cosine.
# T = 150 + 50 * np.sin(np.pi * X.ravel()) * np.cos(np.pi * Y.ravel()) * np.sin(np.pi * Z.ravel())
# # Alternatively, you could try a simpler function:
# # T = 300 * (X.ravel() + Y.ravel() + Z.ravel()) / 3.0
#
# plot_temperature_distribution(plotter,
#                               (0, 2),
#                               nodes,
#                               T,
#                               dims=dims)



# AFTER

# Geometry
plot_grid(plotter, (1, 0), centers, element_size, densities=None)
plot_stl_file(plotter, (1, 0), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_after, rotation=comp_1_rotation_after, opacity=0.25, color='blue')
plot_spheres(plotter, (1, 0), comp_1_spheres_after, comp_1_radii_after, 'blue', opacity=0.5)
# plot_capsules2(plotter, (1, 0), comp_1_start_points_after, comp_1_end_points_after, comp_1_radii_after, color='blue', opacity=0.5)
plot_capsules2(plotter, (1, 0), comp_2_start_points_after, comp_2_end_points_after, comp_2_radii_after, color='red', opacity=0.5)
plot_capsules(plotter, (1, 0), int_1_points_after, 0.25, color='green', opacity=0.5)

# Projection
plot_grid(plotter, (1, 1), centers, element_size, densities=None)
plot_grid(plotter, (1, 1), centers, element_size, densities=densities_after)

plot_AABB(plotter, (1, 1), bounds_after, color='blue', opacity=0.15)
plot_translation_sensitivities(plotter, (1, 1), comp_1_spheres_after[0], tot_after_comp_1, color='red', factor=3.0)
plot_translation_sensitivities(plotter, (1, 1), comp_2_start_points_after[2], tot_after_comp_2, color='red', factor=3.0)
plot_translation_sensitivities(plotter, (1, 1), int_1_control_points_after, tot_after_int_1, color='red', factor=3.0)

# FEA
plot_grid(plotter, (1, 2), centers, element_size, densities=None)
plot_stl_file(plotter, (1, 2), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_after, rotation=comp_1_rotation_after, opacity=1.0, color='black')
plot_capsules2(plotter, (1, 2), comp_2_start_points_after, comp_2_end_points_after, comp_2_radii_after, color='black', opacity=1.0)
plot_capsules(plotter, (1, 2), int_1_points_after, 0.25, color='black', opacity=1.0)
plot_temperature_distribution(plotter,
                              (1, 2),
                              np.array(nodes),
                              T_after,
                              heat_load_nodes=heat_load_nodes_after,
                              robin_nodes=robin_nodes,
                              dirichlet_nodes=dirichlet_nodes,
                              dims=(nx + 1, ny + 1, nz + 1),
                              cmap='jet')



# Plot the geometries before optimization
# plotter.subplot(0, 0)
# plotter.add_title("Before Optimization")
# plot_stl_file(plotter, (0, 0), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_before, rotation=comp_1_rotation_before, opacity=0.25, color='purple')
# plot_stl_file(plotter, (0, 0), 'models/Bot_Eye_scaled.stl', translation=comp_2_translation_before, rotation=comp_2_rotation_before, opacity=0.25, color='blue')
# plot_spheres(plotter, (0, 0), sphere_positions_before, sphere_radii_before, 'purple', opacity=0.5)
# plot_AABB(plotter, (0, 0), bounds_before, color='blue')

# Plot the geometries after optimization
# plotter.subplot(0, 1)
# plotter.add_title("After Optimization")

# plot_stl_file(plotter, (0, 1), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_after, rotation=comp_1_rotation_after, opacity=0.25, color='purple')
# plot_stl_file(plotter, (0, 1), 'models/Bot_Eye_scaled.stl', translation=comp_2_translation_after, rotation=comp_2_rotation_after, opacity=0.25, color='blue')
# plot_spheres(plotter, (0, 1), sphere_positions_after, sphere_radii_after, 'purple', opacity=0.5)
# plot_AABB(plotter, (0, 1), bounds_after, color='blue')




plotter.link_views()
# plotter.show_axes()
plotter.show()



t6 = time_ns()
print(f"Plot time: {(t6 - t5) / 1e9} seconds")
print(f"Total time: {(t6 - t0) / 1e9} seconds")
print('Done')


# Code for debugging
# prob.model.approx_totals(method='exact')
# prob.model.approx_totals(method='fd')
# totals_c_approx = copy(prob.compute_totals(of=['projections.aggregator.max_density'], wrt=['system.components.comp_2.translation']))
# pf = prob.check_partials(includes='FEA')
# tot = prob.compute_totals(of=['FEA.max_temperature'], wrt=['system.components.comp_2.translation'])
# tot = tot[('FEA.max_temperature', 'system.components.comp_2.translation')][0]
# print(tot)
# d_inputs = {'density': jnp.ones_like(densities_combined, dtype=jnp.float64), 'heat_loads': jnp.ones_like(densities_combined, dtype=jnp.float64)}
# d_outputs = {'temperature': jnp.ones_like(T, dtype=jnp.float64), 'max_temperature': jnp.array([1], dtype=jnp.float64)}
# jvp_vals = prob.model.FEA.compute_jacvec_product(prob.model.FEA._inputs, d_inputs, d_outputs, mode='fwd')
# vjp_vals = prob.model.FEA.compute_jacvec_product(prob.model.FEA._inputs, d_inputs, d_outputs, mode='rev')
# print('JVP:', jvp_vals)
# print('VJP:', vjp_vals)