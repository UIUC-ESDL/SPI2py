"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# Standard Python libraries
import numpy as np
from copy import copy
import pyvista as pv
from time import time_ns

import jax
import jax.numpy as jnp
import openmdao.api as om

# SPI2py libraries
from SPI2py.API.system import System, Components, LinearSplineComponent, Interconnects, Component, Interconnect
from SPI2py.API.projection import Projections, ProjectionAggregator, ProjectMDBDComponent, ProjectInterconnect, \
    ProjectLinearSplineComponent
from SPI2py.API.FEA import Mesh, ExplicitFEA
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer

from SPI2py.models.physics.distributed.mesh import generate_mesh_vec
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_capsules, plot_stl_file, plot_AABB, plot_capsules2
from SPI2py.models.utilities.visualization import plot_temperature_distribution
from SPI2py.models.physics.distributed.mesh import generate_mesh_vec, find_active_nodes, find_face_nodes

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
# interconnects = Interconnects()
projections = Projections()


model.add_subsystem('system', system)
model.system.add_subsystem('components', components)
# model.system.add_subsystem('interconnects', interconnects)
model.add_subsystem('projections', projections)


# Initialize the Mesh
# x_min, x_max = (-2, 2)
# y_min, y_max = (0, 2.5)
# z_min, z_max = (0, 2.5)
# x_min, x_max = (-5, 5)
# y_min, y_max = (-5, 5)
# z_min, z_max = (-5, 5)
x_min, x_max = (-3, 3)
y_min, y_max = (-3, 3)
z_min, z_max = (-3, 3)

# element_size = 0.0675
# element_size = 0.125
# element_size = 0.25
element_size = 0.5
# element_size = 1.0


nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh_vec(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
centers = centers.reshape(nx, ny, nz, 1, 3)



# Mult kernels cause error
kernel_steps_per_unit_length = 1
kernel_points, kernel_radii = create_uniform_kernel(kernel_steps_per_unit_length, mode='circumscription')
kernel_points = kernel_points.reshape(-1, 3)
kernel_radii = kernel_radii.reshape(-1, 1)


# Define the system elements
# comp_1 = Component(description='Cross Head Pin', filepath='csvs/CrossHead_Pin_5k_300s.csv', n_spheres=50, ports=[[0.0, 0.415, 0.415], [2.850, 0.415, 0.415]], color='blue')
# comp_2 = LinearSplineComponent(start_points=[[0, 0, 0], [0, 2, 0]], end_points=[[0, 2, 0], [0, 2, 1]], radii=[0.25, 0.5], ports=[[0, 0, 0]], color='red')
comp_1 = LinearSplineComponent(start_points=[[0, 0, 0], [0, 2, 0]], end_points=[[0, 2, 0], [0, 2, 1]], radii=[0.5, 0.25], ports=[[0, 0, 0]], color='blue')
comp_2 = LinearSplineComponent(start_points=[[0, 0, 0], [-1, 0, 0]], end_points=[[0, 2, 0], [1, 0, 0]], radii=[0.5, 0.75], ports=[[0, 0, 0]], color='red')
# int_1 = Interconnect(n_segments=3, radius=0.25, color='green')
model.system.components.add_subsystem('comp_1', comp_1)
model.system.components.add_subsystem('comp_2', comp_2)
# model.system.interconnects.add_subsystem('int_1', int_1)


# Interconnect system elements
# model.connect('system.components.comp_1.transformed_ports', 'system.interconnects.int_1.start_point', src_indices=om.slicer[0, :])
# model.connect('system.components.comp_2.updated_ports', 'system.interconnects.int_1.end_point', src_indices=om.slicer[0, :])


# Define the projections
proj_1 = ProjectLinearSplineComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_2 = ProjectLinearSplineComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
# proj_3 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
model.projections.add_subsystem('proj_1', proj_1)
model.projections.add_subsystem('proj_2', proj_2)
# model.projections.add_subsystem('proj_3', proj_3)



# Connect the system elements to the projections
# model.connect('system.components.comp_1.transformed_sphere_positions', 'projections.proj_1.centers')
# model.connect('system.components.comp_1.transformed_sphere_radii', 'projections.proj_1.radii')
model.connect('system.components.comp_1.updated_start_points', 'projections.proj_1.start_points')
model.connect('system.components.comp_1.updated_end_points', 'projections.proj_1.end_points')
model.connect('system.components.comp_1.updated_radii', 'projections.proj_1.radii')
model.connect('system.components.comp_2.updated_start_points', 'projections.proj_2.start_points')
model.connect('system.components.comp_2.updated_end_points', 'projections.proj_2.end_points')
model.connect('system.components.comp_2.updated_radii', 'projections.proj_2.radii')
# model.connect('system.interconnects.int_1.transformed_cyl_positions', 'projections.proj_3.control_points')
# model.connect('system.interconnects.int_1.transformed_cyl_radius', 'projections.proj_3.radius')


# Aggregate the spheres of each component
mux_centers = Multiplexer(n_i=[2, 2, 2, 2], m=3)
prob.model.add_subsystem('mux_centers', mux_centers)
mux_radii = Multiplexer(n_i=[2, 2, 2, 2], m=1)
prob.model.add_subsystem('mux_radii', mux_radii)

bbv = BoundingBoxVolume()
model.add_subsystem('bbv', bbv)

# prob.model.connect('system.components.comp_1.transformed_sphere_positions', 'mux_centers.input_0')
# prob.model.connect('system.components.comp_2.transformed_sphere_positions', 'mux_centers.input_1')
# prob.model.connect('mux_centers.stacked_output', 'bbv.centers')

# prob.model.connect('system.components.comp_1.transformed_sphere_radii', 'mux_radii.input_0')
# prob.model.connect('system.components.comp_2.transformed_sphere_radii', 'mux_radii.input_1')
# prob.model.connect('mux_radii.stacked_output', 'bbv.radii')

prob.model.connect('system.components.comp_1.updated_start_points', 'mux_centers.input_0')
prob.model.connect('system.components.comp_1.updated_end_points', 'mux_centers.input_1')
prob.model.connect('system.components.comp_2.updated_start_points', 'mux_centers.input_2')
prob.model.connect('system.components.comp_2.updated_end_points', 'mux_centers.input_3')
prob.model.connect('mux_centers.stacked_output', 'bbv.centers')

prob.model.connect('system.components.comp_1.updated_radii', 'mux_radii.input_0')
prob.model.connect('system.components.comp_1.updated_radii', 'mux_radii.input_1')
prob.model.connect('system.components.comp_2.updated_radii', 'mux_radii.input_2')
prob.model.connect('system.components.comp_2.updated_radii', 'mux_radii.input_3')
prob.model.connect('mux_radii.stacked_output', 'bbv.radii')


# Aggregate the pseudo-densities
# projection_aggregator = ProjectionAggregator(n_projections=2, rho_min=3e-3)
projection_aggregator = ProjectionAggregator(n_projections=2, rho_min=1e-3)
model.projections.add_subsystem('aggregator', projection_aggregator)

model.connect('projections.proj_1.penalized_densities', 'projections.aggregator.densities_0')
model.connect('projections.proj_2.penalized_densities', 'projections.aggregator.densities_1')
# model.connect('projections.proj_3.penalized_densities', 'projections.aggregator.densities_2')

model.connect('projections.proj_1.penalized_heat_loads', 'projections.aggregator.heat_loads_0')
model.connect('projections.proj_2.penalized_heat_loads', 'projections.aggregator.heat_loads_1')
# model.connect('projections.proj_3.penalized_heat_loads', 'projections.aggregator.heat_loads_2')

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
                  robin_h=1.0, #10.0,
                  robin_T_inf=200.0,
                  fea_solution_scheme='penalty')

model.add_subsystem('FEA', FEA)
model.connect('projections.aggregator.aggregated_densities', 'FEA.density')
model.connect('projections.aggregator.aggregated_heat_loads', 'FEA.heat_loads')


# Define the design variables
# prob.model.add_design_var('system.components.comp_1.translation', ref=0.25, lower=-3, upper=3)
# prob.model.add_design_var('system.components.comp_1.rotation', ref=0.5, lower=0, upper=1)
prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=-5, upper=5)
prob.model.add_design_var('system.components.comp_2.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
# prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=-3, upper=3, indices=[0], flat_indices=True)
# prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=0, upper=3, indices=[1], flat_indices=True)
# prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=0, upper=3, indices=[2], flat_indices=True)

# Define the objective and constraints
prob.model.add_objective('bbv.volume', ref=1)
prob.model.add_constraint('projections.aggregator.max_density', upper=1.1)


# Set the initial state
prob.setup()


# # Configure the system
# TODO Rotation?
# prob.set_val('system.components.comp_1.translation', [0, 0, 0])
# prob.set_val('system.components.comp_1.rotation', [0, 0, 0])
# prob.set_val('system.components.comp_2.translation', [1.5, 0, 0])
prob.set_val('system.components.comp_2.translation', [3, 0, 0])
prob.set_val('system.components.comp_2.rotation', [np.pi/3, 0, 0])
# prob.set_val('system.interconnects.int_1.control_points', [[1.75, 1, 0], [1, 1, 0]])


# Set up the optimizer
prob.driver = om.ScipyOptimizeDriver()
# prob.driver = om.pyOptSparseDriver(optimizer='mma')
# prob.driver.options['maxiter'] = 30
prob.driver.options['maxiter'] = 10
# prob.driver.options['optimizer'] = 'trust-constr'
# prob.driver.options['optimizer'] = 'COBYLA'
# prob.driver.options['optimizer'] = 'SLSQP'

# Run the model once
t1 = time_ns()
prob.run_model()
t2 = time_ns()

print(f"Run time: {(t2 - t1) / 1e9} seconds")


# Check the initial state
print("BBV Before:", prob.get_val('bbv.volume'))
# print("BBV Bounds:", prob.get_val('bbv.bounds'))
print("Max Density:", prob.get_val('projections.aggregator.max_density'))
comp_1_translation_before = copy(prob.get_val('system.components.comp_1.translation'))
comp_1_rotation_before = copy(prob.get_val('system.components.comp_1.rotation'))
comp_1_start_points_before = copy(prob.get_val('system.components.comp_1.updated_start_points'))
comp_1_end_points_before = copy(prob.get_val('system.components.comp_1.updated_end_points'))
comp_1_radii_before = copy(prob.get_val('system.components.comp_1.updated_radii'))
comp_2_translation_before = copy(prob.get_val('system.components.comp_2.translation'))
comp_2_rotation_before = copy(prob.get_val('system.components.comp_2.rotation'))
comp_2_start_points_before = copy(prob.get_val('system.components.comp_2.updated_start_points'))
comp_2_end_points_before = copy(prob.get_val('system.components.comp_2.updated_end_points'))
comp_2_radii_before = copy(prob.get_val('system.components.comp_2.updated_radii'))
# int_1_points_before = copy(prob.get_val('system.interconnects.int_1.transformed_cyl_positions'))
# sphere_positions_before = copy(prob.get_val('mux_centers.stacked_output'))
# sphere_radii_before = copy(prob.get_val('mux_radii.stacked_output'))
# bounds_before = copy(prob.get_val('bbv.bounds'))
densities_before = copy(prob.get_val('projections.aggregator.aggregated_densities'))


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
comp_1_start_points_after = prob.get_val('system.components.comp_1.updated_start_points')
comp_1_end_points_after = prob.get_val('system.components.comp_1.updated_end_points')
comp_1_radii_after = prob.get_val('system.components.comp_1.updated_radii')
comp_2_translation_after = prob.get_val('system.components.comp_2.translation')
comp_2_rotation_after = prob.get_val('system.components.comp_2.rotation')
comp_2_start_points_after = prob.get_val('system.components.comp_2.updated_start_points')
comp_2_end_points_after = prob.get_val('system.components.comp_2.updated_end_points')
comp_2_radii_after = prob.get_val('system.components.comp_2.updated_radii')
# # int_1_points_after = prob.get_val('system.interconnects.int_1.transformed_cyl_positions')
# sphere_positions_after = prob.get_val('mux_centers.stacked_output')
# sphere_radii_after = prob.get_val('mux_radii.stacked_output')
# bounds_after = prob.get_val('bbv.bounds')
densities_after = prob.get_val('projections.aggregator.aggregated_densities')


# Convert JAX arrays to NumPy arrays
centers = np.array(centers)
comp_1_translation_before = tuple(np.array(comp_1_translation_before).tolist())
comp_1_rotation_before = tuple(np.array(comp_1_rotation_before).tolist())
comp_1_start_points_before = np.array(comp_1_start_points_before)
comp_1_end_points_before = np.array(comp_1_end_points_before)
comp_1_radii_before = np.array(comp_1_radii_before)
comp_2_translation_before = tuple(np.array(comp_2_translation_before).tolist())
comp_2_rotation_before = tuple(np.array(comp_2_rotation_before).tolist())
comp_2_start_points_before = np.array(comp_2_start_points_before)
comp_2_end_points_before = np.array(comp_2_end_points_before)
comp_2_radii_before = np.array(comp_2_radii_before)
comp_1_translation_after = tuple(np.array(comp_1_translation_after).tolist())
comp_1_rotation_after = tuple(np.array(comp_1_rotation_after).tolist())
comp_1_start_points_after = np.array(comp_1_start_points_after)
comp_1_end_points_after = np.array(comp_1_end_points_after)
comp_1_radii_after = np.array(comp_1_radii_after)
comp_2_translation_after = tuple(np.array(comp_2_translation_after).tolist())
comp_2_rotation_after = tuple(np.array(comp_2_rotation_after).tolist())
comp_2_start_points_after = np.array(comp_2_start_points_after)
comp_2_end_points_after = np.array(comp_2_end_points_after)
comp_2_radii_after = np.array(comp_2_radii_after)
# bounds_before = np.array(bounds_before)
# bounds_after = np.array(bounds_after)
densities_before = np.array(densities_before)
densities_after = np.array(densities_after)
# sphere_positions_before = np.array(sphere_positions_before)
# sphere_radii_before = np.array(sphere_radii_before)
# sphere_positions_after = np.array(sphere_positions_after)
# sphere_radii_after = np.array(sphere_radii_after)
# int_1_points_before = np.array(int_1_points_before)
# int_1_points_after = np.array(int_1_points_after)

T = prob.get_val('FEA.temperature')
T_np = np.array(T)

print('Min Temp:', np.min(T_np))
print('Max Temp:', np.max(T_np))
print('Mean Temp:', np.mean(T_np))

# Plot the results
t5 = time_ns()
plotter = pv.Plotter(shape=(2, 2), window_size=(1500, 500))



densities_combined = prob.get_val('projections.aggregator.aggregated_densities')
heat_load_nodes = find_active_nodes(densities_combined, elements, threshold=1e-3)

plot_temperature_distribution(plotter,
                              (0, 0),
                              np.array(nodes),
                              T_np,
                              heat_load_nodes=np.array(heat_load_nodes),
                              robin_nodes=robin_nodes,
                              dirichlet_nodes=dirichlet_nodes,
                              dims=(nx + 1, ny + 1, nz + 1),
                              cmap='jet')

plot_grid(plotter, (0, 0), centers, element_size, densities=densities_before)


# Plot the geometries before optimization
plotter.subplot(0, 0)
plotter.add_title("Before Optimization")
plot_grid(plotter, (0, 0), centers, element_size, densities=None)
# plot_stl_file(plotter, (0, 0), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_before, rotation=comp_1_rotation_before, opacity=0.25, color='purple')
# plot_stl_file(plotter, (0, 0), 'models/Bot_Eye_scaled.stl', translation=comp_2_translation_before, rotation=comp_2_rotation_before, opacity=0.25, color='blue')
plot_capsules2(plotter, (0, 0), comp_1_start_points_before, comp_1_end_points_before, comp_1_radii_before, color='blue', opacity=0.5)
plot_capsules2(plotter, (0, 0), comp_2_start_points_before, comp_2_end_points_before, comp_2_radii_before, color='red', opacity=0.5)
# plot_spheres(plotter, (0, 0), sphere_positions_before, sphere_radii_before, 'purple', opacity=0.5)
# plot_capsules(plotter, (0, 0), int_1_points_before, 0.25, color='green', opacity=0.5)
# plot_AABB(plotter, (0, 0), bounds_before, color='blue')

# Plot the pseudo-densities before optimization
plot_grid(plotter, (1, 0), centers, element_size, densities=None)
plot_grid(plotter, (1, 0), centers, element_size, densities=densities_before)

# Plot the geometries after optimization
plotter.subplot(0, 1)
plotter.add_title("After Optimization")
plot_grid(plotter, (0, 1), centers, element_size, densities=None)
plot_capsules2(plotter, (0, 1), comp_1_start_points_after, comp_1_end_points_after, comp_1_radii_after, color='blue', opacity=0.5)
plot_capsules2(plotter, (0, 1), comp_2_start_points_after, comp_2_end_points_after, comp_2_radii_after, color='red', opacity=0.5)
# plot_stl_file(plotter, (0, 1), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_after, rotation=comp_1_rotation_after, opacity=0.25, color='purple')
# plot_stl_file(plotter, (0, 1), 'models/Bot_Eye_scaled.stl', translation=comp_2_translation_after, rotation=comp_2_rotation_after, opacity=0.25, color='blue')
# plot_spheres(plotter, (0, 1), sphere_positions_after, sphere_radii_after, 'purple', opacity=0.5)
# plot_capsules(plotter, (0, 1), int_1_points_after, 0.25, color='green', opacity=0.5)
# plot_AABB(plotter, (0, 1), bounds_after, color='blue')

# Plot the pseudo-densities after optimization
plot_grid(plotter, (1, 1), centers, element_size, densities=densities_after)


# plot the origin (0,0,0)
plotter.subplot(*(0, 0))
plotter.add_mesh(pv.Sphere(radius=0.25), color='red', show_edges=True)

plotter.subplot(*(0, 1))
plotter.add_mesh(pv.Sphere(radius=0.25), color='red', show_edges=True)

plotter.link_views()
plotter.show_axes()
plotter.show()



t6 = time_ns()
print(f"Plot time: {(t6 - t5) / 1e9} seconds")

# Print the total time
print(f"Total time: {(t6 - t0) / 1e9} seconds")




print('Done')


# Code for debugging
# prob.model.approx_totals(method='exact')
# prob.model.approx_totals(method='fd')
# totals_c_approx = copy(prob.compute_totals(of=['projections.aggregator.max_density'], wrt=['system.components.comp_2.translation']))
# pf = prob.check_partials(includes='FEA')
# tot = prob.compute_totals(of=['FEA.max_temperature'], wrt=['system.components.comp_2.translation'])
# tot = tot[('FEA.max_temperature', 'system.components.comp_2.translation')][0]
# d_inputs = {'density': jnp.ones_like(densities_combined), 'heat_loads': jnp.ones_like(densities_combined)}
# d_outputs = {'temperature': jnp.ones_like(T), 'max_temperature': jnp.array(1)}
# jvp_vals = prob.model.FEA.compute_jacvec_prod(prob.model.FEA._inputs, d_inputs, d_outputs, mode='fwd')
# vjp_vals = prob.model.FEA.compute_jacvec_prod(prob.model.FEA._inputs, d_inputs, d_outputs, mode='rev')