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
from SPI2py.API.system import System, Components, Interconnects, MDBDComponent, Interconnect
from SPI2py.API.projection import Projections, ProjectionAggregator, ProjectMDBDComponent, ProjectInterconnect
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer

# Import SPI2py supporting models
from SPI2py.models.mechanics.homogenous_transformation import transform_points
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_active_nodes, find_face_nodes
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_capsules, plot_stl_file, plot_AABB, plot_capsules2
from SPI2py.models.utilities.visualization import plot_temperature_distribution, plot_translation_sensitivities


# Start the timer
t0 = time_ns()

# Set up the JAX backend
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)

# Initialize the main problem elements/groups
prob          = om.Problem()
model         = prob.model
system        = System()
components    = Components()
interconnects = Interconnects()
projections   = Projections()

model.add_subsystem('system', system)
model.system.add_subsystem('components', components)
model.system.add_subsystem('interconnects', interconnects)
model.add_subsystem('projections', projections)


# Initialize the Mesh
x_min, x_max = (0, 7)
y_min, y_max = (0, 7)
z_min, z_max = (0, 2)
element_size = 0.5
nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
centers = centers.reshape(nx, ny, nz, 1, 3)


# Define kernel...
# Multi kernels cause error
kernel_steps_per_unit_length = 1
kernel_points, kernel_radii = create_uniform_kernel(kernel_steps_per_unit_length, mode='circumscription')
kernel_points = kernel_points.reshape(-1, 3)
kernel_radii = kernel_radii.reshape(-1, 1)



# %%

# Part 1
# >=4.0e-2 for up to 122 points
# >=3.0e-2 for up to 238 points
# >=2.0e-2 for up to 623 points
# S_c = 4.0e-2
S_c = 4.0e-2

# Define the system elements
comp_1 = MDBDComponent(filepath='csvs/Bot_Eye_5k_300s.csv',port_positions=[[0, 0.75, 0], [0.75, 0, 0]], minimum_radius=S_c) # Blue
comp_2 = MDBDComponent(filepath='csvs/CogDrivenGear_5k_300s.csv',port_positions=[[-1.75, 0, 0], [0, 1.75, 0]], minimum_radius=S_c) # Orange
comp_3 = MDBDComponent(filepath='csvs/CrossHead_Pin_5k_300s.csv',port_positions=[[1.25, 0.5, 1.0], [0.15, 0.5, 1.0]], minimum_radius=S_c) # Red
model.system.components.add_subsystem('comp_1', comp_1)
model.system.components.add_subsystem('comp_2', comp_2)
model.system.components.add_subsystem('comp_3', comp_3)
int_1 = Interconnect(radius=0.35, n_segments=2)
# int_2 = Interconnect(radius=0.75, n_segments=2)
# int_3 = Interconnect(radius=0.75, n_segments=2)
model.system.interconnects.add_subsystem('int_1', int_1)
# model.system.interconnects.add_subsystem('int_2', int_2)
# model.system.interconnects.add_subsystem('int_3', int_3)

# Connect the Interconnects to the components
model.connect('system.components.comp_1.ports', 'system.interconnects.int_1.start_points')
# model.connect('system.components.comp_2.ports', 'system.interconnects.int


# # Define the projections
proj_c1 = ProjectMDBDComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_c2 = ProjectMDBDComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_c3 = ProjectMDBDComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_i1 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
# proj_i2 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
# proj_i3 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
model.projections.add_subsystem('proj_c1', proj_c1)
model.projections.add_subsystem('proj_c2', proj_c2)
model.projections.add_subsystem('proj_c3', proj_c3)
model.projections.add_subsystem('proj_i1', proj_i1)
# model.projections.add_subsystem('proj_i2', proj_i2)
# model.projections.add_subsystem('proj_i3', proj_i3)



# Connect the system elements to the projections
model.connect('system.components.comp_1.updated_sphere_positions', 'projections.proj_c1.centers')
model.connect('system.components.comp_1.updated_sphere_radii', 'projections.proj_c1.radii')
model.connect('system.components.comp_2.updated_sphere_positions', 'projections.proj_c2.centers')
model.connect('system.components.comp_2.updated_sphere_radii', 'projections.proj_c2.radii')
model.connect('system.components.comp_3.updated_sphere_positions', 'projections.proj_c3.centers')
model.connect('system.components.comp_3.updated_sphere_radii', 'projections.proj_c3.radii')
model.connect('system.interconnects.int_1.updated_cyl_positions', 'projections.proj_i1.control_points')
model.connect('system.interconnects.int_1.updated_cyl_radius', 'projections.proj_i1.radius')
# model.connect('system.interconnects.int_2.control_points', 'projections.proj_i2.control_points')
# model.connect('system.interconnects.int_2.radii', 'projections.proj_i2.radii')
# model.connect('system.interconnects.int_3.control_points', 'projections.proj_i3.control_points')
# model.connect('system.interconnects.int_3.radii', 'projections.proj_i3.radii')


# Aggregate the spheres of each component
mux_centers = Multiplexer(n_i=[122, 1119, 195, 3, 3, 3], m=3)
mux_radii = Multiplexer(n_i=[122, 1119, 195, 3, 3, 3], m=1)
prob.model.add_subsystem('mux_centers', mux_centers)
prob.model.add_subsystem('mux_radii', mux_radii)
prob.model.connect('system.components.comp_1.updated_sphere_positions', 'mux_centers.input_0')
prob.model.connect('system.components.comp_2.updated_sphere_positions', 'mux_centers.input_1')
prob.model.connect('system.components.comp_3.updated_sphere_positions', 'mux_centers.input_2')
prob.model.connect('system.components.comp_1.updated_sphere_radii', 'mux_radii.input_0')
prob.model.connect('system.components.comp_2.updated_sphere_radii', 'mux_radii.input_1')
prob.model.connect('system.components.comp_3.updated_sphere_radii', 'mux_radii.input_2')
prob.model.connect('system.interconnects.int_1.updated_cyl_positions', 'mux_centers.input_3')
# prob.model.connect('system.interconnects.int_2.control_points', 'mux_centers.input_4')
# prob.model.connect('system.interconnects.int_3.control_points', 'mux_centers.input_5')
prob.model.connect('system.interconnects.int_1.updated_cyl_radius', 'mux_radii.input_3')
# prob.model.connect('system.interconnects.int_2.radii', 'mux_radii.input_4')
# prob.model.connect('system.interconnects.int_3.radii', 'mux_radii.input_5')



# Define and connect the objective
bbv = BoundingBoxVolume()
model.add_subsystem('bbv', bbv)
prob.model.connect('mux_centers.stacked_output', 'bbv.centers')
prob.model.connect('mux_radii.stacked_output', 'bbv.radii')


# Aggregate the pseudo-densities
rho_min = 1e-2
projection_aggregator = ProjectionAggregator(n_projections=4, rho_min=rho_min)
model.projections.add_subsystem('aggregator', projection_aggregator)
model.connect('projections.proj_c1.penalized_densities', 'projections.aggregator.densities_0')
model.connect('projections.proj_c2.penalized_densities', 'projections.aggregator.densities_1')
model.connect('projections.proj_c3.penalized_densities', 'projections.aggregator.densities_2')
model.connect('projections.proj_c1.penalized_heat_loads', 'projections.aggregator.heat_loads_0')
model.connect('projections.proj_c2.penalized_heat_loads', 'projections.aggregator.heat_loads_1')
model.connect('projections.proj_c3.penalized_heat_loads', 'projections.aggregator.heat_loads_2')
model.connect('projections.proj_i1.penalized_densities', 'projections.aggregator.densities_3')
# model.connect('projections.proj_i2.penalized_densities', 'projections.aggregator.densities_5')
# model.connect('projections.proj_i3.penalized_densities', 'projections.aggregator.densities_6')
model.connect('projections.proj_i1.penalized_heat_loads', 'projections.aggregator.heat_loads_3')
# model.connect('projections.proj_i2.penalized_heat_loads', 'projections.aggregator.heat_loads_5')
# model.connect('projections.proj_i3.penalized_heat_loads', 'projections.aggregator.heat_loads_6')



# # Define the design variables
# # prob.model.add_design_var('system.components.comp_1.translation', ref=0.25, lower=-5, upper=5)
# # prob.model.add_design_var('system.components.comp_1.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
# prob.model.add_design_var('system.components.comp_2.translation', ref=0.25, lower=-5, upper=5)
# # prob.model.add_design_var('system.components.comp_2.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
#
# # Define the objective and constraints
# prob.model.add_objective('bbv.volume', ref=1)
# prob.model.add_constraint('projections.aggregator.max_density', upper=1.1)


# Set the initial state
prob.setup()


# Configure the system
# prob.set_val('system.components.comp_1.translation', [0.625, 0.625, 0.125])
# prob.set_val('system.components.comp_1.rotation', [0, 0, 0])
# prob.set_val('system.components.comp_2.translation', [0.5, 4, 1.5])
# prob.set_val('system.components.comp_2.rotation', [np.pi/2, 0, np.pi/2])
# prob.set_val('system.components.comp_3.translation', [0, 0.5, 1.75])
# prob.set_val('system.components.comp_3.rotation', [0, 0, 0])

prob.set_val('system.components.comp_1.translation', [1, 1, 1])  # Blue
prob.set_val('system.components.comp_2.translation', [5, 2, 0])  # Orange
prob.set_val('system.components.comp_3.translation', [0.5, 4, 1])  # Red
prob.set_val('system.components.comp_3.rotation', [-np.pi/2, 0, 0])
prob.set_val('system.interconnects.int_1.control_points', [[1.6, 1.6, 0.5]])
# prob.set_val('system.interconnects.int_2.control_points', [[2.35, 3.75, 0.5]])
# prob.set_val('system.interconnects.int_3.control_points', [[0.65, 1.75, 0.5]])

#
# # Set up the optimizer
# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['maxiter'] = 10


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
comp_1_start_points_before = copy(prob.get_val('system.components.comp_1.updated_sphere_positions'))
comp_1_radii_before = copy(prob.get_val('system.components.comp_1.updated_sphere_radii'))
comp_2_translation_before = copy(prob.get_val('system.components.comp_2.translation'))
comp_2_rotation_before = copy(prob.get_val('system.components.comp_2.rotation'))
comp_2_start_points_before = copy(prob.get_val('system.components.comp_2.updated_sphere_positions'))
comp_2_radii_before = copy(prob.get_val('system.components.comp_2.updated_sphere_radii'))
comp_3_translation_before = copy(prob.get_val('system.components.comp_3.translation'))
comp_3_rotation_before = copy(prob.get_val('system.components.comp_3.rotation'))
comp_3_start_points_before = copy(prob.get_val('system.components.comp_3.updated_sphere_positions'))
comp_3_radii_before = copy(prob.get_val('system.components.comp_3.updated_sphere_radii'))

int_1_points_before = prob.get_val('system.interconnects.int_1.updated_cyl_positions')
int_1_radius_before = prob.get_val('system.interconnects.int_1.updated_cyl_radius')


bounds_before = copy(prob.get_val('bbv.bounds'))
densities_before = copy(prob.get_val('projections.aggregator.aggregated_densities'))

# For ...
tot_before = prob.compute_totals(of=['bbv.volume'], wrt=['system.components.comp_1.translation','system.components.comp_2.translation','system.components.comp_3.translation'])
tot_before_comp_1 = copy(tot_before[('bbv.volume', 'system.components.comp_1.translation')][0])
tot_before_comp_2 = copy(tot_before[('bbv.volume', 'system.components.comp_2.translation')][0])
tot_before_comp_3 = copy(tot_before[('bbv.volume', 'system.components.comp_3.translation')][0])


# # Run the optimization
# t3 = time_ns()
# prob.run_driver()
# t4 = time_ns()
# print(f"Optimization time: {(t4 - t3) / 1e9} seconds")


# # Check the final state
print("BBV After:", prob.get_val('bbv.volume'))
print("Max Density:", prob.get_val('projections.aggregator.max_density'))
comp_1_translation_after = prob.get_val('system.components.comp_1.translation')
comp_1_rotation_after = prob.get_val('system.components.comp_1.rotation')
comp_1_start_points_after = prob.get_val('system.components.comp_1.updated_sphere_positions')
comp_1_radii_after = prob.get_val('system.components.comp_1.updated_sphere_radii')
comp_2_translation_after = prob.get_val('system.components.comp_2.translation')
comp_2_rotation_after = prob.get_val('system.components.comp_2.rotation')
comp_2_start_points_after = prob.get_val('system.components.comp_2.updated_sphere_positions')
comp_2_radii_after = prob.get_val('system.components.comp_2.updated_sphere_radii')



bounds_after = prob.get_val('bbv.bounds')
densities_after = prob.get_val('projections.aggregator.aggregated_densities')

tot_after = prob.compute_totals(of=['bbv.volume'], wrt=['system.components.comp_1.translation','system.components.comp_2.translation', 'system.components.comp_3.translation'])
tot_after_comp_1 = copy(tot_after[('bbv.volume', 'system.components.comp_1.translation')][0])
tot_after_comp_2 = copy(tot_after[('bbv.volume', 'system.components.comp_2.translation')][0])



#%% Plot the results

# Create multiple subplots
plotter = pv.Plotter(shape=(2, 2), window_size=(1500, 500), lighting='light kit')



#%% Plot the System Before Optimization

# Subplot 1: Geometry and Components
plotter.subplot(0, 0)
plotter.add_title("Geometry Before")

# The problem domain
plot_grid(plotter, (0, 0), centers, element_size, densities=None, min_opacity=0.0)

# Components, MDBD Representation
plot_spheres(plotter, (0, 0), comp_1_start_points_before, comp_1_radii_before, 'blue', opacity=0.5)
plot_spheres(plotter, (0, 0), comp_2_start_points_before, comp_2_radii_before, 'orange', opacity=0.5)
plot_spheres(plotter, (0, 0), comp_3_start_points_before, comp_3_radii_before, 'red', opacity=0.5)

# Interconnects
plot_capsules(plotter, (0, 0), int_1_points_before, int_1_radius_before, color='black', opacity=0.5)

# Axis-aligned bounding box
plot_AABB(plotter, (0, 0), bounds_before, color='gray', opacity=0.15)

# Subplot 2: Pseudo-Density Projections
plotter.subplot(0, 1)
plotter.add_title("Densities Before")

# Projection
plot_grid(plotter, (0, 1), centers, element_size, densities=None)
plot_grid(plotter, (0, 1), centers, element_size, densities=densities_before)

# Axis-aligned bounding box
plot_AABB(plotter, (0, 1), bounds_before, color='gray', opacity=0.15)

plot_translation_sensitivities(plotter, (0, 1), comp_1_start_points_before[0], tot_before_comp_1, color='blue', factor=2.0)
plot_translation_sensitivities(plotter, (0, 1), comp_2_start_points_before[0], tot_before_comp_2, color='orange', factor=2.0)
plot_translation_sensitivities(plotter, (0, 1), comp_3_start_points_before[0], tot_before_comp_3, color='red', factor=2.0)


#%% Plot the System After Optimization

# # Geometry
# # plot_grid(plotter, (1, 0), centers, element_size, densities=None)
# # plot_stl_file(plotter, (1, 0), 'models/CrossHead_Pin_scaled.stl', translation=comp_1_translation_after, rotation=comp_1_rotation_after, opacity=0.25, color='blue')
# # plot_spheres(plotter, (1, 0), comp_1_spheres_after, comp_1_radii_after, 'blue', opacity=0.5)
# # # plot_capsules2(plotter, (1, 0), comp_1_start_points_after, comp_1_end_points_after, comp_1_radii_after, color='blue', opacity=0.5)
# # plot_capsules2(plotter, (1, 0), comp_2_start_points_after, comp_2_end_points_after, comp_2_radii_after, color='red', opacity=0.5)
# # plot_capsules(plotter, (1, 0), int_1_points_after, 0.25, color='green', opacity=0.5)
# #
# # # Projection
# # plot_grid(plotter, (1, 1), centers, element_size, densities=None)
# # plot_grid(plotter, (1, 1), centers, element_size, densities=densities_after)
# #
# # plot_AABB(plotter, (1, 1), bounds_after, color='blue', opacity=0.15)
# # plot_translation_sensitivities(plotter, (1, 1), comp_1_spheres_after[0], tot_after_comp_1, color='red', factor=3.0)
# # plot_translation_sensitivities(plotter, (1, 1), comp_2_start_points_after[2], tot_after_comp_2, color='red', factor=3.0)
# # plot_translation_sensitivities(plotter, (1, 1), int_1_control_points_after, tot_after_int_1, color='red', factor=3.0)
#
#

#
# # Plot the geometries after optimization
# # plotter.subplot(0, 1)
# # plotter.add_title("After Optimization")


# Plotter configurations
plotter.link_views()
plotter.show_axes()
# plotter.enable_shadows()
# plotter.enable_ssao(radius=0.5)
plotter.enable_anti_aliasing('fxaa')
# plotter.view_yz()
plotter.view_xy()
plotter.show()

#
#
# t6 = time_ns()
# print(f"Plot time: {(t6 - t5) / 1e9} seconds")
# print(f"Total time: {(t6 - t0) / 1e9} seconds")
# print('Done')


### Code for Debugging ###
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