"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson

TODO
1. Why prob.check_partials(includes='system.comps.comp_1')
updated end points to end points not correctly identity matrix?
Same for updated ports and radii

Notes:
    1. The order you must add elements to the OpenMDAO problem is very deliberate.
       For example, adding the objective before one of its upstream inputs can result in the derivatives being zero.
    2. ...
"""


# %% Configure the problem


# Standard imports
import jax
import numpy as np
import pyvista as pv
import openmdao.api as om
from copy import copy
from time import time_ns


# SPI2py API imports
from SPI2py.API.system import System, Components, Interconnects, MDBDComponent, Interconnect
from SPI2py.API.projection import Projections, ProjectionAggregator, ProjectMDBDComponent, ProjectInterconnect
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer


# SPI2py supporting model imports
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_active_nodes, find_face_nodes
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_capsules, plot_stl_file, plot_AABB, plot_capsules2
from SPI2py.models.utilities.visualization import plot_translation_sensitivities


# Start a timer for tracking the workflow
t0 = time_ns()


# Configure JAX settings
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_debug_nans", True)


# Define the domain as a uniform grid
x_min, x_max = (0, 7)
y_min, y_max = (0, 7)
z_min, z_max = (0, 2)
element_size = 0.5
nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
centers = centers.reshape(nx, ny, nz, 1, 3)


# Optional: Define a mesh kernel
# Kernels are used to refine the accuracy of grid-based calculations.
kernel_steps_per_unit_length = 1
kernel_points, kernel_radii = create_uniform_kernel(kernel_steps_per_unit_length, mode='circumscription')
kernel_points = kernel_points.reshape(-1, 3)
kernel_radii = kernel_radii.reshape(-1, 1)


# %% Initialize the problem and model structure
# Note: Models are initialized in the order they are added to the model.


# OpenMDAO problem and model
prob   = om.Problem()
model  = prob.model

# Recorder that writes to a sqlite file
rec = om.SqliteRecorder('cases.sql')
prob.add_recorder(rec)
prob.recording_options['record_inputs'] = True
prob.recording_options['record_outputs'] = True

# SPI2py Nested Groups
system = System()
comps  = Components()
ints   = Interconnects()
proj   = Projections()

model.add_subsystem('system', system)
model.system.add_subsystem('comps', comps)
model.system.add_subsystem('ints', ints)
model.add_subsystem('proj', proj)



# %% Define the system


# Define the minimum radius of the MDBD representation
# For Bot_Eye, r >=4.0e-2 ~ 122 points, >=3.0e-2 ~ 238 points, >=2.0e-2 ~ 623 points
min_rad = 4.0e-2

# Define the components
comp_1 = MDBDComponent(filepath='csvs/Bot_Eye_5k_300s.csv', port_positions=[[0.75, 0, 0], [0, 0.75, 0]], minimum_radius=min_rad, color='blue')
comp_2 = MDBDComponent(filepath='csvs/CogDrivenGear_5k_300s.csv', port_positions=[[0, 1.75, 0], [-1.75, 0, 0]], minimum_radius=min_rad, color='orange')
comp_3 = MDBDComponent(filepath='csvs/CrossHead_Pin_5k_300s.csv', port_positions=[[0.15, 0.5, 1.0], [1.25, 0.5, 1.0]], minimum_radius=min_rad, color='red')
model.system.comps.add_subsystem('comp_1', comp_1)
model.system.comps.add_subsystem('comp_2', comp_2)
model.system.comps.add_subsystem('comp_3', comp_3)


# Define the interconnects
int_1 = Interconnect(radius=0.25, n_segments=2)
int_2 = Interconnect(radius=0.25, n_segments=2)
int_3 = Interconnect(radius=0.25, n_segments=2)
model.system.ints.add_subsystem('int_1', int_1)
model.system.ints.add_subsystem('int_2', int_2)
model.system.ints.add_subsystem('int_3', int_3)


# Connect the Interconnects to the comps
# The interconnect start and stop points are dependent on the translation and rotation of comps and their ports.
# The slicer is used to index rows.
model.connect('system.comps.comp_1.updated_ports', 'system.ints.int_1.start_point', src_indices=om.slicer[0, :])
model.connect('system.comps.comp_2.updated_ports', 'system.ints.int_1.end_point', src_indices=om.slicer[1, :])
model.connect('system.comps.comp_2.updated_ports', 'system.ints.int_2.start_point', src_indices=om.slicer[0, :])
model.connect('system.comps.comp_3.updated_ports', 'system.ints.int_2.end_point', src_indices=om.slicer[1, :])
model.connect('system.comps.comp_3.updated_ports', 'system.ints.int_3.start_point', src_indices=om.slicer[0, :])
model.connect('system.comps.comp_1.updated_ports', 'system.ints.int_3.end_point', src_indices=om.slicer[1, :])



# Project each component and interconnect onto independent copies of the mesh.
proj_c1 = ProjectMDBDComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_c2 = ProjectMDBDComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_c3 = ProjectMDBDComponent(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_i1 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_i2 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
proj_i3 = ProjectInterconnect(mesh_size=element_size, mesh_centers=centers, kernel_centers=kernel_points, kernel_radii=kernel_radii)
model.proj.add_subsystem('proj_c1', proj_c1)
model.proj.add_subsystem('proj_c2', proj_c2)
model.proj.add_subsystem('proj_c3', proj_c3)
model.proj.add_subsystem('proj_i1', proj_i1)
model.proj.add_subsystem('proj_i2', proj_i2)
model.proj.add_subsystem('proj_i3', proj_i3)


# Now connect the components and interconnects to their projections
model.connect('system.comps.comp_1.updated_sphere_positions', 'proj.proj_c1.centers')
model.connect('system.comps.comp_1.updated_sphere_radii', 'proj.proj_c1.radii')
model.connect('system.comps.comp_2.updated_sphere_positions', 'proj.proj_c2.centers')
model.connect('system.comps.comp_2.updated_sphere_radii', 'proj.proj_c2.radii')
model.connect('system.comps.comp_3.updated_sphere_positions', 'proj.proj_c3.centers')
model.connect('system.comps.comp_3.updated_sphere_radii', 'proj.proj_c3.radii')

model.connect('system.ints.int_1.updated_cyl_positions', 'proj.proj_i1.control_points')
model.connect('system.ints.int_1.updated_cyl_radius', 'proj.proj_i1.radius')
model.connect('system.ints.int_2.updated_cyl_positions', 'proj.proj_i2.control_points')
model.connect('system.ints.int_2.updated_cyl_radius', 'proj.proj_i2.radius')
model.connect('system.ints.int_3.updated_cyl_positions', 'proj.proj_i3.control_points')
model.connect('system.ints.int_3.updated_cyl_radius', 'proj.proj_i3.radius')


# Now combine (overlay) all the projections
# This requires "multiplexing" the outputs of each projection into single, unified vectors for centers and radii.
# Row dimensions are manually specified at this time. Column dimensions are self-determined (3D coord, 1D radii)
mux_centers = Multiplexer(n_i=[122, 1119, 195, 3, 3, 3, 3, 3, 3], m=3)
mux_radii = Multiplexer(n_i=[122, 1119, 195, 3, 3, 3, 3, 3, 3], m=1)
prob.model.add_subsystem('mux_centers', mux_centers)
prob.model.add_subsystem('mux_radii', mux_radii)
prob.model.connect('system.comps.comp_1.updated_sphere_positions', 'mux_centers.input_0')
prob.model.connect('system.comps.comp_2.updated_sphere_positions', 'mux_centers.input_1')
prob.model.connect('system.comps.comp_3.updated_sphere_positions', 'mux_centers.input_2')
prob.model.connect('system.comps.comp_1.updated_sphere_radii', 'mux_radii.input_0')
prob.model.connect('system.comps.comp_2.updated_sphere_radii', 'mux_radii.input_1')
prob.model.connect('system.comps.comp_3.updated_sphere_radii', 'mux_radii.input_2')
prob.model.connect('system.ints.int_1.updated_cyl_positions', 'mux_centers.input_3')
prob.model.connect('system.ints.int_2.updated_cyl_positions', 'mux_centers.input_4')
prob.model.connect('system.ints.int_3.updated_cyl_positions', 'mux_centers.input_5')
prob.model.connect('system.ints.int_1.updated_cyl_radius', 'mux_radii.input_3')
prob.model.connect('system.ints.int_2.updated_cyl_radius', 'mux_radii.input_4')
prob.model.connect('system.ints.int_3.updated_cyl_radius', 'mux_radii.input_5')


# Aggregate the pseudo-densities
n_proj = 6
projection_aggregator = ProjectionAggregator(n_projections=n_proj, rho_min=1e-2, mesh_size=element_size, mesh_centers=centers)
model.proj.add_subsystem('aggregator', projection_aggregator)
model.connect('proj.proj_c1.penalized_densities', 'proj.aggregator.densities_0')
model.connect('proj.proj_c2.penalized_densities', 'proj.aggregator.densities_1')
model.connect('proj.proj_c3.penalized_densities', 'proj.aggregator.densities_2')
model.connect('proj.proj_i1.penalized_densities', 'proj.aggregator.densities_3')
model.connect('proj.proj_i2.penalized_densities', 'proj.aggregator.densities_4')
model.connect('proj.proj_i3.penalized_densities', 'proj.aggregator.densities_5')


# Optional: Aggregate the loads
# Default values are zero. Explicit connections left for clarity.
model.connect('proj.proj_c1.penalized_heat_loads', 'proj.aggregator.heat_loads_0')
model.connect('proj.proj_c2.penalized_heat_loads', 'proj.aggregator.heat_loads_1')
model.connect('proj.proj_c3.penalized_heat_loads', 'proj.aggregator.heat_loads_2')
model.connect('proj.proj_i1.penalized_heat_loads', 'proj.aggregator.heat_loads_3')
model.connect('proj.proj_i2.penalized_heat_loads', 'proj.aggregator.heat_loads_4')
model.connect('proj.proj_i3.penalized_heat_loads', 'proj.aggregator.heat_loads_5')



# %% Configure the optimization


# Set the design variables
prob.model.add_design_var('system.comps.comp_1.translation', ref=0.25, lower=-5, upper=5)
# prob.model.add_design_var('system.comps.comp_1.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
prob.model.add_design_var('system.comps.comp_2.translation', ref=0.25, lower=-5, upper=5)
# prob.model.add_design_var('system.comps.comp_2.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
prob.model.add_design_var('system.comps.comp_3.translation', ref=0.25, lower=-5, upper=5)
# prob.model.add_design_var('system.comps.comp_3.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
# prob.model.add_design_var('system.ints.int_1.control_points', ref=0.25, lower=-5, upper=5)
# prob.model.add_design_var('system.ints.int_2.control_points', ref=0.25, lower=-5, upper=5)
# prob.model.add_design_var('system.ints.int_3.control_points', ref=0.25, lower=-5, upper=5)



# Define the objective function
bbv = BoundingBoxVolume()
model.add_subsystem('bbv', bbv)
prob.model.connect('mux_centers.stacked_output', 'bbv.centers')
prob.model.connect('mux_radii.stacked_output', 'bbv.radii')
prob.model.add_objective('bbv.volume', ref=1)


# Set the constraint(s)
prob.model.add_constraint('proj.aggregator.max_density', upper=1.1)


# Set up the optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 10


# ...
prob.setup()


# %% Run the model once to check the initial state and sensitivities


# Set the initial state
prob.set_val('system.comps.comp_1.translation', [1, 1, 1])  # Blue
prob.set_val('system.comps.comp_2.translation', [5, 2, 0])  # Orange
prob.set_val('system.comps.comp_3.translation', [0.5, 4, 1])  # Red
prob.set_val('system.comps.comp_3.rotation', [-np.pi/2, 0, 0])
prob.set_val('system.ints.int_1.control_points', [[1.6, 1.6, 0]])
prob.set_val('system.ints.int_2.control_points', [[2.5, 5, 0]])
prob.set_val('system.ints.int_3.control_points', [[0.65, 1.75, 0]])





# Run the model once
t1 = time_ns()
prob.run_model()
t2 = time_ns()

print(f"Run time: {(t2 - t1) / 1e9} seconds")


# Check the initial state
print("BBV Before:", prob.get_val('bbv.volume'))
print("Max Density:", prob.get_val('proj.aggregator.max_density'))



# %% Now run the optimization




prob.record('before')


# Run the optimization
t3 = time_ns()
# prob.run_driver()
t4 = time_ns()
print(f"Optimization time: {(t4 - t3) / 1e9} seconds")

prob.record('after')
prob.cleanup()


# Check the final state
print("BBV After:", prob.get_val('bbv.volume'))
print("Max Density:", prob.get_val('proj.aggregator.max_density'))

bounds_after = prob.get_val('bbv.bounds')
densities_after = prob.get_val('proj.aggregator.aggregated_densities')





#%% Plot the results


t5 = time_ns()

# Create multiple subplots
plotter = pv.Plotter(shape=(2, 2), window_size=(1500, 500), lighting='light kit')

# Define the subplots
sp1, sp2, sp3, sp4 = (0, 0), (1, 0), (0, 1), (1, 1)

# Load the case recorded
cr     = om.CaseReader(prob.get_outputs_dir() / 'cases.sql')
before = cr.get_case('before')
after  = cr.get_case('after')


#%% Plot the System Before Optimization


# Load the recorded case before optimization
prob.load_case(before)

# Subplots 1: Geometry and Components
plotter.subplot(*sp1)
plotter.add_title("Before")

# Check the initial sensitivities
tot_before = copy(prob.compute_totals(of=['bbv.volume'], wrt=['system.comps.comp_1.translation','system.comps.comp_2.translation','system.comps.comp_3.translation']))
tot_before_comp_1 = tot_before[('bbv.volume', 'system.comps.comp_1.translation')][0]
tot_before_comp_2 = tot_before[('bbv.volume', 'system.comps.comp_2.translation')][0]
tot_before_comp_3 = tot_before[('bbv.volume', 'system.comps.comp_3.translation')][0]

plot_grid(plotter, sp1, centers, element_size, densities=None, min_opacity=0.0)
model.system.draw(plotter, sp1, prob)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.comps.comp_1.updated_sphere_positions")[0], tot_before_comp_1, color="blue",factor=2.0)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.comps.comp_2.updated_sphere_positions")[0], tot_before_comp_2, color="orange",factor=2.0)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.comps.comp_3.updated_sphere_positions")[0], tot_before_comp_3, color="red",factor=2.0)
plot_AABB(plotter, sp1, prob.get_val('bbv.bounds'), color='gray', opacity=0)

model.proj.aggregator.draw(plotter, sp2, prob)


#%% Plot the System After Optimization

# Load the recorded case after optimization
prob.load_case(after)

# Subplots 1: Geometry and Components
plotter.subplot(*sp3)
plotter.add_title("After")

plot_grid(plotter, sp3, centers, element_size, densities=None, min_opacity=0.0)
model.system.draw(plotter, sp3, prob)
plot_AABB(plotter, sp3, prob.get_val('bbv.bounds'), color='gray', opacity=0)

model.proj.aggregator.draw(plotter, sp4, prob)


# %% Plotter configurations


plotter.link_views()
plotter.show_axes()
# plotter.enable_shadows()
# plotter.enable_ssao(radius=0.5)
plotter.enable_anti_aliasing('fxaa')
# plotter.view_yz()
plotter.view_xy()
plotter.show()



t6 = time_ns()
print(f"Plot time: {(t6 - t5) / 1e9} seconds")
print(f"Total time: {(t6 - t0) / 1e9} seconds")
print('Done')


# %% Additional code for debugging


# prob.check_totals(compact_print=True)
# prob.check_partials(compact_print=True, show_only_incorrect=True)

# Select whether to use exact or finite difference totals for the optimization
# prob.model.approx_totals(method='exact')
# or
# prob.model.approx_totals(method='fd')

# Check the final sensitivities
# tot_after = prob.compute_totals(of=['bbv.volume'], wrt=['system.comps.comp_1.translation','system.comps.comp_2.translation', 'system.comps.comp_3.translation'])
# tot_after_comp_1 = copy(tot_after[('bbv.volume', 'system.comps.comp_1.translation')][0])
# tot_after_comp_2 = copy(tot_after[('bbv.volume', 'system.comps.comp_2.translation')][0])

# Check the total derivative of an output wrt a design variable
# totals_c_approx = copy(prob.compute_totals(of=['proj.aggregator.max_density'], wrt=['system.comps.comp_2.translation']))
# pf = prob.check_partials(includes='FEA')
# tot = prob.compute_totals(of=['FEA.max_temperature'], wrt=['system.comps.comp_2.translation'])
# tot = tot[('FEA.max_temperature', 'system.comps.comp_2.translation')][0]
# print(tot)

# Check the Jacobian-vector products and vector-Jacobian products for the FEA component
# d_inputs = {'density': jnp.ones_like(densities_combined, dtype=jnp.float64), 'heat_loads': jnp.ones_like(densities_combined, dtype=jnp.float64)}
# d_outputs = {'temperature': jnp.ones_like(T, dtype=jnp.float64), 'max_temperature': jnp.array([1], dtype=jnp.float64)}
# jvp_vals = prob.model.FEA.compute_jacvec_product(prob.model.FEA._inputs, d_inputs, d_outputs, mode='fwd')
# vjp_vals = prob.model.FEA.compute_jacvec_product(prob.model.FEA._inputs, d_inputs, d_outputs, mode='rev')
# print('JVP:', jvp_vals)
# print('VJP:', vjp_vals)