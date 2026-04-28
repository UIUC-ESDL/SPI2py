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
import matplotlib
import numpy as np
import pyvista as pv
import openmdao.api as om
from copy import copy
from time import time_ns

matplotlib.use('Agg')
import matplotlib.pyplot as plt


OBJECTIVE_NAME = 'bbv.volume'
DENSITY_CONSTRAINT_NAME = 'proj.density_constraint.max_density'
DENSITY_CONSTRAINT_UPPER = 1.1
DRIVER_CASES_FILENAME = 'driver_cases.sql'
TRAJECTORY_PLOT_FILENAME = 'optimization_trajectory.png'


def _recorded_scalar(case, getter, variable_name):
    values = getter(scaled=False)
    if variable_name not in values:
        raise KeyError(f"'{variable_name}' was not recorded. Available names: {list(values.keys())}")
    return float(np.asarray(values[variable_name]).reshape(-1)[0])


def plot_driver_trajectory(outputs_dir, reports_dir,
                           objective_name=OBJECTIVE_NAME,
                           constraint_name=DENSITY_CONSTRAINT_NAME,
                           constraint_upper=DENSITY_CONSTRAINT_UPPER):
    case_db = outputs_dir / DRIVER_CASES_FILENAME
    if not case_db.exists():
        print(f"Driver trajectory not plotted because {case_db} was not found.")
        return None

    cr = om.CaseReader(case_db)
    case_names = cr.list_cases('driver', out_stream=None)
    if not case_names:
        print(f"Driver trajectory not plotted because {case_db} contains no driver cases.")
        return None

    iterations = []
    objectives = []
    constraints = []
    for i, case_name in enumerate(case_names):
        case = cr.get_case(case_name)
        iterations.append(i)
        objectives.append(_recorded_scalar(case, case.get_objectives, objective_name))
        constraints.append(_recorded_scalar(case, case.get_constraints, constraint_name))

    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_path = reports_dir / TRAJECTORY_PLOT_FILENAME

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5), constrained_layout=True)
    axes[0].plot(iterations, objectives, marker='o', linewidth=1.5)
    axes[0].set_ylabel(objective_name)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(iterations, constraints, marker='o', linewidth=1.5)
    axes[1].axhline(constraint_upper, color='tab:red', linestyle='--',
                    linewidth=1.0, label=f'upper = {constraint_upper:g}')
    axes[1].set_xlabel('Driver iteration')
    axes[1].set_ylabel(constraint_name)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best')

    fig.suptitle('Optimization Trajectory')
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return plot_path


# SPI2py API imports
from SPI2py.API.system import System, Components, Interconnects, LinearSplineComponent, Interconnect
from SPI2py.API.projection import Projections, ProjectionConstraint
from SPI2py.API.objectives import BoundingBoxVolume
from SPI2py.API.utilities import Multiplexer


# SPI2py supporting model imports
from SPI2py.models.physics.distributed.mesh import generate_mesh
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_capsules, plot_stl_file, plot_AABB
from SPI2py.models.utilities.visualization import plot_translation_sensitivities


# Start a timer for tracking the workflow
t0 = time_ns()


# Configure JAX settings
# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)


# Define the domain as a uniform grid
x_min, x_max = (0, 14)
y_min, y_max = (0, 7)
z_min, z_max = (0, 7)
element_size = 0.3
nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
centers = centers.reshape(nx, ny, nz, 1, 3)


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



# %% Define the system


# Define the components
bot_eye_ports = [[0.75, 0, 0], [0, 0.75, 0]]
cog_ports = [[0, 1.75, 0], [-1.75, 0, 0]]
pin_ports = [[0.15, 0.5, 1.0], [1.25, 0.5, 1.0]]

comp_1 = LinearSplineComponent(start_points=[[0.0, 0.0, 0.35]], end_points=[[0.0, 0.0, 0.65]],
                               radii=[0.35], port_positions=bot_eye_ports, color='blue')
comp_2 = LinearSplineComponent(start_points=[[0.0, 0.0, 0.5]], end_points=[[0.0, 0.0, 0.5]],
                               radii=[1.5], port_positions=cog_ports, color='orange')
comp_3 = LinearSplineComponent(start_points=[[0.7, 0.5, 0.5]], end_points=[[0.7, 0.5, 1.75]],
                               radii=[0.45], port_positions=pin_ports, color='red')
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



# Now combine (overlay) all the projections
# This requires "multiplexing" the outputs of each projection into single, unified vectors for centers and radii.
# Row dimensions are manually specified at this time. Column dimensions are self-determined (3D coord, 1D radii)
comp_1_capsule_points = Multiplexer(n_i=[1, 1], m=3)
comp_1_capsule_radii = Multiplexer(n_i=[1, 1], m=1)
comp_2_capsule_points = Multiplexer(n_i=[1, 1], m=3)
comp_2_capsule_radii = Multiplexer(n_i=[1, 1], m=1)
comp_3_capsule_points = Multiplexer(n_i=[1, 1], m=3)
comp_3_capsule_radii = Multiplexer(n_i=[1, 1], m=1)
mux_centers = Multiplexer(n_i=[2, 2, 2, 3, 3, 3], m=3)
mux_radii = Multiplexer(n_i=[2, 2, 2, 3, 3, 3], m=1)
prob.model.add_subsystem('comp_1_capsule_points', comp_1_capsule_points)
prob.model.add_subsystem('comp_1_capsule_radii', comp_1_capsule_radii)
prob.model.add_subsystem('comp_2_capsule_points', comp_2_capsule_points)
prob.model.add_subsystem('comp_2_capsule_radii', comp_2_capsule_radii)
prob.model.add_subsystem('comp_3_capsule_points', comp_3_capsule_points)
prob.model.add_subsystem('comp_3_capsule_radii', comp_3_capsule_radii)
prob.model.add_subsystem('mux_centers', mux_centers)
prob.model.add_subsystem('mux_radii', mux_radii)
prob.model.connect('system.comps.comp_1.updated_start_points', 'comp_1_capsule_points.input_0')
prob.model.connect('system.comps.comp_1.updated_end_points', 'comp_1_capsule_points.input_1')
prob.model.connect('system.comps.comp_1.updated_radii', 'comp_1_capsule_radii.input_0')
prob.model.connect('system.comps.comp_1.updated_radii', 'comp_1_capsule_radii.input_1')
prob.model.connect('system.comps.comp_2.updated_start_points', 'comp_2_capsule_points.input_0')
prob.model.connect('system.comps.comp_2.updated_end_points', 'comp_2_capsule_points.input_1')
prob.model.connect('system.comps.comp_2.updated_radii', 'comp_2_capsule_radii.input_0')
prob.model.connect('system.comps.comp_2.updated_radii', 'comp_2_capsule_radii.input_1')
prob.model.connect('system.comps.comp_3.updated_start_points', 'comp_3_capsule_points.input_0')
prob.model.connect('system.comps.comp_3.updated_end_points', 'comp_3_capsule_points.input_1')
prob.model.connect('system.comps.comp_3.updated_radii', 'comp_3_capsule_radii.input_0')
prob.model.connect('system.comps.comp_3.updated_radii', 'comp_3_capsule_radii.input_1')
prob.model.connect('comp_1_capsule_points.stacked_output', 'mux_centers.input_0')
prob.model.connect('comp_2_capsule_points.stacked_output', 'mux_centers.input_1')
prob.model.connect('comp_3_capsule_points.stacked_output', 'mux_centers.input_2')
prob.model.connect('comp_1_capsule_radii.stacked_output', 'mux_radii.input_0')
prob.model.connect('comp_2_capsule_radii.stacked_output', 'mux_radii.input_1')
prob.model.connect('comp_3_capsule_radii.stacked_output', 'mux_radii.input_2')
prob.model.connect('system.ints.int_1.updated_cyl_positions', 'mux_centers.input_3')
prob.model.connect('system.ints.int_2.updated_cyl_positions', 'mux_centers.input_4')
prob.model.connect('system.ints.int_3.updated_cyl_positions', 'mux_centers.input_5')
prob.model.connect('system.ints.int_1.updated_cyl_radius', 'mux_radii.input_3')
prob.model.connect('system.ints.int_2.updated_cyl_radius', 'mux_radii.input_4')
prob.model.connect('system.ints.int_3.updated_cyl_radius', 'mux_radii.input_5')


# Aggregate the pseudo-densities
model.add_subsystem('proj', proj)

projection_constraint = ProjectionConstraint(n_components=0, n_interconnects=6,
                                             rho_min=1e-2, mesh_size=element_size, mesh_centers=centers)
model.proj.add_subsystem('density_constraint', projection_constraint)

model.connect('comp_1_capsule_points.stacked_output', 'proj.density_constraint.interconnect_points_0')
model.connect('comp_1_capsule_radii.stacked_output', 'proj.density_constraint.interconnect_radius_0')
model.connect('comp_2_capsule_points.stacked_output', 'proj.density_constraint.interconnect_points_1')
model.connect('comp_2_capsule_radii.stacked_output', 'proj.density_constraint.interconnect_radius_1')
model.connect('comp_3_capsule_points.stacked_output', 'proj.density_constraint.interconnect_points_2')
model.connect('comp_3_capsule_radii.stacked_output', 'proj.density_constraint.interconnect_radius_2')
model.connect('system.ints.int_1.updated_cyl_positions', 'proj.density_constraint.interconnect_points_3')
model.connect('system.ints.int_1.updated_cyl_radius', 'proj.density_constraint.interconnect_radius_3')
model.connect('system.ints.int_2.updated_cyl_positions', 'proj.density_constraint.interconnect_points_4')
model.connect('system.ints.int_2.updated_cyl_radius', 'proj.density_constraint.interconnect_radius_4')
model.connect('system.ints.int_3.updated_cyl_positions', 'proj.density_constraint.interconnect_points_5')
model.connect('system.ints.int_3.updated_cyl_radius', 'proj.density_constraint.interconnect_radius_5')




# %% Configure the optimization


# Set the design variables
prob.model.add_design_var('system.comps.comp_1.translation', ref=0.25, lower=-5, upper=5)
prob.model.add_design_var('system.comps.comp_2.translation', ref=0.25, lower=-5, upper=5)
prob.model.add_design_var('system.comps.comp_3.translation', ref=0.25, lower=-5, upper=5)
prob.model.add_design_var('system.comps.comp_1.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
prob.model.add_design_var('system.comps.comp_2.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
prob.model.add_design_var('system.comps.comp_3.rotation', ref=(np.pi/2)/4, lower=-np.pi, upper=np.pi)
prob.model.add_design_var('system.ints.int_1.control_points', ref=0.25, lower=-5, upper=5)
prob.model.add_design_var('system.ints.int_2.control_points', ref=0.25, lower=-5, upper=5)
prob.model.add_design_var('system.ints.int_3.control_points', ref=0.25, lower=-5, upper=5)



# Define the objective function
bbv = BoundingBoxVolume()
model.add_subsystem('bbv', bbv)
prob.model.connect('mux_centers.stacked_output', 'bbv.centers')
prob.model.connect('mux_radii.stacked_output', 'bbv.radii')
prob.model.add_objective(OBJECTIVE_NAME, ref=1)


# Set the constraint(s)
prob.model.add_constraint(DENSITY_CONSTRAINT_NAME, upper=DENSITY_CONSTRAINT_UPPER)


# Set up the optimizer
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 10

# Driver-level recorder for objective/constraint trajectories.
driver_rec = om.SqliteRecorder(DRIVER_CASES_FILENAME)
prob.driver.add_recorder(driver_rec)
prob.driver.recording_options['record_desvars'] = True
prob.driver.recording_options['record_objectives'] = True
prob.driver.recording_options['record_constraints'] = True
prob.driver.recording_options['record_inputs'] = False
prob.driver.recording_options['record_outputs'] = True
prob.driver.recording_options['record_residuals'] = False
prob.driver.recording_options['includes'] = [OBJECTIVE_NAME, DENSITY_CONSTRAINT_NAME]


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
print("Max Density:", prob.get_val('proj.density_constraint.max_density'))



# %% Now run the optimization

prob.record('before')


# Run the optimization
t3 = time_ns()
# prob.run_driver()
t4 = time_ns()
print(f"Optimization time: {(t4 - t3) / 1e9} seconds")

outputs_dir = prob.get_outputs_dir()
reports_dir = prob.get_reports_dir()
prob.record('after')
prob.cleanup()

trajectory_plot = plot_driver_trajectory(outputs_dir, reports_dir)
if trajectory_plot is not None:
    print(f"Optimization trajectory plot: {trajectory_plot}")


# Check the final state
print("BBV After:", prob.get_val('bbv.volume'))
print("Max Density:", prob.get_val('proj.density_constraint.max_density'))

bounds_after = prob.get_val('bbv.bounds')
densities_after = prob.get_val('proj.density_constraint.aggregated_densities')





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
tot_before = copy(prob.compute_totals(of=['bbv.volume'], wrt=['system.comps.comp_1.translation','system.comps.comp_2.translation','system.comps.comp_3.translation',
                                                              'system.ints.int_1.control_points','system.ints.int_2.control_points','system.ints.int_3.control_points']))
tot_before_comp_1 = tot_before[('bbv.volume', 'system.comps.comp_1.translation')][0]
tot_before_comp_2 = tot_before[('bbv.volume', 'system.comps.comp_2.translation')][0]
tot_before_comp_3 = tot_before[('bbv.volume', 'system.comps.comp_3.translation')][0]
tot_before_int_1 = tot_before[('bbv.volume', 'system.ints.int_1.control_points')][0]
tot_before_int_2 = tot_before[('bbv.volume', 'system.ints.int_2.control_points')][0]
tot_before_int_3 = tot_before[('bbv.volume', 'system.ints.int_3.control_points')][0]

plot_grid(plotter, sp1, centers, element_size, densities=None, min_opacity=0.0)
model.system.draw(plotter, sp1, prob)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.comps.comp_1.updated_start_points")[0], tot_before_comp_1, color="blue",factor=2.0)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.comps.comp_2.updated_start_points")[0], tot_before_comp_2, color="orange",factor=2.0)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.comps.comp_3.updated_start_points")[0], tot_before_comp_3, color="red",factor=2.0)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.ints.int_1.updated_cyl_positions")[1], tot_before_int_1, color="gray",factor=2.0)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.ints.int_2.updated_cyl_positions")[1], tot_before_int_2, color="gray",factor=2.0)
plot_translation_sensitivities(plotter, sp1, prob.get_val("system.ints.int_3.updated_cyl_positions")[1], tot_before_int_3, color="gray",factor=2.0)
plot_AABB(plotter, sp1, prob.get_val('bbv.bounds'), color='gray', opacity=0)

model.proj.density_constraint.draw(plotter, sp2, prob)


#%% Plot the System After Optimization

# Load the recorded case after optimization
prob.load_case(after)

# Subplots 1: Geometry and Components
plotter.subplot(*sp3)
plotter.add_title("After")

plot_grid(plotter, sp3, centers, element_size, densities=None, min_opacity=0.0)
model.system.draw(plotter, sp3, prob)
plot_AABB(plotter, sp3, prob.get_val('bbv.bounds'), color='gray', opacity=0)

model.proj.density_constraint.draw(plotter, sp4, prob)


# %% Plotter configurations


plotter.link_views()
plotter.show_axes()
# plotter.enable_shadows()
# plotter.enable_ssao(radius=0.5)
plotter.enable_anti_aliasing('fxaa')
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
# totals_c_approx = copy(prob.compute_totals(of=['proj.density_constraint.max_density'], wrt=['system.comps.comp_2.translation']))
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
