"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import torch
import openmdao.api as om
# from SPI2py import Component, Interconnect, System
# from SPI2py.group_model.utilities.visualization import plot
from SPI2py.group_model.OpenMDAO_Objects.Components import Component
from SPI2py.group_model.OpenMDAO_Objects.Systems import System
# from SPI2py.group_model.OpenMDAO_Objects.Systems import System

# Initialize the problem
prob = om.Problem()
model = prob.model
model.add_subsystem('components', om.Group())

# Initialize components
# TODO Add names, number components...
radiator_and_ion_exchanger_1a = Component(spheres_filepath='components/radiator_and_ion_exchanger.xyzr',
                                          ports={'left': [-1.475, 0, 0], 'right': [1.475, 0, 0]},
                                          color='red')

radiator_and_ion_exchanger_1b = Component(spheres_filepath='components/radiator_and_ion_exchanger.xyzr',
                                          ports={'left': [-1.475, 0, 0], 'right': [1.475, 0, 0]},
                                          color='red')

pump_2a = Component(spheres_filepath='components/pump.xyzr',
                    ports={'top': [0, 0.275, 0], 'bottom': [0, -0.275, 0]},
                    color='blue')

pump_2b = Component(spheres_filepath='components/pump.xyzr',
                    ports={'top': [0, 0.275, 0], 'bottom': [0, -0.275, 0]},
                    color='blue')

particle_filter_3a = Component(spheres_filepath='components/particle_filter.xyzr',
                               ports={'top': [0, 0.215, 0], 'bottom': [0, -0.215, 0]},
                               color='yellow')

particle_filter_3b = Component(spheres_filepath='components/particle_filter.xyzr',
                               ports={'top': [0, 0.215, 0], 'bottom': [0, -0.215, 0]},
                               color='yellow')

fuel_cell_Stack_4a = Component(spheres_filepath='components/fuel_cell_Stack.xyzr',
                               ports={'top': [0, 0.85, 0], 'bottom': [0, -0.85, 0]},
                               color='green')

fuel_cell_Stack_4b = Component(spheres_filepath='components/fuel_cell_Stack.xyzr',
                               ports={'top': [0, 0.85, 0], 'bottom': [0, -0.85, 0]},
                               color='green')

WEG_heater_and_pump_5 = Component(spheres_filepath='components/WEG_heater_and_pump.xyzr',
                                  ports={'top': [0, 0.9665, 0], 'left': [-0.8775, 0, 0]},
                                  color='gray')

heater_core_6 = Component(spheres_filepath='components/heater_core.xyzr',
                          ports={'left': [-0.7225, 0, 0], 'right': [0.7225, 0, 0]},
                          color='red')

# Add components to the system
model.components.add_subsystem('radiator_and_ion_exchanger_1a', radiator_and_ion_exchanger_1a)
model.components.add_subsystem('radiator_and_ion_exchanger_1b', radiator_and_ion_exchanger_1b)
model.components.add_subsystem('pump_2a', pump_2a)
model.components.add_subsystem('pump_2b', pump_2b)
# model.components.add_subsystem('particle_filter_3a', particle_filter_3a)
# model.components.add_subsystem('particle_filter_3b', particle_filter_3b)
# model.components.add_subsystem('fuel_cell_Stack_4a', fuel_cell_Stack_4a)
# model.components.add_subsystem('fuel_cell_Stack_4b', fuel_cell_Stack_4b)
# model.components.add_subsystem('WEG_heater_and_pump_5', WEG_heater_and_pump_5)
# model.components.add_subsystem('heater_core_6', heater_core_6)

# # Initialize interconnects
# c1a_c1b = 1
# c1b_c2b = 1
# c2b_c3b = 1
# c3b_c4b = 1
# c4b_c5 = 1
# c5_c6 = 1
# c6_c4a = 1
# c4a_c3a = 1
# c3a_c2a = 1
# c2a_c1a = 1

# Add interconnects to the system
# system.add_subsystem()

# Connect the components and interconnects as necessary
# system.connect(...)
# system.connect(...)

# Initialize the System
system = System(num_components=4)  # ...
model.add_subsystem('system', system)


model.connect('components.radiator_and_ion_exchanger_1a.sphere_positions', 'system.comp_0_sphere_positions')
model.connect('components.radiator_and_ion_exchanger_1a.sphere_radii', 'system.comp_0_sphere_radii')
model.connect('components.radiator_and_ion_exchanger_1b.sphere_positions', 'system.comp_1_sphere_positions')
model.connect('components.radiator_and_ion_exchanger_1b.sphere_radii', 'system.comp_1_sphere_radii')
model.connect('components.pump_2a.sphere_positions', 'system.comp_2_sphere_positions')
model.connect('components.pump_2a.sphere_radii', 'system.comp_2_sphere_radii')
model.connect('components.pump_2b.sphere_positions', 'system.comp_3_sphere_positions')
model.connect('components.pump_2b.sphere_radii', 'system.comp_3_sphere_radii')

# Set the initial state
prob.setup()

# Configure the system
prob.set_val('components.radiator_and_ion_exchanger_1a.translation', [2, 7, 0])
prob.set_val('components.radiator_and_ion_exchanger_1b.translation', [5.5, 7, 0])
prob.set_val('components.pump_2a.translation', [0.5, 6, 0])
prob.set_val('components.pump_2b.translation', [7.5, 6, 0])
# prob.set_val('components.particle_filter_3a.translation', [0.5, 4.5, 0])
# prob.set_val('components.particle_filter_3b.translation', [7.5, 5, 0])
# prob.set_val('components.fuel_cell_Stack_4a.translation', [0.5, 2, 0])
# prob.set_val('components.fuel_cell_Stack_4b.translation', [7.5, 3, 0])
# prob.set_val('components.WEG_heater_and_pump_5.translation', [6, 1, 0])
# prob.set_val('components.heater_core_6.translation', [2, 0, 0])

# prob.set_val('system.c1a_c1b.control_points', [[3.475, 7, 0], [3.75, 7, 0], [4.025, 7, 0]])
# prob.set_val('system.c1b_c2b.control_points', [[6.975, 7, 0], [7.5, 7, 0], [7.5, 6.275, 0]])
# prob.set_val('system.c2b_c3b.control_points', [[7.5, 5.725, 0], [7.5, 5.47, 0], [7.5, 5.215, 0]])
# prob.set_val('system.c3b_c4b.control_points', [[7.5, 4.785, 0], [7.5, 4.3175, 0], [7.5, 3.85, 0]])
# prob.set_val('system.c4b_c5.control_points', [[7.5, 2.15, 0], [6.5, 2, 0], [6.0, 1.9665, 0]])
# prob.set_val('system.c5_c6.control_points', [[5.1225, 1.0, 0], [3, 0, 0], [2.7225, 0, 0]])
# prob.set_val('system.c6_c4a.control_points', [[1.2775, 0, 0], [0.5, 0, 0], [0.5, 1.15, 0]])
# prob.set_val('system.c4a_c3a.control_points', [[0.5, 2.85, 0], [0.5, 3, 0], [0.5, 4.285, 0]])
# prob.set_val('system.c3a_c2a.control_points', [[0.5, 4.715, 0], [0.5, 5, 0], [0.5, 5.725, 0]])
# prob.set_val('system.c2a_c1a.control_points', [[0.5, 6.275, 0], [0.5, 6.5, 0], [0.525, 7, 0]])



# from torch.autograd.functional import jacobian
# translations = torch.tensor(translations_0, dtype=torch.float64, requires_grad=True)
# rotations = torch.tensor(rotations_0, dtype=torch.float64, requires_grad=True)
# routings = torch.tensor(routings_0, dtype=torch.float64, requires_grad=True)
#
# jac_f = jacobian(system.calculate_objective, (translations, rotations, routings))
# jac_g = jacobian(system.calculate_constraints, (translations, rotations, routings))



# kinematics_component = KinematicsInterface()
# kinematics_component.options.declare('kinematics', default=system)
#
# model.add_subsystem('kinematics', kinematics_component, promotes=['*'])
#
# prob.model.add_design_var('translations', ref=10)
# prob.model.add_design_var('rotations', ref=2*torch.pi)
#
# # TODO Unbound routing
# prob.model.add_design_var('routings', ref=10, lower=routings_0, upper=routings_0)
# prob.model.add_objective('f')
# prob.model.add_constraint('g', upper=0)
# # prob.model.add_constraint('g_c', upper=0)
# # prob.model.add_constraint('g_i', upper=0)
# # prob.model.add_constraint('g_ci', upper=0)
#

#
# prob.set_val('translations', translations_0)
# prob.set_val('rotations', rotations_0)
# prob.set_val('routings', routings_0)
#
#
# # TODO Equality bound all the routing variables
#
prob.run_model()
#
# # Plot initial spatial configuration
# print('Initial objective: ', prob['f'])
# print('Initial constraint values: ', prob['g'])
# translations_i = prob['translations']
# rotations_i = prob['rotations']
# routings_i = prob['routings']
# translations_i = torch.tensor(translations_i, dtype=torch.float64)
# rotations_i = torch.tensor(rotations_i, dtype=torch.float64)
# routings_i = torch.tensor(routings_i, dtype=torch.float64)
#
# plot_objects, colors = kinematics_component.kinematics.plot_inputs(translations_i, rotations_i, routings_i)
# plot(plot_objects, colors)

#
# # Run the optimization
#
# prob.driver = om.ScipyOptimizeDriver()
# # prob.driver.options['maxiter'] = 50
#
# prob.run_driver()
#
#
# # Plot optimized spatial
# print('Optimized objective: ', prob['f'])
# print('Optimized constraint values: ', prob['g'])
# translations_f = prob['translations']
# rotations_f = prob['rotations']
# routings_f = prob['routings']
# translations_f = torch.tensor(translations_f, dtype=torch.float64)
# rotations_f = torch.tensor(rotations_f, dtype=torch.float64)
# routings_f = torch.tensor(routings_f, dtype=torch.float64)
#
# plot_objects_f, colors_f = kinematics_component.kinematics.plot_inputs(translations_f, rotations_f, routings_f)
# plot(plot_objects_f, colors_f)


# def plot_inputs(self, translations, rotations, routings):
#     """
#     Plot the model at a given state.
#     """
#
#     positions_dict = self.calculate_positions(translations, rotations, routings)
#
#     # Create the plot objects
#     objects = []
#     colors = []
#     for obj in self.objects:
#
#         positions = positions_dict[str(obj)]['positions']
#         radii = positions_dict[str(obj)]['radii']
#
#         spheres = []
#         for position, radius in zip(positions, radii):
#             spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
#
#         merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
#         # merged_clipped = merged.clip(normal='z')
#         # merged_slice = merged.slice(normal=[0, 0, 1])
#
#         objects.append(merged)
#         colors.append(obj.color)
#
#     return objects, colors

for subsystem in model.components._subsystems_myproc:
    print(subsystem.name, prob.get_val('components.' + subsystem.name + '.sphere_positions'))

import pyvista as pv

# Create the plot objects
components = []
component_colors = []

for subsystem in model.components._subsystems_myproc:

    positions = prob.get_val('components.' + subsystem.name + '.sphere_positions')
    radii = prob.get_val('components.' + subsystem.name + '.sphere_radii')
    color = subsystem.options['color']

    spheres = []
    for position, radius in zip(positions, radii):
        spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))

    merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
    # merged_clipped = merged.clip(normal='z')
    # merged_slice = merged.slice(normal=[0, 0, 1])

    components.append(merged)
    component_colors.append(color)


print('BB Bounds', prob.get_val('system.bounding box bounds'))

# Plot the bounding box
bounds = prob.get_val('system.bounding box bounds')
bb_object = pv.Box(bounds=bounds)
bb_color = 'black'


from SPI2py.group_model.utilities.visualization import plot
plot(components, component_colors, bb_object, bb_color)