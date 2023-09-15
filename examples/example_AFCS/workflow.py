"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages

import torch
import openmdao.api as om
from SPI2py import KinematicsInterface, Kinematics, Component, Interconnect  # , System

# %% Define the kinematics model

import tomli

with open("input.toml", mode="rb") as fp:
    config = tomli.load(fp)

components_inputs = config['components']

components = []
for component_inputs in components_inputs.items():
    name = component_inputs[0]
    component_inputs = component_inputs[1]
    color = component_inputs['color']
    degrees_of_freedom = component_inputs['degrees_of_freedom']
    filepath = component_inputs['filepath']
    ports = component_inputs['ports']
    components.append(Component(name=name, color=color, degrees_of_freedom=degrees_of_freedom, filepath=filepath, ports=ports))


# c1a = Component(name='radiator_and_ion_exchanger_1a',
#                 color='purple',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/radiator_and_ion_exchanger.xyzr',
#                 ports=[{'name': 'left', 'origin': [-(2.850/2+0.05), 0, 0], 'radius': 0.05},
#                        {'name': 'right', 'origin': [(2.850/2+0.05), 0, 0], 'radius': 0.05}])
#
# c1b = Component(name='radiator_and_ion_exchanger_1b',
#                 color='purple',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/radiator_and_ion_exchanger.xyzr',
#                 ports=[{'name': 'left', 'origin': [-(2.850/2+0.05), 0, 0], 'radius': 0.05},
#                        {'name': 'right', 'origin': [(2.850/2+0.05), 0, 0], 'radius': 0.05}])
#
# c2a = Component(name='pump_2a',
#                 color='blue',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/pump.xyzr',
#                 ports=[{'name': 'top', 'origin': [0, (0.450/2+0.05), 0], 'radius': 0.05},
#                        {'name': 'bottom', 'origin': [0, -(0.450/2+0.05), 0], 'radius': 0.05}])
#
# c2b = Component(name='pump_2b',
#                 color='blue',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/pump.xyzr',
#                 ports=[{'name': 'top', 'origin': [0, (0.450/2+0.05), 0], 'radius': 0.05},
#                        {'name': 'bottom', 'origin': [0, -(0.450/2+0.05), 0], 'radius': 0.05}])
#
# c3a = Component(name='particle_filter_3a',
#                 color='yellow',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/particle_filter.xyzr',
#                 ports=[{'name': 'top', 'origin': [0, (0.330 / 2 + 0.05), 0], 'radius': 0.05},
#                        {'name': 'bottom', 'origin': [0, -(0.330 / 2 + 0.05), 0], 'radius': 0.05}])
#
# c3b = Component(name='particle_filter_3b',
#                 color='yellow',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/particle_filter.xyzr',
#                 ports=[{'name': 'top', 'origin': [0, (0.330 / 2 + 0.05), 0], 'radius': 0.05},
#                        {'name': 'bottom', 'origin': [0, -(0.330 / 2 + 0.05), 0], 'radius': 0.05}])
#
# c4a = Component(name='fuel_cell_stack_4a',
#                 color='green',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/fuel_cell_stack.xyzr',
#                 ports=[{'name': 'top', 'origin': [0, (1.600 / 2 + 0.05), 0], 'radius': 0.05},
#                        {'name': 'bottom', 'origin': [0, -(1.600 / 2 + 0.05), 0], 'radius': 0.05}])
#
# c4b = Component(name='fuel_cell_stack_4b',
#                 color='green',
#                 degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
#                 filepath='part_models/fuel_cell_stack.xyzr',
#                 ports=[{'name': 'top', 'origin': [0, (1.600 / 2 + 0.05), 0], 'radius': 0.05},
#                        {'name': 'bottom', 'origin': [0, -(1.600 / 2 + 0.05), 0], 'radius': 0.05}])
#
# c5 = Component(name='WEG_heater_and_pump_5',
#                color='gray',
#                degrees_of_freedom=(),
#                filepath='part_models/WEG_heater_and_pump.xyzr',
#                ports=[{'name': 'top', 'origin': [0, (1.833 / 2 + 0.05), 0], 'radius': 0.05},
#                       {'name': 'left', 'origin': [-(1.833 / 2 + 0.05), 0, 0], 'radius': 0.05}])

c6 = Component(name='heater_core_6',
               color='red',
               degrees_of_freedom=(),
               filepath='part_models/heater_core.xyzr',
               ports=[{'name': 'left', 'origin': [-(1.345 / 2 + 0.05), 0, 0], 'radius': 0.05},
                      {'name': 'right', 'origin': [(1.345 / 2 + 0.05), 0, 0], 'radius': 0.05}])

ic0 = Interconnect(name='c1a_c1b',
                   component_1='radiator_and_ion_exchanger_1a',
                   component_1_port='right',
                   component_2='radiator_and_ion_exchanger_1b',
                   component_2_port='left',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())

ic1 = Interconnect(name='c1b_c2b',
                   component_1='radiator_and_ion_exchanger_1b',
                   component_1_port='right',
                   component_2='pump_2b',
                   component_2_port='top',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())

ic2 = Interconnect(name='c2b_c3b',
                   component_1='pump_2b',
                   component_1_port='bottom',
                   component_2='particle_filter_3b',
                   component_2_port='top',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())

ic3 = Interconnect(name='c3b_c4b',
                   component_1='particle_filter_3b',
                   component_1_port='bottom',
                   component_2='fuel_cell_stack_4b',
                   component_2_port='top',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())


ic4 = Interconnect(name='c4b_c5',
                   component_1='fuel_cell_stack_4b',
                   component_1_port='bottom',
                   component_2='WEG_heater_and_pump_5',
                   component_2_port='top',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())


ic5 = Interconnect(name='c5_c6',
                   component_1='WEG_heater_and_pump_5',
                   component_1_port='left',
                   component_2='heater_core_6',
                   component_2_port='right',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())


ic6 = Interconnect(name='c6_c4a',
                   component_1='heater_core_6',
                   component_1_port='left',
                   component_2='fuel_cell_stack_4a',
                   component_2_port='bottom',
                   radius=0.05,
                   linear_spline_segments=2,
                   degrees_of_freedom=('x', 'y', 'z'))


ic7 = Interconnect(name='c4a_c3a',
                   component_1='fuel_cell_stack_4a',
                   component_1_port='top',
                   component_2='particle_filter_3a',
                   component_2_port='bottom',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())


ic8 = Interconnect(name='c3a_c2a',
                   component_1='particle_filter_3a',
                   component_1_port='top',
                   component_2='pump_2a',
                   component_2_port='top',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())


ic9 = Interconnect(name='c2a_c1a',
                   component_1='pump_2a',
                   component_1_port='top',
                   component_2='radiator_and_ion_exchanger_1a',
                   component_2_port='left',
                   radius=0.05,
                   linear_spline_segments=1,
                   degrees_of_freedom=())

# components = [c1a, c1b, c2a, c2b, c3a, c3b, c4a, c4b, c5, c6]

kinematics = Kinematics(components=components,
                        interconnects=[ic0, ic1, ic2, ic3, ic4, ic5, ic6, ic7, ic8, ic9],
                        objective='bounding box volume')

# %% Define the initial spatial configuration

default_positions_dict = {'radiator_and_ion_exchanger_1a':
                              {'translation': [2, 7, 0],
                               'rotation': [0., 0., 0.]},
                          'radiator_and_ion_exchanger_1b':
                              {'translation': [5.5, 7, 0],
                               'rotation': [0., 0., 0.]},
                          'pump_2a':
                              {'translation': [0.5, 6, 0],
                               'rotation': [0., 0., 0.]},
                          'pump_2b':
                              {'translation': [7.5, 6, 0],
                               'rotation': [0., 0., 0.]},
                          'particle_filter_3a':
                              {'translation': [0.5, 4.5, 0],
                               'rotation': [0., 0., 0.]},
                          'particle_filter_3b':
                              {'translation': [7.5, 5, 0],
                               'rotation': [0., 0., 0.]},
                          'fuel_cell_stack_4a':
                              {'translation': [0.5, 2, 0],
                               'rotation': [0., 0., 0.]},
                          'fuel_cell_stack_4b':
                              {'translation': [7.5, 3, 0],
                               'rotation': [0., 0., 0.]},
                          'WEG_heater_and_pump_5':
                              {'translation': [6, 1., 0],
                               'rotation': [0., 0., 0.]},
                          'heater_core_6':
                              {'translation': [2, 0., 0],
                               'rotation': [0., 0., 0.]},
                          'c1a_c1b': {'waypoints': []},
                          'c1b_c2b': {'waypoints': []},
                          'c2b_c3b': {'waypoints': []},
                          'c3b_c4b': {'waypoints': []},
                          'c4b_c5': {'waypoints': []},
                          'c5_c6': {'waypoints': []},
                          'c6_c4a': {'waypoints': [[1, 0, 0]]},
                          'c4a_c3a': {'waypoints': []},
                          'c3a_c2a': {'waypoints': []},
                          'c2a_c1a': {'waypoints': []}}

kinematics.set_default_positions(default_positions_dict)

# %% Define the system

prob = om.Problem()
model = prob.model

kinematics_component = KinematicsInterface()
kinematics_component.options.declare('kinematics', default=kinematics)

model.add_subsystem('kinematics', kinematics_component, promotes=['*'])

prob.model.add_design_var('x')
prob.model.add_objective('f')
prob.model.add_constraint('g', upper=0)

prob.setup()


# %% Run the optimization


# x0 = kinematics_component.kinematics.design_vector
# model.set_val('x', kinematics_component.kinematics.design_vector)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 300

prob.run_model()

# Plot initial spatial configuration
# kinematics_component.kinematics_interface.plot()

# Perform gradient-based optimization

# print('Initial design vector: ', prob['x'])
# print('Initial objective: ', prob['f'])
# print('Initial constraint values: ', prob['g'])

# x0 = torch.tensor(prob['x'], dtype=torch.float64)
# objects_dict = kinematics_component.kinematics_interface.calculate_positions(x0)
# kinematics_component.kinematics_interface.set_positions(objects_dict)
# kinematics_component.kinematics_interface.plot()

# prob.run_driver()


print('Optimized design vector: ', prob['x'])
print('Optimized objective: ', prob['f'])
print('Optimized constraint values: ', prob['g'])

# Plot optimized spatial
xf = torch.tensor(prob['x'], dtype=torch.float64)
objects_dict = kinematics_component.kinematics.calculate_positions(xf)
kinematics_component.kinematics.set_positions(objects_dict)
kinematics_component.kinematics.plot()
