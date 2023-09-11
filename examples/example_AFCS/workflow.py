"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages

import torch
import openmdao.api as om
from SPI2py import KinematicsInterface, Kinematics, Component, Interconnect, System


# %% Define the kinematics model

c0 = Component(name='control_valve_1',
               color='aquamarine',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               filepath='part_models/control_valve_1.xyzr',
               ports=[{'name': 'supply', 'origin': [2, 1, 2.5], 'radius': 0.5},
                      {'name': 'return', 'origin': [4, 1, 2.5], 'radius': 0.5}])

c1 = Component(name='actuator_1',
               color='orange',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               filepath='part_models/actuator_1.xyzr',
               ports=[{'name': 'supply', 'origin': [1, 0, 1], 'radius': 0.5},
                      {'name': 'return', 'origin': [2, 0, 1], 'radius': 0.5}])

c2 = Component(name='component_2',
               color='indigo',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               filepath='part_models/component_2.xyzr')

c3 = Component(name='component_3',
               color='olive',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               filepath='part_models/component_3.xyzr')

c4 = Component(name='structure_1',
               color='gray',
               degrees_of_freedom=(),
               filepath='part_models/structure_1.xyzr')

ic0 = Interconnect(name='hp_cv_actuator',
                   color='black',
                   component_1_name='control_valve_1',
                   component_1_port_index=c0.port_indices['supply'],
                   component_2_name='actuator_1',
                   component_2_port_index=c1.port_indices['supply'],
                   radius=0.25,
                   number_of_waypoints=2,
                   degrees_of_freedom=('x', 'y', 'z'))

ic1 = Interconnect(name='lp_cv_actuator',
                   color='blue',
                   component_1_name='control_valve_1',
                   component_1_port_index=c0.port_indices['return'],
                   component_2_name='actuator_1',
                   component_2_port_index=c1.port_indices['return'],
                   radius=0.25,
                   number_of_waypoints=1,
                   degrees_of_freedom=('x', 'y', 'z'))

system = System(components=[c0, c1, c2, c3, c4],
                interconnects=[ic0, ic1])


kinematics = Kinematics(components=[c0, c1, c2, c3, c4],
                                           interconnects=[ic0, ic1],
                                           objective='bounding box volume')


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


# %% Define the initial spatial configuration

default_positions_dict = {'control_valve_1': {'translation': [-3., -4.41, -0.24],
                                              'rotation': [0., 0., 0.],
                                              'scale': [1., 1., 1.]},
                          'actuator_1': {'translation': [2., 4.41, 0.24],
                                         'rotation': [0., 0., 0.],
                                         'scale': [1., 1., 1.]},
                          'component_2': {'translation': [-5, 3, 1],
                                          'rotation': [0., 0., 0.],
                                          'scale': [1., 1., 1.]},
                          'component_3': {'translation': [3., 1., -3.],
                                          'rotation': [0., 0., 0.],
                                          'scale': [1., 1., 1.]},
                          'structure_1': {'translation': [0., 0., -1.],
                                          'rotation': [0., 0., 0.],
                                          'scale': [1., 1., 1.]},
                          'hp_cv_actuator': {'waypoints': [[-3., -2., 2.], [-1., 0., 2.]]},
                          'lp_cv_actuator': {'waypoints': [[4., 0., 1.]]}}

kinematics_component.kinematics.set_default_positions(default_positions_dict)





# %% Run the optimization


x0 = kinematics_component.kinematics.design_vector
model.set_val('x', kinematics_component.kinematics.design_vector)



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