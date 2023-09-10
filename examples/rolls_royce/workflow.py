"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages

import torch
import openmdao.api as om
from SPI2py import KinematicsInterface, KinematicsComponent, Component, Interconnect


# %% Define the kinematics model

c0 = Component(name='engine',
               color='aquamarine',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               filepath='spheres.txt',
               ports=[{'name': 'supply', 'origin': [2, 1, 2.5], 'radius': 0.5},
                      {'name': 'return', 'origin': [4, 1, 2.5], 'radius': 0.5}])

ic0 = Interconnect(name='hp_cv_actuator',
                   color='black',
                   component_1_name='engine',
                   component_1_port_index=c0.port_indices['supply'],
                   component_2_name='engine',
                   component_2_port_index=c0.port_indices['supply'],
                   radius=0.25,
                   number_of_waypoints=2,
                   degrees_of_freedom=('x', 'y', 'z'))


kinematics_interface = KinematicsInterface(components=[c0],
                                           interconnects=[ic0],
                                           objective='bounding box volume')


# %% Define the system

prob = om.Problem()
model = prob.model

kinematics_component = KinematicsComponent()
kinematics_component.options.declare('kinematic_interface', default=kinematics_interface)

model.add_subsystem('kinematics', kinematics_component, promotes=['*'])

prob.model.add_design_var('x')
prob.model.add_objective('f')
prob.model.add_constraint('g', upper=0)

prob.setup()


# %% Define the initial spatial configuration

default_positions_dict = {'engine': {'translation': [-3., -4.41, -0.24],
                                              'rotation': [0., 0., 0.],
                                              'scale': [1., 1., 1.]},
                          'hp_cv_actuator': {'waypoints': [[-3., -2., 2.],[-1., 0., 2.]]}
                          }

kinematics_component.kinematics.set_default_positions(default_positions_dict)





# %% Run the optimization


x0 = kinematics_component.kinematics.design_vector
model.set_val('x', kinematics_component.kinematics.design_vector)



prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 300



prob.run_model()

# Plot initial spatial configuration
# kinematics_component.kinematics_interface.plot()

# # Perform gradient-based optimization
#
# print('Initial design vector: ', prob['x'])
# print('Initial objective: ', prob['f'])
# print('Initial constraint values: ', prob['g'])
#
# # x0 = torch.tensor(prob['x'], dtype=torch.float64)
# # objects_dict = kinematics_component.kinematics_interface.calculate_positions(x0)
# # kinematics_component.kinematics_interface.set_positions(objects_dict)
# # kinematics_component.kinematics_interface.plot()
#
# prob.run_driver()
#
#
# print('Optimized design vector: ', prob['x'])
# print('Optimized objective: ', prob['f'])
# print('Optimized constraint values: ', prob['g'])
#
# # Plot optimized spatial
# xf = torch.tensor(prob['x'], dtype=torch.float64)
# objects_dict = kinematics_component.kinematics_interface.calculate_positions(xf)
# kinematics_component.kinematics_interface.set_positions(objects_dict)
# kinematics_component.kinematics_interface.plot()