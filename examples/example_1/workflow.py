"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages


import os

import jax.numpy as np
from jax import grad, jacrev
import openmdao.api as om

from SPI2py import SpatialComponent, Component, Interconnect


# %% Define the components

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

# c4 = Component(name='structure_1',
#                color='gray',
#                degrees_of_freedom=(),
#                filepath='part_models/structure_1.xyzr')


# %% Define the interconnects

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




# %% Define the system

prob = om.Problem()
model = prob.model

spatial_component = SpatialComponent()
spatial_component.options.declare('name', default='DemoSystem', types=str)
spatial_component.options.declare('components', default=[c0, c1, c2, c3], types=list)  # TODO Add c4
spatial_component.options.declare('interconnects', default=[ic0, ic1], types=list)

model.add_subsystem('system', spatial_component, promotes=['*'])


prob.setup()


# %% Define the initial spatial configuration


spatial_component.spatial_interface.set_position('control_valve_1',
                            translation=[-3., -4.41, -0.24],
                            rotation=[0., 0., 0.],
                            scale=[1., 1., 1.])

spatial_component.spatial_interface.set_position('actuator_1',
                            translation=[2., 4.41, 0.24],
                            rotation=[0., 0., 0.],
                            scale=[1., 1., 1.])

spatial_component.spatial_interface.set_position('component_2',
                            translation=[-5, 3, 1],
                            rotation=[0., 0., 0.],
                            scale=[1., 1., 1.])

spatial_component.spatial_interface.set_position('component_3',
                            translation=[3., 1., -3.],
                            rotation=[0., 0., 0.],
                            scale=[1., 1., 1.])

# spatial_component.spatial_interface.set_position('structure_1',
#                             translation=[0., 0., -1.],
#                             rotation=[0., 0., 0.],
#                             scale=[1., 1., 1.])

spatial_component.spatial_interface.set_position('hp_cv_actuator',
                            waypoints=[[-3., -2., 2.],[-1., 0., 2.]])

spatial_component.spatial_interface.set_position('lp_cv_actuator',
                            waypoints=[[4., 0., 1.]])


# %% Configure the system objective and constraints


spatial_component.spatial_interface.set_objective(objective='bounding box volume')

spatial_component.spatial_interface.set_constraint(constraint='collision',
                      options={'object class 1': 'component',
                               'object class 2': 'component',
                               'constraint tolerance': 0.0,
                               'constraint aggregation parameter': 3.0})

spatial_component.spatial_interface.set_constraint(constraint='collision',
                      options={'object class 1': 'component',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.01,
                               'constraint aggregation parameter': 3.0})

spatial_component.spatial_interface.set_constraint(constraint='collision',
                      options={'object class 1': 'interconnect',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.0,
                               'constraint aggregation parameter': 3.0})


# %% Run the optimization


x0 = spatial_component.spatial_interface.design_vector
model.set_val('x', spatial_component.spatial_interface.design_vector)


prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 1

prob.run_model()

# Plot initial spatial configuration
# spatial_component.spatial_interface.plot()

# Perform gradient-based optimization

# print('Initial design vector: ', prob['x'])
# print('Initial objective: ', prob['f'])
# print('Initial constraint values: ', prob['g1'])
# print('Initial constraint gradient values: ', prob.compute_totals('g1','x'))

# from jax import config
# config.update("jax_debug_nans", True)

x0 = spatial_component.spatial_interface.design_vector

# grad_f = grad(spatial_component.spatial_interface.calculate_objective)
# grad_f_val = grad_f(x0)
# print(grad_f_val)

grad_c = grad(spatial_component.spatial_interface.calculate_constraints)
grad_c_val = grad_c(x0)
print(grad_c_val)



# prob.run_driver()
#
# # Plot optimized spatial configuration
# # spatial_component.spatial_interface.plot()
#
# print('Optimized design vector: ', prob['x'])
# print('Optimized objective: ', prob['f'])
