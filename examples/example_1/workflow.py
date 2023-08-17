"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages


import openmdao.api as om
from SPI2py import KinematicsComponent, Component, Interconnect


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

c4 = Component(name='structure_1',
               color='gray',
               degrees_of_freedom=(),
               filepath='part_models/structure_1.xyzr')


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

kinematics_component = KinematicsComponent()
kinematics_component.options.declare('components', default=[c0, c1, c2, c3, c4], types=list)
kinematics_component.options.declare('interconnects', default=[ic0, ic1], types=list)

model.add_subsystem('kinematic', kinematics_component, promotes=['*'])

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
                          'hp_cv_actuator': {'waypoints': [[-3., -2., 2.],[-1., 0., 2.]]},
                          'lp_cv_actuator': {'waypoints': [[4., 0., 1.]]}}

kinematics_component.kinematics_interface.set_default_positions(default_positions_dict)




# %% Configure the system objective and constraints


kinematics_component.kinematics_interface.set_objective(objective='bounding box volume')



# %% Run the optimization


x0 = kinematics_component.kinematics_interface.design_vector
model.set_val('x', kinematics_component.kinematics_interface.design_vector)


prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 100

prob.run_model()

# Plot initial spatial configuration
kinematics_component.kinematics_interface.plot()

# Perform gradient-based optimization

# print('Initial design vector: ', prob['x'])
# print('Initial objective: ', prob['f'])
# print('Initial constraint values: ', prob['g'])

# spatial_component.spatial_interface.plot()


# prob.run_driver()
#
#
# print('Optimized design vector: ', prob['x'])
# print('Optimized objective: ', prob['f'])

# Plot optimized spatial
# objects_dict = spatial_component.spatial_interface.calculate_positions(prob['x'])
# spatial_component.spatial_interface.set_positions(objects_dict=objects_dict)
# spatial_component.spatial_interface.plot()