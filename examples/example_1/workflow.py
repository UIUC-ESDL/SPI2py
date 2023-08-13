"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages


import os

import jax.numpy as np
from jax import grad, jacrev

from SPI2py import (SpatialInterface, Component, Interconnect, DesignStudy)


# %% Define the components

c0 = Component(name='control_valve_1',
               color='aquamarine',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [6, 2, 2], 'rotation': [0, 0, 0]}],
               ports=[{'name': 'supply', 'origin': [2, 1, 2.5], 'radius': 0.5},
                      {'name': 'return', 'origin': [4, 1, 2.5], 'radius': 0.5}])

c1 = Component(name='actuator_1',
               color='orange',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                       {'type': 'box', 'origin': [0, 0, 1.5], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                       {'type': 'box', 'origin': [0, 0, 3], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                       {'type': 'box', 'origin': [1, 1, 3.5], 'dimensions': [1, 1, 5], 'rotation': [0, 0, 0]}],
               ports=[{'name': 'supply', 'origin': [1, 0, 1], 'radius': 0.5},
                      {'name': 'return', 'origin': [2, 0, 1], 'radius': 0.5}])

c2 = Component(name='component_2',
               color='indigo',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               # Defining a component geometry using a mdbd file (a text file where each line is "x y z radius")
               mdbd_filepath='part_models/demo_part_1.txt')

c3 = Component(name='component_3',
               color='olive',
               degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
               shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [1, 1, 1], 'rotation': [0, 0, 0]},
                       {'type': 'box', 'origin': [1, 0, 0], 'dimensions': [1, 2, 1], 'rotation': [0, 0, 0]},
                       {'type': 'box', 'origin': [1, 1, 0.5], 'dimensions': [1, 1, 3], 'rotation': [0, 0, 0]},
                       {'type': 'box', 'origin': [1, 1, 3], 'dimensions': [2, 1, 1], 'rotation': [0, 0, 0]}])

c4 = Component(name='structure_1',
               color='gray',
               degrees_of_freedom=(),
               shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [2, 2, 0.5], 'rotation': [0, 0, 0]}])


# %% Define the interconnects

ic0 = Interconnect(name='hp_cv_to_actuator',
                   color='black',
                   component_1_name='control_valve_1',
                   component_1_port_name='supply',
                   component_2_name='actuator_1',
                   component_2_port_name='supply',
                   radius=0.25,
                   number_of_waypoints=2,
                   degrees_of_freedom=('x', 'y', 'z'))

ic1 = Interconnect(name='hp_cv_to_actuator2',
                   color='blue',
                   component_1_name='control_valve_1',
                   component_1_port_name='return',
                   component_2_name='actuator_1',
                   component_2_port_name='return',
                   radius=0.25,
                   number_of_waypoints=1,
                   degrees_of_freedom=('x', 'y', 'z'))

# %% Define the system


system = SpatialInterface(name='Demo System',
                          components=[c0, c1, c2, c3, c4],
                          interconnects=[ic0, ic1])


# %% Define the design study


# Obtain the local path of this example's directory
local_directory = os.path.dirname(__file__) + '/'

# Initialize the design study
study = DesignStudy(directory=local_directory,
                    study_name='Example 1')

# Add the defined system to the design study
study.add_system(system)

# Define the username and problem description
study.config['Username'] = 'Chad Peterson'
study.config['Problem Description'] = 'Simple optimization of a 3D layout'

# %% Define a spatial configuration for the design study


# Map the system to a single spatial configuration

# TODO replace add initial design vector to set_position
# TODO Set initial design vector, including the static object... enter as dict arguments to manual
# Specify for interconnect, have multiple waypoints

# Set the position of each component and interconnect waypoint
# For objects with at least one degree of freedom, the position is used as a starting point for the optimization.
# For objects with no degrees of freedom, the position is fixed.
study.set_initial_position('control_valve_1', 'spatial_config_1', [-3., -4.41, -0.24, 0., 0., 0.])
study.set_initial_position('actuator_1', 'spatial_config_1', [2., 4.41, 0.24, 0., 0., 0.])
study.set_initial_position('component_2', 'spatial_config_1', [5, -3, -1, 0., 0., 0.])
study.set_initial_position('component_3', 'spatial_config_1', [-3., -1., 3., 0., 0., 0.])
study.set_initial_position('hp_cv_to_actuator', 'spatial_config_1', [-3., -2., 2., -1., 0., 2.])
study.set_initial_position('hp_cv_to_actuator2', 'spatial_config_1', [4., 0., 1.])

# Map static objects to the spatial configuration (they do not have
study.set_initial_position('structure_1', 'spatial_config_1', [0, 0, -1, 0, 0, 0])

# Generate the spatial configuration
study.generate_spatial_configuration(name='spatial_config_1', method='manual')

# Plot initial spatial configuration
# system.plot()

# Perform gradient-based optimization

system.set_objective(objective='bounding box volume')

system.set_constraint(constraint='collision',
                      options={'object class 1': 'component',
                               'object class 2': 'component',
                               'constraint tolerance': 0.0,
                               'constraint aggregation parameter': 3.0})

system.set_constraint(constraint='collision',
                      options={'object class 1': 'component',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.01,
                               'constraint aggregation parameter': 3.0})

system.set_constraint(constraint='collision',
                      options={'object class 1': 'interconnect',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.0,
                               'constraint aggregation parameter': 3.0})

x0 = system.design_vector
objective = system.calculate_objective(x0)
constraints = system.calculate_constraints(x0)

print('Initial objective: ', objective)
print('Initial constraints: ', constraints)


def constraint_function(x):
    return system.calculate_constraints(x)[0]


grad_c = grad(constraint_function)(x0)

print('Initial constraint gradient: ', grad_c)

# study.optimize_spatial_configuration(options={'maximum number of iterations': 1,
#                                               'convergence tolerance': 1e-2})

# Post-processing

# Plot the final spatial configuration
# new_positions = system.calculate_positions(study.result.x)
# system.set_positions(new_positions)
# # system.plot()
#
# # Write output file
# study.create_report()
#
# # Print the log to see the optimization results and if any warnings or errors occurred
# study.print_log()
