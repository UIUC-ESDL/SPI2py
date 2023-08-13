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


system = SpatialInterface(name='DemoSystem',
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


# %% Define a spatial configuration for the design study


# Map the system to a single spatial configuration

x0_dict = {'control_valve_1': [-3., -4.41, -0.24, 0., 0., 0.],
           'actuator_1': [2., 4.41, 0.24, 0., 0., 0.],
           'component_2': [5, -3, -1, 0., 0., 0.],
           'component_3': [-3., -1., 3., 0., 0., 0.],
           'structure_1': [0., 0., -1., 0., 0., 0.],
           'hp_cv_to_actuator': [-3., -2., 2., -1., 0., 2.],
           'hp_cv_to_actuator2': [4., 0., 1.]}

system.set_positions(design_vector_dict=x0_dict)


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
