"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""


# %% Import packages


import os
from SPI2py import System, DesignStudy


# %% Define the system


system = System(name='Demo System')

system.add_component(name='control_valve_1',
                     color='aquamarine',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     # Defining a component geometry using geometric primitives (currently, only 'box' is supported)
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [6, 2, 2], 'rotation': [0, 0, 0]}],
                     # Defining ports; the origin is local WRT the object, not global WRT the axes
                     ports=[{'name': 'supply', 'origin': [2, 1, 2.5], 'radius': 0.5},
                            {'name': 'return', 'origin': [4, 1, 2.5], 'radius': 0.5}])

system.add_component(name='actuator_1',
                     color='orange',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [0, 0, 1.5], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [0, 0, 3], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 3.5], 'dimensions': [1, 1, 5], 'rotation': [0, 0, 0]}],
                     ports=[{'name': 'supply', 'origin': [1, 0, 1], 'radius': 0.5},
                            {'name': 'return', 'origin': [2, 0, 1], 'radius': 0.5}])


system.add_component(name='component_2',
                     color='indigo',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     # Defining a component geometry using a mdbd file (a text file where each line is "x y z radius")
                     mdbd_filepath='part_models/demo_part_1.txt')

system.add_component(name='component_3',
                     color='olive',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [1, 1, 1], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 0, 0], 'dimensions': [1, 2, 1], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 0.5], 'dimensions': [1, 1, 3], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 3], 'dimensions': [2, 1, 1], 'rotation': [0, 0, 0]}])

system.add_component(name='structure_1',
                     color='gray',
                     movement_class='static',
                     degrees_of_freedom=(),
                     shapes=[
                         {'type': 'box', 'origin': [0, 0, 0], 'dimensions': [2, 2, 0.5], 'rotation': [0, 0, 0]}])

system.add_interconnect(name='hp_cv_to_actuator',
                        color='black',
                        component_1='control_valve_1',
                        component_1_port='supply',
                        component_2='actuator_1',
                        component_2_port='supply',
                        radius=0.25,
                        number_of_bends=2,
                        degrees_of_freedom=(('x', 'y', 'z'), ('x', 'y', 'z')))

system.add_interconnect(name='hp_cv_to_actuator2',
                        color='blue',
                        component_1='control_valve_1',
                        component_1_port='return',
                        component_2='actuator_1',
                        component_2_port='return',
                        radius=0.25,
                        number_of_bends=1,
                        degrees_of_freedom=('x', 'y', 'z'))


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
system.map_static_object(object_name='structure_1', design_vector=[0, 0, -1, 0, 0, 0])


# Generate the spatial configuration
study.generate_spatial_configuration(name='spatial_config_1', method='manual')



# Plot initial spatial configuration
system.plot()

# Perform gradient-based optimization

system.add_objective(objective='normalized aggregate gap distance',
                     options={'design vector scaling type': 'constant',
                              'design vector scaling factor': 1,
                              'objective scaling type': 'constant',
                              'objective scaling factor': 1 / 500})

system.add_constraint(constraint='signed distances',
                      options={'type': 'collision',
                               'object class 1': 'component',
                               'object class 2': 'component',
                               'constraint tolerance': 0.0,
                               'constraint aggregation': 'induced exponential',
                               'constraint aggregation parameter': 3.0})

system.add_constraint(constraint='signed distances',
                      options={'type': 'collision',
                               'object class 1': 'component',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.01,
                               'constraint aggregation': 'induced exponential',
                               'constraint aggregation parameter': 3.0})

system.add_constraint(constraint='signed distances',
                      options={'type': 'collision',
                               'object class 1': 'interconnect',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.0,
                               'constraint aggregation': 'induced exponential',
                               'constraint aggregation parameter': 3.0})


study.optimize_spatial_configuration(options={'maximum number of iterations': 50,
                                              'convergence tolerance': 1e-2})

# Post-processing

# Plot the final spatial configuration
new_positions = system.calculate_positions(study.result.x)
system.set_positions(new_positions)
system.plot()

# Write output file
study.create_report()

# Print the log to see the optimization results and if any warnings or errors occurred
study.print_log()




