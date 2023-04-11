"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# Import packages
import os
import numpy as np
from SPI2py import System, DesignStudy

# Define the system

system = System(name='Demo System')

# Add objects to the system
# TODO Add ports...
system.add_component(name='control_valve_1',
                     color='aquamarine',
                     movement_class='independent',
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [6, 2, 2], 'rotation': [0, 0, 0]}])

system.add_component(name='actuator_1',
                     color='orange',
                     movement_class='independent',
                     shapes=[{'type': 'box', 'origin': [-3, 0, -6], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [-3, 0, -4.5], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [-3, 0, -3], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [-2, 1, -2.5], 'dimensions': [1, 1, 5], 'rotation': [0, 0, 0]}])

system.add_component(name='component_2',
                     color='indigo',
                     movement_class='independent',
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [1, 3, 3], 'rotation': [0, 0, 0]}])

system.add_component(name='component_3',
                     color='olive',
                     movement_class='independent',
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [1, 1, 1], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 0, 0], 'dimensions': [1, 2, 1], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 0.5], 'dimensions': [1, 1, 3], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 3], 'dimensions': [2, 1, 1], 'rotation': [0, 0, 0]}])

system.add_component(name='structure_1',
                     color='gray',
                     movement_class='static',
                     shapes=[
                         {'type': 'box', 'origin': [0, 0, -5], 'dimensions': [2, 2, 0.5], 'rotation': [0, 0, 0]}])

system.add_port(component_name='control_valve_1',
                port_name='supply',
                color='red',
                radius=0.5,
                reference_point_offset=[0, 0, 1.5],
                movement_class='fully dependent')

system.add_port(component_name='control_valve_1',
                port_name='return',
                color='blue',
                radius=0.5,
                reference_point_offset=[2, 0, 1.5],
                movement_class='fully dependent')

system.add_port(component_name='actuator_1',
                port_name='supply',
                color='red',
                radius=0.5,
                reference_point_offset=[0, -1, 0],
                movement_class='fully dependent')

system.add_port(component_name='actuator_1',
                port_name='return',
                color='blue',
                radius=0.25,
                reference_point_offset=[1, -1, 0],
                movement_class='fully dependent')

system.add_interconnect(name='hp_cv_to_actuator',
                        color='black',
                        component_1='control_valve_1',
                        component_1_port='supply',
                        component_2='actuator_1',
                        component_2_port='supply',
                        radius=0.25,
                        number_of_bends=2)

system.add_interconnect(name='hp_cv_to_actuator2',
                        color='blue',
                        component_1='control_valve_1',
                        component_1_port='return',
                        component_2='actuator_1',
                        component_2_port='return',
                        radius=0.25,
                        number_of_bends=1)



# MAP STATIC OBJECTS
system.map_static_object(object_name='structure_1', design_vector=np.array([0, 0, 0, 0, 0, 0]))

# Define the design study

# Obtain the local path of this example's directory
local_directory = os.path.dirname(__file__) + '/'

# Initialize the design study
study = DesignStudy(directory=local_directory,
                    study_name='Example 1')

study.add_system(system)

# Define the username and problem description
study.config['Username'] = 'Chad Peterson'
study.config['Problem Description'] = 'Simple optimization of a 3D layout'

# Map the system to a single spatial configuration

# TODO replace add initial design vector to set_position
# TODO Set initial design vector, including the static object... enter as dict arguments to manual
# Specify for interconnect, have multiple waypoints
study.add_initial_design_vector('control_valve_1', 'spatial_config_1', [-3., -4.41, -0.24, 0., 0., 0.])
study.add_initial_design_vector('actuator_1', 'spatial_config_1', [2., 4.41, 0.24, 0., 0., 0.])
study.add_initial_design_vector('component_2', 'spatial_config_1', [5, -3, -1, 0., 0., 0.])
study.add_initial_design_vector('component_3', 'spatial_config_1', [-3., -1., 3., 0., 0., 0.])
study.add_initial_design_vector('control_valve_1-supply_actuator_1-supply_node_0', 'spatial_config_1', [-3., -2., 2.])
study.add_initial_design_vector('control_valve_1-supply_actuator_1-supply_node_1', 'spatial_config_1', [-1., 0., 2.])
study.add_initial_design_vector('control_valve_1-return_actuator_1-return_node_0', 'spatial_config_1', [4., 0., 1.])
# study.add_initial_design_vector('structure_1', 'spatial_configuration_1', [0, 0, 0, 0, 0, 0])

study.generate_spatial_configuration(name='spatial_config_1', method='manual')

# Plot initial spatial configuration
study.plot()

# Perform gradient-based optimization

study.add_objective(objective='normalized aggregate gap distance',
                    model=system,
                    options={'design vector scaling type': 'constant',
                             'design vector scaling factor': 1,
                             'objective scaling type': 'constant',
                             'objective scaling factor': 1 / 500})

study.add_constraint(constraint='signed distances',
                     model=system,
                     options={'type': 'collision',
                              'object class 1': 'component',
                              'object class 2': 'component',
                              'constraint tolerance': 0.01,
                              'constraint aggregation': 'induced exponential',
                              'constraint aggregation parameter': 3.0})

study.add_constraint(constraint='signed distances',
                     model=system,
                     options={'type': 'collision',
                              'object class 1': 'component',
                              'object class 2': 'interconnect',
                              'constraint tolerance': 0.01,
                              'constraint aggregation': 'induced exponential',
                              'constraint aggregation parameter': 3.0})

study.add_constraint(constraint='signed distances',
                     model=system,
                     options={'type': 'collision',
                              'object class 1': 'interconnect',
                              'object class 2': 'interconnect',
                              'constraint tolerance': 0.01,
                              'constraint aggregation': 'induced exponential',
                              'constraint aggregation parameter': 3.0})


study.optimize_spatial_configuration(options={'maximum number of iterations': 10,
                                              'convergence tolerance': 1e-2})

# Post-processing

# Plot the final spatial configuration
new_positions = system.calculate_positions(study.result.x)
system.set_positions(new_positions)
study.plot()

# Write output file
study.create_report()

# Print the log to see the optimization results and if any warnings or errors occurred
study.print_log()
