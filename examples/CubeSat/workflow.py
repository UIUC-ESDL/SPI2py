"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages

import numpy as np

# %% Obtain the local path of this example's directory

import os
directory = os.path.dirname(__file__) + '/'

# %% Import the SPI2py library

# EntryPoint is the main class for interacting with the SPI2py library
from SPI2py import EntryPoint

# Initialize the EntryPoint class with your current working directory and input files
demo = EntryPoint(directory=directory, input_file='input.yaml')

# Define the username and problem description
demo.config['Username'] = 'Chad Peterson'
demo.config['Problem Description'] = 'Simple optimization of a 3D layout'

# %% Map the system to a single spatial configuration

# Define the initial design vector
component_0_position = np.array([-3., -4.41, -0.24, 0., 0., 0.])
component_1_position = np.array([2., 4.41, 0.24, 0., 0., 0.])
component_2_position = np.array([5, -3, -1, 0., 0., 0.])
component_3_position = np.array([-3., -1., 3., 0., 0., 0.])

interconnect_0_node_0_position = np.array([-3., -2., 2.])
interconnect_0_node_1_position = np.array([-1., 0., 2.])
interconnect_1_node_0_position = np.array([4., 0., 1.])

initial_design_vector = np.concatenate((component_0_position,
                                        component_1_position,
                                        component_2_position,
                                        component_3_position,
                                        interconnect_0_node_0_position,
                                        interconnect_0_node_1_position,
                                        interconnect_1_node_0_position))

# noinspection DuplicatedCode
demo.create_spatial_configuration(method='manual', inputs=initial_design_vector)

# Plot initial spatial configuration
demo.spatial_configuration.plot()

# %% Perform gradient-based optimization

demo.optimize_spatial_configuration(objective_function='normalized aggregate gap distance',
                                    constraint_function='signed distances',
                                    constraint_aggregation_function='induced exponential')

# %% Post-processing

# Plot the final spatial configuration
positions_dict = demo.spatial_configuration.calculate_positions(demo.result.x)
demo.spatial_configuration.set_positions(positions_dict)
demo.spatial_configuration.plot()

# Generate GIF animation; caution uncommenting this function call will increase runtime
# demo.create_gif()

# Write output file
demo.create_report()

# Print the log to see the optimization results and if any warnings or errors occurred
demo.print_log()





