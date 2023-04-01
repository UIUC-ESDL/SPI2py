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
from SPI2py.main import EntryPoint

# Initialize the EntryPoint class with your current working directory, and the config & input files
demo = EntryPoint(directory=directory,
                  input_file='input.yaml')

# %% Map the system to a single spatial configuration

# Define the initial design vector
# In this example, all components and structures are static.
# Therefore, the only design variables are the 3D positions of the interconnect waypoints (waypoints can't rotate).
# Note: For now, the order of the waypoints must be the same as the order in the input file.
# Below, interconnect 0 has 2 waypoints, and interconnect 1 has 1 waypoint.
interconnect_0_node_0_position = np.array([-3., -2., 2.])
interconnect_0_node_1_position = np.array([-1., 0., 2.])
interconnect_1_node_0_position = np.array([4., 0., 1.])

initial_design_vector = np.concatenate((interconnect_0_node_0_position,
                                        interconnect_0_node_1_position,
                                        interconnect_1_node_0_position))

demo.create_spatial_configuration(method='manual', inputs=initial_design_vector)

# Plot initial spatial configuration
demo.spatial_configuration.plot()

# %% Perform gradient-based optimization

demo.optimize_spatial_configuration()

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


