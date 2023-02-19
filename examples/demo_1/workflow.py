"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson


"""

# Import packages
import os
import numpy as np

# EntryPoint is the main class used to interact with the SPI2Py library
from SPI2Py.main import EntryPoint

# Define the example's directory
directory = os.path.dirname(__file__) + '/'

# Initialize the EntryPoint class with your current working directory, and the config & input files
demo = EntryPoint(directory=directory,
                  config_file='config.yaml',
                  input_file='input.yaml')

# Define the initial design vectors for the waypoints that can move
# [x,y,z] Since design variables to rotate individual spheres adds unnecessary complexity
pos_interconnect_0_node_0 = np.array([-3., -2., 2.])
pos_interconnect_0_node_1 = np.array([-1., 0., 2.])
pos_interconnect_1_node_0 = np.array([4., 0., 1.])
initial_design_vector = np.concatenate((pos_interconnect_0_node_0,
                                        pos_interconnect_0_node_1,
                                        pos_interconnect_1_node_0))

# Generate the layout
demo.generate_layout(method='manual', inputs=initial_design_vector)

# Plot initial layout
demo.layout.plot_layout()

# Perform gradient-based optimization
demo.optimize_spatial_configuration()

# Generate GIF animation
# Caution: Uncommenting this command may significantly increase the runtime
# of the script depending on the number of solver iterations.
# demo.create_gif_animation()


# For development: Plot the final layout to see the change
positions_dict = demo.layout.calculate_positions(demo.result.x)
demo.layout.set_positions(positions_dict)
demo.layout.plot_layout()


# Write output file
# TODO Log successful writing?
demo.create_report()


# Print the log for ...
with open(demo.logger_name) as f:
    print(f.read())

print('Done')
