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

# Import EntryPoint, the main class used to interact with the SPI2Py library

from SPI2py.main import EntryPoint

# %%




# Initialize the EntryPoint class with your current working directory, and the config & input files
demo = EntryPoint(directory=directory,
                  config_file='config.yaml',
                  input_file='input.yaml')

# %% Define the initial design vector

# In this example, all components and structures are static.
# Therefore, the only design variables are the positions of the interconnect waypoints
# Since waypoints are single spheres, we only need design variables for x,y,z translation
pos_interconnect_0_node_0 = np.array([-3., -2., 2.])
pos_interconnect_0_node_1 = np.array([-1., 0., 2.])
pos_interconnect_1_node_0 = np.array([4., 0., 1.])
initial_design_vector = np.concatenate((pos_interconnect_0_node_0,
                                        pos_interconnect_0_node_1,
                                        pos_interconnect_1_node_0))

# Generate the layout
demo.generate_spatial_configuration(method='manual', inputs=initial_design_vector)

# Plot initial layout
demo.spatial_configuration.plot_layout()

# Perform gradient-based optimization
demo.optimize_spatial_configuration()

# Generate GIF animation; caution this function will increase runtime
# demo.create_gif_animation()

# For development: Plot the final layout to see the change
positions_dict = demo.spatial_configuration.calculate_positions(demo.result.x)
demo.spatial_configuration.set_positions(positions_dict)
demo.spatial_configuration.plot_layout()


# Write output file
# TODO Log successful writing?
demo.create_report()


# Print the log for ...
with open(demo.logger_name) as f:
    print(f.read())

print('Done')

