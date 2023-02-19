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

# Map static objects to the layout
demo.layout.map_object('control_valve_1', [-3., -4.41, -0.24, 0., 0., 0.])
demo.layout.map_object('actuator_1', [2., 4.41, 0.24, 0., 0., 0.])
demo.layout.map_object('component_2', [5, -3, -1, 0., 0., 0.])
demo.layout.map_object('component_3', [-3., -1., 3., 0., 0., 0.])
demo.layout.map_object('bedplate', [-1., -1., -3, 0., 0., 0.])

# Define the initial design vectors for the waypoints
pos_int0node0 = np.array([-3., -2., 2.])
pos_int0node1 = np.array([-1., 0., 2.])
pos_int1node0 = np.array([4., 0., 1.])
locations = np.concatenate((pos_int0node0, pos_int0node1, pos_int1node0))

# Map the objects to a 3D layout
layout_generation_method = 'manual'

# Generate the layout
demo.generate_layout(layout_generation_method, inputs=locations)

# For development: Plot initial layout
demo.layout.plot_layout()

# Perform gradient-based optimization
demo.optimize_spatial_configuration()

# Generate GIF animation
# demo.create_gif_animation()

# For development: Print Results
print('Result:', demo.result)

# For development: Plot the final layout to see the change
positions_dict = demo.layout.calculate_positions(demo.result.x)
demo.layout.set_positions(positions_dict)
demo.layout.plot_layout()

# Write output file
demo.create_report()

print('Done')
