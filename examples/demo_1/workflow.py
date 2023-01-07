"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson

Notes:
1. You must run this script from the top-level directory of the project (i.e., SPI2Py)
    
    (venv) C:/Users/.../SPI2Py>
    >>> python examples/demo_1/workflow.py

2. ...

"""

# %% Optional: Run in Python Interactive

# Import packages
import os

import numpy as np
from SPI2Py.main import SPI2

# Initialize the class
demo = SPI2()

# Specify the example's directory
demo.add_directory(os.path.dirname(__file__) + '/')

# Initialize the logger
demo.initialize_logger()

# Specify the input file
demo.add_input_file('input_1.yaml')
# demo.add_input_file('input_2.yaml')

# Specify the config file
demo.add_configuration_file('config.yaml')

# Generate classes from the inputs file
demo.create_objects_from_input()

# Map the objects to a 3D layout
# layout_generation_method = 'manual'

# pos_comp0 = np.array([-3., -4.41, -0.24, 0., 0., 0.])
# pos_comp1 = np.array([2., 4.41, 0.24, 0., 0., 0.])
# pos_comp2 = np.array([5,-3,-1, 0., 0., 0.])
# pos_comp3 = np.array([3., 5., 3., 0., 0., 0.])

# pos_int0node0 = np.array([-1., 2., 2.])
# pos_int1node0 = np.array([1., 2., 3.])



# locations = np.concatenate((pos_comp0, pos_comp1, pos_int0node0))
# locations = np.concatenate((pos_comp0, pos_comp1, pos_comp2, pos_comp3, pos_int0node0, pos_int1node0))


from SPI2Py.data.objects.dynamic_objects import Component

# Define component inputs
positions = [[1, 1, 1]]
radii = [1]
color = 'blue'
node = 1
name = 'test component'

# Define single port information
port_nodes_single = [0]
port_names_single = ['supply']
port_colors_single = ['blue']
port_locations_single = [[1, 2, 1]] 
port_number_of_connections_single = [1]

test_component = Component(positions, radii, color, node, name,
                                port_nodes = port_nodes_single,
                                port_names = port_names_single, 
                                port_colors = port_colors_single,
                                port_locations = port_locations_single, 
                                port_num_connections = port_number_of_connections_single)

print('next')
# demo.generate_layout(layout_generation_method, inputs=locations)

# For development: Plot initial layout
# demo.layout.plot_layout()

# # Perform gradient-based optimization
# demo.optimize_spatial_configuration()

# # For development: Print Results
# print('Result:', demo.result)

# # For development: Plot the final layout to see the change
# demo.layout.set_positions(demo.result.x)
# demo.layout.plot_layout()

# # Write output file
# output_filepath = folderpath + 'output.json'
# demo.write_output(output_filepath)

