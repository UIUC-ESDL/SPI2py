"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson

Notes:
1. You must run this script from the top-level directory of the project (i.e., SPI2Py)
    
    (venv) C:/Users/.../SPI2Py>
    >>> python examples/demo_1/workflow.py

2. ...

TODO Move design vector global statement from optimization to main?

"""

# Import packages
import os

import numpy as np

from SPI2Py.main import SPI2

# Define the example's directory
directory = os.path.dirname(__file__) + '/'

# Initialize the class
demo = SPI2(directory=directory,
            input_file='input_3.yaml',
            config_file='config.yaml')


# Map the objects to a 3D layout
layout_generation_method = 'manual'

# Define the design vectors of each object
pos_comp0 = np.array([-3., -4.41, -0.24, 0., 0., 0.])
pos_comp1 = np.array([2., 4.41, 0.24, 0., 0., 0.])
pos_comp2 = np.array([5, -3, -1, 0., 0., 0.])
pos_comp3 = np.array([-3., -1., 3., 0., 0., 0.])
pos_int0node0 = np.array([-3., -2., 2.])
pos_int0node1 = np.array([-1., 0., 2.])
pos_int1node0 = np.array([4., 0., 1.])
pos_int1node1 = np.array([4., 2., 1.])

locations = np.concatenate((pos_comp0, pos_comp1, pos_comp2, pos_comp3, pos_int0node0, pos_int0node1, pos_int1node0, pos_int1node1))

# Generate the layout
demo.generate_layout(layout_generation_method, inputs=locations)


demo.layout.components[0].movement = 'static'
demo.layout.components[1].movement = 'static'
demo.layout.components[2].movement = 'static'
demo.layout.components[3].movement = 'static'

# For development: Plot initial layout
demo.layout.plot_layout()

# Perform gradient-based optimization
demo.optimize_spatial_configuration()

# For development: Print Results
print('Result:', demo.result)

# For development: Plot the final layout to see the change
demo.layout.set_positions(demo.result.x)
demo.layout.plot_layout()

# Write output file
demo.write_output('output.json')

print('Done')
