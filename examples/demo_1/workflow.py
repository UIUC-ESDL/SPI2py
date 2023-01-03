"""

TODO Add a logger

Note: Make sure to run this from the top-level SPI2Py directory
"""

# Import the Application
from pathlib import Path
import sys

import numpy as np

SPI2py_path = str(Path(__file__).parent.parent.parent)
sys.path.append(SPI2py_path)

from src.SPI2Py.main import SPI2

# import logging
# logging.basicConfig(filename='examples/demo_1/output.log', encoding='utf-8', level=logging.DEBUG)

# For troubleshooting


# Initialize the class
demo = SPI2()

# TODO Get system location!

# Specify the input file
input_filepath = 'examples/demo_1/input.yaml'
demo.add_input_file(input_filepath)

# Specify the config file
config_filepath = 'examples/demo_1/config.yaml'
demo.add_configuration_file(config_filepath)

# Generate classes from the inputs file
demo.create_objects_from_input()

# Map the objects to a 3D layout
layout_generation_method = 'manual'
locations = np.array([-3., -4.41, -0.24, 0., 0., 0., 2., 4.41, 0.24, 0., 0., 0., -1., 2., 2.])
demo.generate_layout(layout_generation_method, inputs=locations)

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
output_filepath = 'examples/demo_1/output.json'
demo.write_output(output_filepath)
