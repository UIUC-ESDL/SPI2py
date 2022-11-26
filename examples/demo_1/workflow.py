"""

TODO Add a logger
TODO place workflow in if name == __main__ statement
TODO Ensure add src to python path for pytest
Test commit for develop branch post-rebase

Note: Make sure to run this from the top-level SPI2Py directory
"""
import json
from datetime import datetime
import yaml

from src.SPI2Py.utils.optimizers.gradient_based_optimization import optimize
from src.SPI2Py.utils.spatial_topologies.force_directed_layouts import generate_random_layout
from src.SPI2Py.utils.visualization.visualization import generate_gif
from src.SPI2Py.main import SPI2

'''Set the Filepaths'''


# Assumes current working directory is main SPI2py






'''Initialize the Layout'''

# Initialize the class
demo = SPI2()

# Specify the input and config file
input_filepath = 'examples/demo_1/input.yaml'
demo.add_input_file(input_filepath)

config_filepath = 'examples/demo_1/config.yaml'
demo.add_configuration_file(config_filepath)


# Generate classes from the inputs file
layout_generation_method = 'force directed'
demo.generate_layout(layout_generation_method)

# For development: Plot initial layout
demo.layout.plot_layout()



# '''Perform Gradient-Based Optimization'''
#
#
res, design_vector_log = optimize(demo.layout)
#
#
# '''Post Processing'''
#
#
# # For development: Print Results
# print('Result:', res)
#
# # For development: Plot the final layout to see the change
demo.layout.set_positions(res.x)
demo.layout.plot_layout()
#

# Generate GIF
if demo.config['Visualization']['Output GIF'] is True:
    generate_gif(demo.layout, design_vector_log, 1, demo.config['Outputs']['Folderpath'])


'''Write output file'''


# Create a timestamp
now = datetime.now()
now_formatted = now.strftime("%d/%m/%Y %H:%M:%S")

# TODO Create a prompt to ask user for comments on the results

# Create the output dictionary
outputs = {'Placeholder': 1,
           '':1,
           'Date and time': now_formatted,
           '':1,
           'Comments': 'Placeholder'}



output_file = demo.config['Outputs']['Folderpath'] + demo.config['Outputs']['Report Filename']
with open(output_file, 'w') as f:
    json.dump(outputs, f)
