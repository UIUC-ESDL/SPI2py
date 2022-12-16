"""

TODO Add a logger

Note: Make sure to run this from the top-level SPI2Py directory
"""


from src.SPI2Py.main import SPI2

# For troubleshooting
import numpy as np
from src.SPI2Py.data.classes.objects import Interconnect

# Initialize the class
demo = SPI2()



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
locations = np.array([-3., -4.41, -0.24, 0.,  0., 0.,  2., 4.41, 0.24, 0., 0., 0., 2., 2., 2.])

pos_dict = demo.interconnect_nodes[1].calculate_positions(np.array([0,0,0]))
demo.interconnect_nodes[1].update_positions(pos_dict)

# demo.layout.set_positions(locations)

# demo.generate_layout(layout_generation_method, inputs=locations)


# mint = Interconnect(demo.layout.components[0], demo.layout.components[1], 0.25, 'black')


# For development: Plot initial layout
# demo.layout.plot_layout()


# # Perform gradient-based optimization
# demo.optimize_spatial_configuration()
#
#
# # For development: Print Results
# print('Result:', demo.result)
#
#
# # For development: Plot the final layout to see the change
# demo.layout.set_positions(demo.result.x)
# demo.layout.plot_layout()
#
#
# # Write output file
# output_filepath = 'src/SPI2py/result/output/output.json'
# demo.write_output(output_filepath)
