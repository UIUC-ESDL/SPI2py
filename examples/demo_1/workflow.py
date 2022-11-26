"""

TODO Add a logger
TODO place workflow in if name == __main__ statement
TODO Ensure add src to python path for pytest
Test commit for develop branch post-rebase

Note: Make sure to run this from the top-level SPI2Py directory
"""

from src.SPI2Py.main import SPI2

# Initialize the class
demo = SPI2()

# Specify the input file
input_filepath = 'examples/demo_1/input.yaml'
demo.add_input_file(input_filepath)

# Specify the config file
config_filepath = 'examples/demo_1/config.yaml'
demo.add_configuration_file(config_filepath)

# Generate classes from the inputs file
layout_generation_method = 'force directed'
demo.generate_layout(layout_generation_method)

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
output_filepath = 'src/SPI2py/output/output.json'
demo.write_output(output_filepath)
