"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson


"""

# Import packages
import os
import numpy as np
from SPI2Py.main import SPI2

# Define the example's directory
directory = os.path.dirname(__file__) + '/'

# Initialize the class
demo = SPI2(directory=directory,
            config_file='config.yaml',
            input_file='input.yaml')


# Map the objects to a 3D layout
layout_generation_method = 'manual'

# Define the design vectors of each object
# ['control_valve_1','actuator_1','component_2','component_3']
# pos_comp0 = np.array([-3., -4.41, -0.24, 0., 0., 0.])
# pos_comp1 = np.array([2., 4.41, 0.24, 0., 0., 0.])
# pos_comp2 = np.array([5, -3, -1, 0., 0., 0.])
# pos_comp3 = np.array([-3., -1., 3., 0., 0., 0.])

# Set waypoint positions
pos_int0node0 = np.array([-3., -2., 2.])
pos_int0node1 = np.array([-1., 0., 2.])
pos_int1node0 = np.array([4., 0., 1.])
pos_int1node1 = np.array([4., 2., 1.])
locations = np.concatenate((pos_int0node0, pos_int0node1, pos_int1node0, pos_int1node1))

# Generate the layout
demo.generate_layout(layout_generation_method, inputs=locations)

# print(demo.layout.static_objects)


# demo.layout.components[0].movement_class = 'static'
# demo.layout.components[1].movement_class = 'static'
# demo.layout.components[2].movement_class = 'static'
# demo.layout.components[3].movement_class = 'static'

# For development: Plot initial layout
demo.layout.plot_layout()

# Perform gradient-based optimization
demo.optimize_spatial_configuration()

# For development: Print Results
print('Result:', demo.result)

# For development: Plot the final layout to see the change
positions_dict = demo.layout.calculate_positions(demo.result.x)
demo.layout.set_positions(positions_dict)
demo.layout.plot_layout()

# Write output file
demo.write_output('output.json')

print('Done')
