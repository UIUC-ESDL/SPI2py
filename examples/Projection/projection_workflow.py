"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import openmdao.api as om
from SPI2py.API.system import System
from SPI2py.API.utilities import Multiplexer, MaxAggregator
from SPI2py.API.projection import Projection
from SPI2py.models.utilities.visualization import plot_problem
from SPI2py.models.utilities.inputs import read_input_file


# Read the input file
input_file = read_input_file('input.toml')

# Initialize the problem
prob = om.Problem()
model = prob.model

# Initialize the groups
model.add_subsystem('system', System(input_dict=input_file))
model.add_subsystem('projection', Projection(min_xyz=-3, max_xyz=10, n_el_xyz=20))


# Set the initial state
prob.setup()


# Configure the system
prob.set_val('system.components.comp_0.translation', [2, 7, 0])



prob.run_model()


# Check the initial state
plot_problem(prob)



