"""
TODO: do this...
TODO: Set object position
TODO Add logger
TODO place workflow in if name == __main__ statement
TODO Ensure add src to python path for pytest
2. Moving objects and objective/constraint functions
3.
4.
5.
This is Chad's commit
"""

import numpy as np

import json
import yaml
from datetime import datetime

from time import perf_counter_ns

from utils.layout_generator import generate_layout

from utils.optimizer import optimize

# Set filepaths

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open(config['Input Filepath'], 'r') as f:
    inputs = yaml.safe_load(f)

# Initialize the layout
layout = generate_layout(inputs)
layout.plot_layout()

initial_layout_design_vector = layout.generate_random_layout()
layout.update_positions(initial_layout_design_vector)

positions_dict = layout.get_positions(initial_layout_design_vector)


# Generate random initial layouts
# Set random seed


# Set positions...

layout.plot_layout()

from utils.constraint_functions import constraint_component_component
print(constraint_component_component(initial_layout_design_vector, layout))


# Run solver...
res = optimize(layout)
#
print('res:', res)

layout.update_positions(res.x)
layout.plot_layout()

print('Constraint:',constraint_component_component(res.x, layout))

# outputs = {'Placeholder': 1}
#
# with open(config['Output Filepath'], 'w') as f:
#     json.dump(outputs, f)
