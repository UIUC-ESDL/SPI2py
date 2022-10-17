"""
TODO: do this...
TODO: Set object position
TODO Add logger
TODO place workflow in if name == __main__ statement
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

# from utils.optimizer import optimize

# Set filepaths

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open(config['Input Filepath'], 'r') as f:
    inputs = yaml.safe_load(f)

# Initialize the layout
layout = generate_layout(inputs)

layout.plot_layout()

# design_vector = layout.design_vector
# new_design_vector = design_vector + 3

# Update component 1

component0 = layout.components[0]

component0_dv = component0.design_vector

component0_ndv = component0_dv + [1, 1, 1, 0, 0, 0]

component0.update_positions(component0_ndv)

layout.plot_layout()

# start, stop = layout.slice_design_vector()

# layout.update_positions(new_design_vector)
# initial_layout = layout.generate_random_layout()

# layout.reference_positions = initial_layout

# layout.plot_layout()


# Generate random initial layouts
# Set random seed
# positions = layout.generate_random_layout()

# Set positions...


# Run solver...


# outputs = {'Placeholder': 1}
#
# with open(config['Output Filepath'], 'w') as f:
#     json.dump(outputs, f)
