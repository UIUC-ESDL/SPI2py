"""
To Do:
1. Set object position
2. Moving objects and objective/constraint functions
3.
4.
5.

"""

import numpy as np
import json
import yaml
from datetime import datetime
from scipy.optimize import minimize, show_options
from scipy.spatial.distance import pdist, cdist
from time import perf_counter_ns

from utils.layout_generator import generate_layout
from utils.shape_generator import generate_rectangular_prism, generate_rectangular_prisms
from utils.objects import Component, Interconnect, InterconnectNode, Structure, Layout
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
#
# new_design_vector = design_vector + 1
#
# layout.design_
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


