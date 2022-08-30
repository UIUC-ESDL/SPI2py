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


# Set filepaths

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open(config['Input Filepath'], 'r') as f:
    inputs = yaml.safe_load(f)


# Initialize the layout
layout = generate_layout(inputs)



# Generate random initial layouts
positions = layout.generate_random_layout()

# Set positions...


# Run solver...






# outputs = {'Placeholder': 1}
#
# with open(config['Output Filepath'], 'w') as f:
#     json.dump(outputs, f)


