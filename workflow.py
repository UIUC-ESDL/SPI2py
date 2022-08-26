import numpy as np
import json
import yaml
from datetime import datetime
from scipy.optimize import minimize, show_options
from scipy.spatial.distance import pdist, cdist
from time import perf_counter_ns

from utils.layout_generator import generate_layout
from utils.shape_generator import generate_rectangular_prism, generate_rectangular_prisms
from utils.objects import Component, Interconnect, InterconnectNode, Structure


# Set filepaths

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open(config['Input Filepath'], 'r') as f:
    inputs = yaml.safe_load(f)


# Initialize the layout
# generate_layout(inputs)

components = []
for component in inputs['components']:
    name = inputs['components'][component]['name']
    origins = inputs['components'][component]['origins']
    dimensions = inputs['components'][component]['dimensions']
    diameters = inputs['components'][component]['diameters']

    positions, radii = generate_rectangular_prisms(origins, dimensions, diameters)
    components.append(Component(name,positions,radii))


# Generate random initial layouts


# outputs = {'Placeholder': 1}
#
# with open(config['Output Filepath'], 'w') as f:
#     json.dump(outputs, f)


