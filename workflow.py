import numpy as np
import json
import yaml
from datetime import datetime
from scipy.optimize import minimize, show_options
from scipy.spatial.distance import pdist, cdist
from time import perf_counter_ns



# Set filepaths

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open(config['Input Filepath'], 'r') as f:
    inputs = yaml.safe_load(f)


# Initialize the layout


# Generate random initial layouts


# outputs = {'Placeholder': 1}
#
# with open(config['Output Filepath'], 'w') as f:
#     json.dump(outputs, f)


