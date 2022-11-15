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


'''Set the Filepaths'''


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

with open(config['Input Filepath'], 'r') as f:
    inputs = yaml.safe_load(f)


'''Initialize the Layout'''

# Generate objects from the inputs file
layout = generate_layout(inputs)

# Generate a random initial layout
initial_layout_design_vector = layout.generate_random_layout()
layout.set_positions(initial_layout_design_vector)

# Plot the initial layout
layout.plot_layout()

'''Perform Gradient-Based Optimization'''


res, design_vector_log = optimize(layout)

print('res:', res)

layout.set_positions(res.x)
layout.plot_layout()


# Generate GIF



'''Write output file'''

outputs = {'Placeholder': 1}

with open(config['Output Filepath'], 'w') as f:
    json.dump(outputs, f)
