"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages

import tomli
import torch
import openmdao.api as om
from SPI2py import KinematicsInterface, Component, Interconnect, System

# %% Define the system

system = System('input.toml')

# %% Define the initial spatial configuration

with open("spatial_configurations.toml", mode="rb") as fp:
    default_positions_dict = tomli.load(fp)


system.set_default_positions(default_positions_dict)

# %% Define the system

prob = om.Problem()
model = prob.model

kinematics_component = KinematicsInterface()
kinematics_component.options.declare('kinematics', default=system)

model.add_subsystem('kinematics', kinematics_component, promotes=['*'])

prob.model.add_design_var('x')
prob.model.add_objective('f')
prob.model.add_constraint('g', upper=0)

prob.setup()


# %% Run the optimization


# x0 = kinematics_component.kinematics.design_vector
# model.set_val('x', kinematics_component.kinematics.design_vector)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 3

prob.run_model()

# Plot initial spatial configuration
# kinematics_component.kinematics_interface.plot()

# Perform gradient-based optimization

# print('Initial design vector: ', prob['x'])
# print('Initial objective: ', prob['f'])
# print('Initial constraint values: ', prob['g'])

# x0 = torch.tensor(prob['x'], dtype=torch.float64)
# objects_dict = kinematics_component.kinematics_interface.calculate_positions(x0)
# kinematics_component.kinematics_interface.set_positions(objects_dict)
# kinematics_component.kinematics_interface.plot()

# prob.()


print('Optimized design vector: ', prob['x'])
print('Optimized objective: ', prob['f'])
print('Optimized constraint values: ', prob['g'])

# Plot optimized spatial
xf = torch.tensor(prob['x'], dtype=torch.float64)
objects_dict = kinematics_component.kinematics.calculate_positions(xf)
kinematics_component.kinematics.set_positions(objects_dict)
kinematics_component.kinematics.plot()
