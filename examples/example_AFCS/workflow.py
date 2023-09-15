"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# %% Import packages

import torch
import openmdao.api as om
from SPI2py import KinematicsInterface, Kinematics, Component, Interconnect  # , System

# %% Define the kinematics model

import tomli

with open("input.toml", mode="rb") as fp:
    config = tomli.load(fp)

components_inputs = config['components']
conductors_inputs = config['conductors']

components = []
for component_inputs in components_inputs.items():
    name = component_inputs[0]
    component_inputs = component_inputs[1]
    color = component_inputs['color']
    degrees_of_freedom = component_inputs['degrees_of_freedom']
    filepath = component_inputs['filepath']
    ports = component_inputs['ports']
    components.append(Component(name=name, color=color, degrees_of_freedom=degrees_of_freedom, filepath=filepath, ports=ports))

conductors = []
for conductor_inputs in conductors_inputs.items():
    name = conductor_inputs[0]
    conductor_inputs = conductor_inputs[1]
    component_1 = conductor_inputs['component_1']
    component_1_port = conductor_inputs['component_1_port']
    component_2 = conductor_inputs['component_2']
    component_2_port = conductor_inputs['component_2_port']
    radius = conductor_inputs['radius']
    color = conductor_inputs['color']
    linear_spline_segments = conductor_inputs['linear_spline_segments']
    degrees_of_freedom = conductor_inputs['degrees_of_freedom']
    conductors.append(Interconnect(name=name, component_1=component_1, component_1_port=component_1_port, component_2=component_2, component_2_port=component_2_port, radius=radius, color=color, linear_spline_segments=linear_spline_segments, degrees_of_freedom=degrees_of_freedom))


kinematics = Kinematics(components=components,
                        interconnects=conductors,
                        objective='bounding box volume')

# %% Define the initial spatial configuration

with open("spatial_configurations.toml", mode="rb") as fp:
    default_positions_dict = tomli.load(fp)


kinematics.set_default_positions(default_positions_dict)

# %% Define the system

prob = om.Problem()
model = prob.model

kinematics_component = KinematicsInterface()
kinematics_component.options.declare('kinematics', default=kinematics)

model.add_subsystem('kinematics', kinematics_component, promotes=['*'])

prob.model.add_design_var('x')
prob.model.add_objective('f')
prob.model.add_constraint('g', upper=0)

prob.setup()


# %% Run the optimization


# x0 = kinematics_component.kinematics.design_vector
# model.set_val('x', kinematics_component.kinematics.design_vector)

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 300

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

# prob.run_driver()


print('Optimized design vector: ', prob['x'])
print('Optimized objective: ', prob['f'])
print('Optimized constraint values: ', prob['g'])

# Plot optimized spatial
xf = torch.tensor(prob['x'], dtype=torch.float64)
objects_dict = kinematics_component.kinematics.calculate_positions(xf)
kinematics_component.kinematics.set_positions(objects_dict)
kinematics_component.kinematics.plot()
