"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import torch
import openmdao.api as om
from SPI2py import ComponentInterface, Interconnect, SystemInterface
from SPI2py.group_model.utilities.visualization import plot

# Initialize the problem
prob = om.Problem()
model = prob.model

# Initialize the System
# ...



# Initialize components and add them to the system


# system.add_subsystem()

# Initialize interconnects and add them to the system
# system.add_subsystem()

# Connect the components and interconnects as necessary
# system.connect(...)
# system.connect(...)

# Connect




# Define the system

system = System('input.toml')

translations_0 = torch.tensor([[2, 7, 0],
                               [5.5, 7, 0],
                               [0.5, 6, 0],
                               [7.5, 6, 0],
                               [0.5, 4.5, 0],
                               [7.5, 5, 0],
                               [0.5, 2, 0],
                               [7.5, 3, 0],
                               [6, 1, 0],
                               [2, 0, 0]], dtype=torch.float64)

rotations_0 = torch.tensor([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 0]], dtype=torch.float64)

routings_0 = torch.tensor([[[3.475, 7, 0], [3.75, 7, 0], [4.025, 7, 0]],
                           [[6.975, 7, 0], [7.5, 7, 0], [7.5, 6.275, 0]],
                           [[7.5, 5.725, 0], [7.5, 5.47, 0], [7.5, 5.215, 0]],
                           [[7.5, 4.785, 0], [7.5, 4.3175, 0], [7.5, 3.85, 0]],
                           [[7.5, 2.15, 0], [6.5, 2, 0], [6.0, 1.9665, 0]],
                           [[5.1225, 1.0, 0], [3, 0, 0], [2.7225, 0, 0]],
                           [[1.2775, 0, 0], [0.5, 0, 0], [0.5, 1.15, 0]],
                           [[0.5, 2.85, 0], [0.5, 3, 0], [0.5, 4.285, 0]],
                           [[0.5, 4.715, 0], [0.5, 5, 0], [0.5, 5.725, 0]],
                           [[0.5, 6.275, 0], [0.5, 6.5, 0], [0.525, 7, 0]]], dtype=torch.float64)

# routings_0 = torch.zeros((10, 3, 3), dtype=torch.float64)

# from torch.autograd.functional import jacobian
# translations = torch.tensor(translations_0, dtype=torch.float64, requires_grad=True)
# rotations = torch.tensor(rotations_0, dtype=torch.float64, requires_grad=True)
# routings = torch.tensor(routings_0, dtype=torch.float64, requires_grad=True)
#
# jac_f = jacobian(system.calculate_objective, (translations, rotations, routings))
# jac_g = jacobian(system.calculate_constraints, (translations, rotations, routings))



kinematics_component = KinematicsInterface()
kinematics_component.options.declare('kinematics', default=system)

model.add_subsystem('kinematics', kinematics_component, promotes=['*'])

prob.model.add_design_var('translations', ref=10)
prob.model.add_design_var('rotations', ref=2*torch.pi)

# TODO Unbound routing
prob.model.add_design_var('routings', ref=10, lower=routings_0, upper=routings_0)
prob.model.add_objective('f')
prob.model.add_constraint('g', upper=0)
# prob.model.add_constraint('g_c', upper=0)
# prob.model.add_constraint('g_i', upper=0)
# prob.model.add_constraint('g_ci', upper=0)

prob.setup()

prob.set_val('translations', translations_0)
prob.set_val('rotations', rotations_0)
prob.set_val('routings', routings_0)


# TODO Equality bound all the routing variables

prob.run_model()

# Plot initial spatial configuration
print('Initial objective: ', prob['f'])
print('Initial constraint values: ', prob['g'])
translations_i = prob['translations']
rotations_i = prob['rotations']
routings_i = prob['routings']
translations_i = torch.tensor(translations_i, dtype=torch.float64)
rotations_i = torch.tensor(rotations_i, dtype=torch.float64)
routings_i = torch.tensor(routings_i, dtype=torch.float64)

plot_objects, colors = kinematics_component.kinematics.plot_inputs(translations_i, rotations_i, routings_i)
plot(plot_objects, colors)

#
# # Run the optimization
#
# prob.driver = om.ScipyOptimizeDriver()
# # prob.driver.options['maxiter'] = 50
#
# prob.run_driver()
#
#
# # Plot optimized spatial
# print('Optimized objective: ', prob['f'])
# print('Optimized constraint values: ', prob['g'])
# translations_f = prob['translations']
# rotations_f = prob['rotations']
# routings_f = prob['routings']
# translations_f = torch.tensor(translations_f, dtype=torch.float64)
# rotations_f = torch.tensor(rotations_f, dtype=torch.float64)
# routings_f = torch.tensor(routings_f, dtype=torch.float64)
#
# plot_objects_f, colors_f = kinematics_component.kinematics.plot_inputs(translations_f, rotations_f, routings_f)
# plot(plot_objects_f, colors_f)


