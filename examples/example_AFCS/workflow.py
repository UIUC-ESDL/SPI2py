"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

import torch
import openmdao.api as om
from SPI2py import KinematicsInterface, Component, Interconnect, System

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

# positions_dict = system.calculate_positions(translations_0, rotations_0, routings_0)
# system.set_positions(positions_dict)

# Define the OpenMDAO Interface

prob = om.Problem()
model = prob.model

kinematics_component = KinematicsInterface()
kinematics_component.options.declare('kinematics', default=system)

model.add_subsystem('kinematics', kinematics_component, promotes=['*'])

prob.model.add_design_var('translations', lower=-10, upper=10)
prob.model.add_design_var('rotations', lower=-2 * torch.pi, upper=2 * torch.pi)
prob.model.add_design_var('routings', lower=-10, upper=10)
prob.model.add_objective('f')
prob.model.add_constraint('g', upper=0)

prob.setup()

prob.set_val('translations', translations_0)
prob.set_val('rotations', rotations_0)
prob.set_val('routings', routings_0)

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
kinematics_component.kinematics.plot(translations_i, rotations_i, routings_i)


# Run the optimization

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['maxiter'] = 3



#
# # x0 = torch.tensor(prob['x'], dtype=torch.float64)
# # objects_dict = kinematics_component.kinematics_interface.calculate_positions(x0)
# # kinematics_component.kinematics_interface.set_positions(objects_dict)
# # kinematics_component.kinematics_interface.plot()
#
# # prob.run_driver()
#
# # TODO Take initial argument for positions as ports...?

print('Optimized objective: ', prob['f'])
print('Optimized constraint values: ', prob['g'])

# Plot optimized spatial
# xf = torch.tensor(prob['x'], dtype=torch.float64)
# objects_dict = kinematics_component.kinematics.calculate_positions(xf)
# kinematics_component.kinematics.set_positions(objects_dict)
# kinematics_component.kinematics.plot()
