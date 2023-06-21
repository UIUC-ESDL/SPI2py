"""
Example 1:  Simple optimization of a 3D layout
Author:     Chad Peterson
"""

# Import packages
import os
import numpy as np
from SPI2py import System, DesignStudy

# Define the system

system = System(name='Demo System')


# from SPI2py.computational_model.geometry.geometric_representation import pseudo_mdbd
#
# pseudo_mdbd(2, 2, 2, 0.5, 0.5)

# TODO Major--reimplement collision detection for interconnect segments.



system.add_component(name='control_valve_1',
                     color='aquamarine',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [6, 2, 2], 'rotation': [0, 0, 0]}],
                     ports=[{'name': 'supply', 'origin': [2, 1, 2.5], 'radius': 0.5},
                            {'name': 'return', 'origin': [4, 1, 2.5], 'radius': 0.5}])

system.add_component(name='actuator_1',
                     color='orange',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [0, 0, 1.5], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [0, 0, 3], 'dimensions': [3, 3, 1.5], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 3.5], 'dimensions': [1, 1, 5], 'rotation': [0, 0, 0]}],
                     ports=[{'name': 'supply', 'origin': [1, 0, 1], 'radius': 0.5},
                            {'name': 'return', 'origin': [2, 0, 1], 'radius': 0.5}])

system.add_component(name='component_2',
                     color='indigo',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [1, 3, 3], 'rotation': [0, 0, 0]}])

system.add_component(name='component_3',
                     color='olive',
                     movement_class='independent',
                     degrees_of_freedom=('x', 'y', 'z', 'rx', 'ry', 'rz'),
                     shapes=[{'type': 'box', 'origin': [0, 0, 0], 'dimensions': [1, 1, 1], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 0, 0], 'dimensions': [1, 2, 1], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 0.5], 'dimensions': [1, 1, 3], 'rotation': [0, 0, 0]},
                             {'type': 'box', 'origin': [1, 1, 3], 'dimensions': [2, 1, 1], 'rotation': [0, 0, 0]}])

system.add_component(name='structure_1',
                     color='gray',
                     movement_class='static',
                     degrees_of_freedom=(),
                     shapes=[
                         {'type': 'box', 'origin': [0, 0, 0], 'dimensions': [2, 2, 0.5], 'rotation': [0, 0, 0]}])

system.add_interconnect(name='hp_cv_to_actuator',
                        color='black',
                        component_1='control_valve_1',
                        component_1_port='supply',
                        component_2='actuator_1',
                        component_2_port='supply',
                        radius=0.25,
                        number_of_bends=2,
                        degrees_of_freedom=(('x', 'y', 'z'), ('x', 'y', 'z')))

system.add_interconnect(name='hp_cv_to_actuator2',
                        color='blue',
                        component_1='control_valve_1',
                        component_1_port='return',
                        component_2='actuator_1',
                        component_2_port='return',
                        radius=0.25,
                        number_of_bends=1,
                        degrees_of_freedom=('x', 'y', 'z'))





# Define the design study

# Obtain the local path of this example's directory
local_directory = os.path.dirname(__file__) + '/'

# Initialize the design study
study = DesignStudy(directory=local_directory,
                    study_name='Example 1')

study.add_system(system)

# Define the username and problem description
study.config['Username'] = 'Chad Peterson'
study.config['Problem Description'] = 'Simple optimization of a 3D layout'

# Map the system to a single spatial configuration

# TODO replace add initial design vector to set_position
# TODO Set initial design vector, including the static object... enter as dict arguments to manual
# Specify for interconnect, have multiple waypoints
study.add_initial_design_vector('control_valve_1', 'spatial_config_1', [-3., -4.41, -0.24, 0., 0., 0.])
study.add_initial_design_vector('actuator_1', 'spatial_config_1', [2., 4.41, 0.24, 0., 0., 0.])
study.add_initial_design_vector('component_2', 'spatial_config_1', [5, -3, -1, 0., 0., 0.])
study.add_initial_design_vector('component_3', 'spatial_config_1', [-3., -1., 3., 0., 0., 0.])
study.add_initial_design_vector('hp_cv_to_actuator', 'spatial_config_1', [-3., -2., 2., -1., 0., 2.])
study.add_initial_design_vector('hp_cv_to_actuator2', 'spatial_config_1', [4., 0., 1.])

system.map_static_object(object_name='structure_1', design_vector=[0, 0, -1, 0, 0, 0])



study.generate_spatial_configuration(name='spatial_config_1', method='manual')



# Plot initial spatial configuration
system.plot()

# Perform gradient-based optimization

system.add_objective(objective='normalized aggregate gap distance',
                     options={'design vector scaling type': 'constant',
                              'design vector scaling factor': 1,
                              'objective scaling type': 'constant',
                              'objective scaling factor': 1 / 500})

system.add_constraint(constraint='signed distances',
                      options={'type': 'collision',
                               'object class 1': 'component',
                               'object class 2': 'component',
                               'constraint tolerance': 0.01,
                               'constraint aggregation': 'induced exponential',
                               'constraint aggregation parameter': 3.0})

system.add_constraint(constraint='signed distances',
                      options={'type': 'collision',
                               'object class 1': 'component',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.01,
                               'constraint aggregation': 'induced exponential',
                               'constraint aggregation parameter': 3.0})

system.add_constraint(constraint='signed distances',
                      options={'type': 'collision',
                               'object class 1': 'interconnect',
                               'object class 2': 'interconnect',
                               'constraint tolerance': 0.01,
                               'constraint aggregation': 'induced exponential',
                               'constraint aggregation parameter': 3.0})


# x0 = list(study.initial_design_vectors['spatial_config_1'].values())
# x0 = [item for sublist in x0 for item in sublist]
#
# obj, con = system.calculate_metrics(x0)
#
# print(obj)
# print(con)

# x = np.array([ 1.82515278,  2.00464052, -0.22418859, -1.43902471,  0.25068738,
#         0.71523626, -0.97297035, -3.46762245, -0.67264752,  1.98852344,
#        -0.56104854, -1.08517915, -3.17785899,  2.01990026,  0.30666861,
#        -0.90755342, -0.69943541,  1.22064721,  2.13678219,  0.5635011 ,
#        -1.5059975 , -0.0930473 ,  2.25286219, -0.74430083, -0.57502765,
#        -0.32839024,  0.72500935, -0.49210847, -0.55807707,  0.58709315,
#         1.57458206,  1.05384339,  1.15976194])
#
# obj, con = system.calculate_metrics(x)
#
# print(obj)
# print(con)
#
# pos_dict = system.calculate_positions(x)
# system.set_positions(pos_dict)
# system.plot()

study.optimize_spatial_configuration(options={'maximum number of iterations': 25,
                                              'convergence tolerance': 1e-3})

# Post-processing

# Plot the final spatial configuration
new_positions = system.calculate_positions(study.result.x)
system.set_positions(new_positions)
system.plot()

# Write output file
study.create_report()

# Print the log to see the optimization results and if any warnings or errors occurred
study.print_log()




# from scipy.optimize import minimize

# # Define the cache
# cache = {}

# def objective(x):
#     # Check if the state is already in the cache
#     if tuple(x) in cache:
#         state = cache[tuple(x)]
#     else:
#         # Evaluate the model and store the state in the cache
#         state = evaluate_model(x)
#         cache[tuple(x)] = state

#     # Calculate the objective function using the model state
#     obj = calculate_objective(state)

#     return obj

# def constraint(x):
#     # Check if the state is already in the cache
#     if tuple(x) in cache:
#         state = cache[tuple(x)]
#     else:
#         # Evaluate the model and store the state in the cache
#         state = evaluate_model(x)
#         cache[tuple(x)] = state

#     # Calculate the constraint function using the model state
#     con = calculate_constraint(state)

#     return con

# # Define the initial design vector
# x0 = [0, 0, 0]

# # Minimize the objective function subject to the constraint function
# res = minimize(objective, x0, constraints={'type': 'ineq', 'fun': constraint})