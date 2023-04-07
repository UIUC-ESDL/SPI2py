"""Scaling

This module contains functions that are used to scale the objective and constraint functions as
well as the design vector.

MDO BOOK Chapter 4 Tip 4.4

and Optimization in Practice with MATLAB

Scaling Options:
-Constant Scaling
-Logarithmic Scaling

TODO
-Add option to scale the design vector based on individual variable sensitivities (e.g., translation vs rotation)

Diagram of Scaling Process:

       x_0                                                x^*
        |                                                 ^
        v                                                 |
[x_0 element-wise division s_x]       [x_bar^* element-wise multiplication s_x]
        |                                                 ^
        v                                                 |
    x_bar_0                                             x_bar^*
        |                                                 |
        v                                                 |
[==============================Optimizer======================================]
        |                                                 ^
        v                                                 |
     x_b ar                                             f_bar
        |                                                 ^
        v                                                 |
[x_bar element-wise multiplication s_x]            [f_bar / s_f]
        |                                                 ^
        v                                                 |
        x                                                 f
        |                                                 ^
        v                                                 |
[==============================Model======================================]

"""

import numpy as np


def scale_design_vector(x,
                        design_vector_scale_factor=1,
                        design_vector_scale_type='constant'):
    """
    Scale the design vector

    TODO Add options for scaling types.

    :param x: The design vector
    :param design_vector_scale_factor: The scale factor for the design vector
    :param design_vector_scale_type: The type of scaling for the design vector
    :return: The scaled design vector
    """

    design_vector_scale_factor_vector = np.ones(len(x)) * design_vector_scale_factor
    scaled_design_vector = x * design_vector_scale_factor_vector

    return scaled_design_vector

def unscale_design_vector(x,
                          design_vector_scale_factor=1,
                          design_vector_scale_type='constant'):
    """
    Unscale the design vector

    TODO Add options for scaling types.

    :param x: The design vector
    :param design_vector_scale_factor: The scale factor for the design vector
    :param design_vector_scale_type: The type of scaling for the design vector
    :return: The unscaled design vector
    """

    design_vector_scale_factor_vector = np.ones(len(x)) * design_vector_scale_factor
    unscaled_design_vector = x / design_vector_scale_factor_vector

    return unscaled_design_vector

def scale_model_based_objective(x,
                                model_based_objective_function,
                                model,
                                design_vector_scale_factor=1,
                                design_vector_scale_type='constant',
                                objective_scale_factor=1,
                                objective_scale_type='constant'):
    """
    A function that applies scaling for the objective function of a model-based simulation.

    This function wraps the given objective function. First, it unscales the design vector input.
    Second, it evaluates the objective function. Third scales the objective function output.

    TODO Add options for scaling types.

    :param x: The design vector
    :param model_based_objective_function: The objective function to scale
    :param model: The model object
    :param design_vector_scale_factor: The scale factor for the design vector
    :param design_vector_scale_type: The type of scaling for the design vector
    :param objective_scale_factor: The scale factor for the objective function
    :param objective_scale_type: The type of scaling for the objective function
    :return: The scaled objective function
    """

    # Unscale the design vector
    unscaled_design_vector = unscale_design_vector(x,
                                                   design_vector_scale_factor=design_vector_scale_factor,
                                                   design_vector_scale_type=design_vector_scale_type)

    # Evaluate the objective function
    unscaled_objective = model_based_objective_function(unscaled_design_vector, model)

    # Scale the objective function
    scaled_objective = unscaled_objective * objective_scale_factor

    return scaled_objective


# def scale_model_based_constraints(x,
#                                   model,
#                                   constraint_functions,
#                                   config):
#     """
#     Scales the constraint functions by the maximum value of the constraint functions
#
#     :param constraint_functions: The constraint functions to scale
#     :param layout: The layout object
#     :param config: The configuration object
#     :return: The scaled constraint functions
#     """
#
#     # Create a scaled constraint function
#     def scaled_constraint(x, layout, config):
#         return constraint(x, layout, config) / max_constraint
#
#     return scaled_constraint

