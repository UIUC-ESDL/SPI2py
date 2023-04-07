"""Scaling

This module contains functions that are used to scale the objective and constraint functions as
well as the design vector.

MDO BOOK Chapter 4 Tip 4.4

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


def scale_objective(x, model, objective_function, config):
    """
    Scales the objective function by the maximum value of the objective function

    :param objective_function: The objective function to scale
    :param layout: The layout object
    :param config: The configuration object
    :return: The scaled objective function
    """

    # Create a scaled objective function
    def scaled_objective(x, layout, config):
        return objective_function(x, layout, config) / max_objective

    return scaled_objective