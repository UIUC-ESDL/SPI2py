"""Constraint Aggregation Functions

This module contains functions for aggregating constraint values. Constraint functions for signed distances yield
or thousands of constraints. These functions are used to aggregate the constraint values into a single value in a
way that is differentiable and continuous.
"""

import jax.numpy as np


def kreisselmeier_steinhauser(g, rho=100):
    """
    The Kreisselmeier-Steinhauser (KS) method for constraint aggregation.

    This alternate implementation of the KS
    method is based on MDO book Chapter 5.7 to avoid overflow from the exponential function disproportionally
    weighing higher positive values.

    While this algorithm uses the recommended default of rho=100, it may need to be evaluated for each problem.
    As rho increases, so does the curvature of the constraint aggregation function, which can cause ill-conditioning.

    :param g: A row vector of constraint values.
    :param rho: Aggregation parameter.
    :return:
    """

    g_bar_ks = np.max(g) + 1/rho * np.log(np.sum(np.exp(rho * (g - np.max(g)))))

    return g_bar_ks




