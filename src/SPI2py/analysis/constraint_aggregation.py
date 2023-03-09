"""Constraint Aggregation Functions

This module contains functions for aggregating constraint values. Constraint functions for signed distances yield
or thousands of constraints. These functions are used to aggregate the constraint values into a single value in a
way that is differentiable and continuous.
"""

import numpy as np


def kreisselmeier_steinhauser(g, rho=100):
    """
    The Kreisselmeier-Steinhauser (KS) method for constraint aggregation.

    This alternate implementation of the KS
    method is based on MDO book Chapter 5.7 to avoid overflow from the exponential function disproportionally
    weighing higher positive values.

    While this algorithm uses the recommended default of rho=100, it may need to be evaluated for each problem.
    As rho increases, so does the curvature of the constraint aggregation function, which can cause ill-conditioning.

    TODO Validate the KS method for constraint aggregation.

    :param g: A row vector of constraint values.
    :param rho: Aggregation parameter.
    :return:
    """

    g_bar_ks = np.max(g) + 1/rho * np.log(np.sum(np.exp(rho * (g - np.max(g)))))

    return g_bar_ks


def p_norm(g, rho=3):
    """
    The p-norm method for constraint aggregation.

    This method should currently be avoided since it uses an absolute value function, which is not differentiable.
    It is included here for completeness and for future consideration.

    TODO Validate the p-norm method for constraint aggregation.

    :param g: A row vector of constraint values.
    :param rho: Aggregation parameter.
    :return:
    """

    g_bar_pn = np.max(np.abs(g)) * np.sum(np.abs(g/np.max(g))**rho)**(1/rho)

    return g_bar_pn


def induced_exponential_function(g, rho=3):
    """
    The induced exponential function method for constraint aggregation.

    TODO Implement the induced exponential function method for constraint aggregation.

    :param g: A row vector of constraint values.
    :param rho: Aggregation parameter.
    :return:
    """

    g_bar_ie = np.sum(g*np.exp(rho*g)) / np.sum(np.exp(rho*g))

    return g_bar_ie


def induced_power_function(g, rho=3):
    """
    The induced power function method for constraint aggregation.

    TODO Validate the induced power function method for constraint aggregation.
    TODO Add input constrains (i.e.,  g_j >= 0)

    :param g: A row vector of constraint values.
    :param rho: Aggregation parameter.
    :return:
    """

    g_bar_ipf = np.sum(g**(rho+1)) / np.sum(g**rho)

    return g_bar_ipf


