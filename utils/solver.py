"""Module...
...
"""

import jax.numpy as jnp
from jax import jit
from jax.numpy.linalg import norm
from scipy.optimize import minimize

from scipy.optimize import minimize, Bounds, NonlinearConstraint

from itertools import combinations


def objective_function(positions):
    """
    Temp-For now calculate the pairwise distances between all design vectors

    jit
    vectorize
    :param positions:
    :return:
    """

    pairwise_distance_pairs = list(combinations(positions,2))

    objective = 0
    for point1, point2 in pairwise_distance_pairs:

        pairwise_distance = norm(point2 - point1)

        objective += pairwise_distance

    return objective


def constrain_function_component_component(positions, radii):
    """
    ...
    Applies hierarchical collision detection to both components

    :param positions:
    :param radii:
    :return:
    """
    pass

def constrain_function_component_interconnect(positions, radii):
    """
    ...
    Applies hierarchical collision detection to component
    :param positions:
    :param radii:
    :return:
    """
    pass

def constrain_function_interconnect_interconnect(positions, radii):
    pass

def constrain_function_structure_all(positions, radii):
    pass

# Can I aggregate all constrains into one function
# With an array for lb/ub?
def constraint_function():
    pass

def solver():
    """
    Constrained optimization...

    There are plans to use the JAX-wrapped implementation of scipy.optimize.minimize;
    however, it currently only supports the BFGS method for unconstrained optimization.

    :return:
    """

    pass