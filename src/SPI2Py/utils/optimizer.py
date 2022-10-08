"""Optimizer

Implements the nonlinear programming solver.

Additionally, defines the objective and constraint functions.

Instead of specifying one constraint function for interference, we split it up into
    1. component-component
    2. component-interconnect
    3. interconnect-interconnect
    4. structure-all
This provides more flexibility. For instance, depending on the level of fidelity, it can be hard to prevent two
interconnects that connect to the same component, from overlapping at the starting point. If you enforce a
zero-tolerance interference, then the solver will never be able to avoid collision and will get all confused. So
we slightly relaxthe interconnect-interconnect constraint. On the other hand, strictly enforcing interference
constraints for component-component doesn't seem to cause any problems.

In the future, if JAX supports adds native Windows support, or Numba adds support to Autograd,
we will also define the objective and constraint gradient functions here. Since the objective
and constraint functions call Numba-JIT-compiled functions, we can't apply algorithmic differentiation.
JAX allows this, but is only natively supported on Mac and Linux... many industry collaborators use Windows
and are not allowed to build from the source (or it just adds another level of difficulty).

"""

import numpy as np
from numba import njit
from itertools import combinations
from scipy.optimize import minimize, Bounds, NonlinearConstraint


def objective(positions):
    """
    Temp-For now calculate the pairwise distances between all design vectors

    Given a flat list and reshape
    jit
    vectorize
    :param positions:
    :return:
    """

    # Fix this comment for actual reshape...
    # Reshape flattened design vector from 1D to 2D
    # [x1,y1,z1,x2,... ] to [[x1,y1,z1],[x2... ]
    positions = positions.reshape(-1, 3)

    #
    pairwise_distance_pairs = list(combinations(positions, 2))

    objective = 0
    for point1, point2 in pairwise_distance_pairs:
        pairwise_distance = np.linalg.norm(point2 - point1)

        objective += pairwise_distance

    # Do I need to return a scalar...?
    return objective


def constraint_component_component(positions, radii):
    """
    ...
    Applies hierarchical collision detection to both components

    TODO Write this function

    :param positions:
    :param radii:
    :return:
    """
    pass


def constraint_component_interconnect(positions, radii):
    """

    TODO Write this function

    Applies hierarchical collision detection to component
    :param positions:
    :param radii:
    :return:
    """
    pass


def constraint_interconnect_interconnect(positions, radii):
    # TODO Write this function
    pass


def constraint_structure_all(positions, radii):
    # TODO Write this function
    pass


def optimize(x0):
    """
    Constrained optimization...

    There are plans to use the JAX-wrapped implementation of scipy.optimize.minimize;
    however, it currently only supports the BFGS method for unconstrained optimization.

    For now objective function and constraints are hard-coded into this solver. Future versions
    may seek more flexibility.
    :param x0: Initial design vector
    :return:
    """

    # Initialize the objective function and its gradient information
    fun = objective

    # Define the bounds of the design vector
    # TODO Implement bounds
    # bounds = Bounds()

    # Initialize the geometric constraints
    nlc_component_component       = NonlinearConstraint(constraint_component_component, -np.inf, 0.5)
    nlc_component_interconnect    = NonlinearConstraint(constraint_component_interconnect, -np.inf, 0.5)
    nlc_interconnect_interconnect = NonlinearConstraint(constraint_interconnect_interconnect, -np.inf, 0.5)
    nlc_structure_all             = NonlinearConstraint(constraint_structure_all, -np.inf, 0.5)

    nlcs = [nlc_component_component, nlc_component_interconnect, nlc_interconnect_interconnect, nlc_structure_all]

    # TODO Implement options
    options = {}

    # TODO Evaluate different solver methods and parametric tunings
    # TODO Check how I pass constraints argument as list instead of dict
    res = minimize(fun, x0, method='trust-constr', constraints=nlcs)

    return res

#
#
#
#
#
