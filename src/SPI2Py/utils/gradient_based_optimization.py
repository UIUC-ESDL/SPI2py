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
we slightly relax the interconnect-interconnect constraint. On the other hand, strictly enforcing interference
constraints for component-component doesn't seem to cause any problems.

In the future, if JAX supports adds native Windows support, or Numba adds support to Autograd,
we will also define the objective and constraint gradient functions here. Since the objective
and constraint functions call Numba-JIT-compiled functions, we can't apply algorithmic differentiation.
JAX allows this, but is only natively supported on Mac and Linux... many industry collaborators use Windows
and are not allowed to build from the source (or it just adds another level of difficulty).

"""

import numpy as np
from numba import njit

from scipy.optimize import minimize, Bounds, NonlinearConstraint

from src.SPI2Py.utils.objective_functions import aggregate_pairwise_distance

from src.SPI2Py.utils.constraint_functions import interference_component_component, interference_component_interconnect, \
    interference_interconnect_interconnect, interference_structure_all


def log_design_vector(xk, state):
    """
    Logs the design vector...

    Note: callback function args for trust-const method
    :return:
    """

    global design_vector_log

    design_vector_log.append(xk)


def optimize(layout):
    """
    Constrained optimization...

    There are plans to use the JAX-wrapped implementation of scipy.optimize.minimize;
    however, it currently only supports the BFGS method for unconstrained optimization.

    For now objective function and constraints are hard-coded into this solver. Future versions
    may seek more flexibility.
    :param layout:
    :param x0: Initial design vector
    :return:
    """

    global design_vector_log

    fun = aggregate_pairwise_distance
    x0 = layout.design_vector

    # TODO Implement bounds
    # bounds = Bounds()

    # NonlinearConstraint object for trust-constr method does not take kwargs
    # Use lambda functions to format constraint functions as needed
    # TODO clarify these aren't actually lambda functions...
    def lambda_interference_component_component(x): return interference_component_component(x, layout)
    def lambda_interference_component_interconnect(x): return interference_component_interconnect(x, layout)
    def lambda_interference_interconnect_interconnect(x): return interference_interconnect_interconnect(x, layout)
    def lambda_interference_structure_all(x): return interference_structure_all(x, layout)

    nlc_component_component = NonlinearConstraint(lambda_interference_component_component, -np.inf, 0.5)
    nlc_component_interconnect = NonlinearConstraint(lambda_interference_component_interconnect, -np.inf, 0.5)
    nlc_interconnect_interconnect = NonlinearConstraint(lambda_interference_interconnect_interconnect, -np.inf, 0.5)
    nlc_structure_all = NonlinearConstraint(lambda_interference_structure_all, -np.inf, 0.5)

    # nlcs = [nlc_component_component]
    nlcs = [nlc_component_component, nlc_component_interconnect]
    # nlcs = [nlc_component_component, nlc_component_interconnect, nlc_interconnect_interconnect, nlc_structure_all]

    # print('TEST', constraint_component_component(x0, layout))
    # nlc = NonlinearConstraint(con, -np.inf, -1)

    options = {}

    # Add initial value
    design_vector_log.append(x0)

    # TODO Evaluate different solver methods and parametric tunings

    res = minimize(fun, x0, args=layout, method='trust-constr', constraints=nlcs, tol=1e-5,
                   options=options, callback=log_design_vector)



    # Add final value
    design_vector_log.append(res.x)

    # For troubleshooting
    print('Constraint is', interference_component_component(res.x, layout))

    return res, design_vector_log


design_vector_log = []
