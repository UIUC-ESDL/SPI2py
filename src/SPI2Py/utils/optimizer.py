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

from src.SPI2Py.utils.objective_functions import objective_1

from src.SPI2Py.utils.constraint_functions import constraint_component_component, constraint_component_interconnect, \
    constraint_interconnect_interconnect, constraint_structure_all


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

    fun = objective_1
    x0 = layout.design_vector

    # scipy minimize does not allow us to pass kwargs to constraint
    # Add ticket to allow this?
    con = lambda x: constraint_component_component(x, layout)
    nlc = NonlinearConstraint(con, -np.inf, 0.5)

    # bounds = Bounds() TODO Implement bounds

    # nlc_component_component = NonlinearConstraint(constraint_component_component, -np.inf, 0.5)
    # nlc_component_interconnect = NonlinearConstraint(constraint_component_interconnect, -np.inf, 0.5)
    # nlc_interconnect_interconnect = NonlinearConstraint(constraint_interconnect_interconnect, -np.inf, 0.5)
    # nlc_structure_all = NonlinearConstraint(constraint_structure_all, -np.inf, 0.5)

    # nlcs = [nlc_component_component, nlc_component_interconnect, nlc_interconnect_interconnect, nlc_structure_all]

    # print('TEST', constraint_component_component(x0, layout))

    # TODO Implement options
    options = {}

    # TODO Evaluate different solver methods and parametric tunings
    # TODO Check how I pass constraints argument as list instead of dict

    res = minimize(fun, x0, args=layout, method='trust-constr', constraints=nlc)

    # res = minimize(fun, x0, method='trust-constr', constraints=nlcs)

    # res = minimize(fun, x0, args=layout, method='trust-constr',constraints=nlc_component_component)
    #
    # return res

    return res

#
#
#
#
#
