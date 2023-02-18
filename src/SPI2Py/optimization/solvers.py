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
from scipy.optimize import minimize, NonlinearConstraint
from ..analysis.objectives import aggregate_pairwise_distance
from ..analysis.constraints import max_interference


def run_optimizer(layout, config):
    """
    This is a helper function that runs the optimization solver.

    Wrapping the solver in a helper function provides a few key benefits:
    1. We can easily switch between different solvers.
    2. Variable scope; we want to log intermediate values and results such as design vectors.
    Solvers such as scipy.optimize.minimize and Matlab fmincon support callback/output functions,
    but these functions might not provide the specific information we want to log.

    Rather than declaring global variables (which is bad practice), we can use a nest the solver, callback,
    objective, and constraint functions inside of a helper function, which allows the callback function.
    See https://www.mathworks.com/help/optim/ug/output-functions.html for a nice example of this.

    Note: We still define portions of the objective and constraint functions outside of the helper function, which
    variable scope does not reach. In the future we may need to move these functions inside the helper function, or
    pass the results we want back to the helper objective/constraint function.

    Note: There are plans to use the JAX-wrapped implementation of scipy.optimize.minimize;
    however, it currently only supports the BFGS method for unconstrained optimization.

    Note: There are plans to develop SPI2py-specific solvers to provide more flexibility.

    TODO Implement design vector bounds (e.g., rotation angles should be between 0 and 2pi)

    :param layout:
    :param x0: Initial design vector
    :return:
    """

    layout = layout

    # Initialize the design vector log
    # Since the log_design_vector function is nested inside this function, it can append the variable
    design_vector_log = []

    def log_design_vector(xk, *argv):
        """
        Logs the design vector...

        argv is since difference solvers pass diff arguments but we only want to capture xk
        Note: callback function args for trust-const method
        :return:
        """

        design_vector_log.append(xk)

    fun = aggregate_pairwise_distance

    x0 = layout.design_vector

    nlcs = []



    # NonlinearConstraint object for trust-constr method does not take kwargs
    # Use lambda functions to format constraint functions as needed with kwargs
    # nlc_cc = NonlinearConstraint(lambda x: max_interference(x, layout, layout.component_component_pairs), -np.inf, -3)
    # nlc_ci = NonlinearConstraint(lambda x: max_interference(x, layout, layout.component_interconnect_pairs), -np.inf, 0)
    # nlc_cs = NonlinearConstraint(lambda x: max_interference(x, layout, layout.component_structure_pairs), -np.inf, 0)
    # nlc_ii = NonlinearConstraint(lambda x: max_interference(x, layout, layout.interconnect_interconnect_pairs), -np.inf, 0)
    # nlc_is = NonlinearConstraint(lambda x: max_interference(x, layout, layout.interconnect_structure_pairs), -np.inf, 0)
    nlcs = []
    # if config['detect collision']['components and components'] is True:
    #     nlcs.append(nlc_cc)
    # if config['detect collision']['components and interconnects'] is True:
    #     nlcs.append(nlc_ci)
    # if config['detect collision']['components and structures'] is True:
    #     nlcs.append(nlc_cs)
    # if config['detect collision']['interconnects and interconnects'] is True:
    #     nlcs.append(nlc_ii)
    # if config['detect collision']['interconnects and structures'] is True:
    #     nlcs.append(nlc_is)

    for object_pair, check_collision, collision_tolerance in zip(layout.object_pairs,layout.object_pairs, config['detect collision'].values()):
        if check_collision is True:
            nlc = NonlinearConstraint(lambda x: max_interference(x, layout, object_pair), -np.inf, collision_tolerance)
            nlcs.append(nlc)

    options = {}

    # Add initial value
    design_vector_log.append(x0)

    # TODO Evaluate different solver methods and parametric tunings

    res = minimize(lambda x: fun(x, layout), x0,
                   method='trust-constr',
                   constraints=nlcs,
                   tol=1e-2,
                   options=options,
                   callback=log_design_vector)

    # Add final value
    design_vector_log.append(res.x)

    # For troubleshooting
    print('Constraint is', max_interference(res.x, layout, layout.component_component_pairs))


    return res, design_vector_log

