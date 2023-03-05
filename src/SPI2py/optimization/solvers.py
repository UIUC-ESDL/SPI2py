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
import logging
logger = logging.getLogger(__name__)


def run_optimizer(layout, objective_function, constraint_function, config):
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

    Notes:
    1. There are plans to develop SPI2py-specific solvers to provide more flexibility.
    2. Since the current objective and constraint functions may be non-smooth / linear we define the
    Hessian ("hessp") as zero.
    3. I originally wrote a for loop to create each NonlinearConstraint object. However, SciPy Minimize completely
    ignores the constraints. Inexplicably, manually defining the NonlinearConstraint objects sequentially works and
    SciPy Minimize enforces the constraints. I use the exact same code, checked that all values are the same, and that
    NonlinearConstraint objects created inside and outside of a loop have the same attributes and the methods produce
    the same results. For example:
    >>> for i in range(3):
    >>>    print(i)
    Should produce the same exact result as:
    >>> print(0)
    >>> print(1)
    >>> print(2)
    But in the case of instantiating NonlinearConstraint objects, it does not. I'm not going to waste any more time
    trying to figure out why this is happening. I'm just going to manually define the NonlinearConstraint objects.

    TODO Implement design vector bounds (e.g., rotation angles should be between 0 and 2pi)
    TODO Evaluate other solvers and methods

    :param layout:
    :param objective_function:
    :param constraint_function:
    :param config:
    :return:
    """

    def log_design_vector(xk, *argv):
        """
        Logs the design vector...

        argv is since difference solvers pass diff arguments, but we only want to capture xk
        Note: callback function args for trust-const method
        :return:
        """

        design_vector_log.append(xk)

    # Initialize the design vector log
    # Since the log_design_vector function is nested inside this function, it can append the variable
    design_vector_log = []

    # Unpack the parameters
    x0 = layout.design_vector
    object_pairs = layout.system.object_pairs

    # Unpack the config dictionary
    convergence_tolerance = float(config['convergence tolerance'])
    check_collisions      = list(config['check collisions'].values())
    collision_tolerances  = list(config['collision tolerance'].values())

    # Add the applicable interference constraints
    nlcs = []
    if check_collisions[0] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_function(x, layout, object_pairs[0]), -np.inf,
                                        collision_tolerances[0]))
    if check_collisions[1] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_function(x, layout, object_pairs[1]), -np.inf,
                                        collision_tolerances[1]))
    if check_collisions[2] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_function(x, layout, object_pairs[2]), -np.inf,
                                        collision_tolerances[2]))
    if check_collisions[3] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_function(x, layout, object_pairs[3]), -np.inf,
                                        collision_tolerances[3]))
    if check_collisions[4] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_function(x, layout, object_pairs[4]), -np.inf,
                                        collision_tolerances[4]))

    options = {'verbose': 3}

    # Run the solver
    res = minimize(lambda x: objective_function(x, layout), x0,
                   method='trust-constr',
                   constraints= nlcs,
                   tol=convergence_tolerance,
                   options=options,
                   callback=log_design_vector,
                   hess=lambda x: np.zeros((len(x0), len(x0))))

    # Log the initial and final design vector
    design_vector_log.insert(0, x0)
    design_vector_log.append(res.x)

    # Log the results
    logger.info('Optimization results: {}'.format(res))

    return res, design_vector_log

