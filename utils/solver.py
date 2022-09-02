"""Module...
...
"""

import jax.numpy as jnp
from jax import jit, grad, jacfwd
from jax.numpy.linalg import norm

from scipy.optimize import minimize, Bounds, NonlinearConstraint

from itertools import combinations


def objective_function(positions):
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
    positions = positions.reshape(-1,3)


    #
    pairwise_distance_pairs = list(combinations(positions,2))

    objective = 0
    for point1, point2 in pairwise_distance_pairs:

        pairwise_distance = norm(point2 - point1)

        objective += pairwise_distance

    # Do I need to return a scalar...?
    return objective

def objective_function_gradient(x):

    return grad(objective_function)(x)


def hessian(x):
    return jacfwd(grad(objective_function))(x)



# x0 = jnp.array([[1., 2., 3.], [4., 3., 1.], [0., 5., 2.]])
# x0 = x0.reshape(-1)
#
#
# def f(x): return x ** 2
#
# def g(x): return grad(f)(x)
#
# def h(x): return jacfwd(g)(x)
#
# ans = objective_function(x0 )
# print(ans)




def constraint_function():

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

    pass

# for testability take objective and constraint functions as args too
def solver(fun, x0, con, jac, hessian):
    """
    Constrained optimization...

    There are plans to use the JAX-wrapped implementation of scipy.optimize.minimize;
    however, it currently only supports the BFGS method for unconstrained optimization.

    :return:
    """

    # def fun(x):
    #     return objective_function(x), objective_function_gradient(x)

    # fun = (objective_function, objective_function_gradient)


    # bounds = Bounds()

    res = minimize(fun, x0, method='trust-constr', jac=jac, hess=hessian)

    return res






