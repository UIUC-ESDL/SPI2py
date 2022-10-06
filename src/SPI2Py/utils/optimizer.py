# """Module...
# ...
#
#
# Scrub all JAX
# """
#
# # import jax.numpy as jnp
# # from jax import jit, grad, jacfwd
# # from jax.numpy.linalg import norm
#
# from scipy.optimize import minimize, Bounds, NonlinearConstraint
#
# from itertools import combinations
#
#
# def objective(positions):
#     """
#     Temp-For now calculate the pairwise distances between all design vectors
#
#     Given a flat list and reshape
#     jit
#     vectorize
#     :param positions:
#     :return:
#     """
#
#     # Fix this comment for actual reshape...
#     # Reshape flattened design vector from 1D to 2D
#     # [x1,y1,z1,x2,... ] to [[x1,y1,z1],[x2... ]
#     positions = positions.reshape(-1,3)
#
#
#     #
#     pairwise_distance_pairs = list(combinations(positions,2))
#
#     objective = 0
#     for point1, point2 in pairwise_distance_pairs:
#
#         pairwise_distance = norm(point2 - point1)
#
#         objective += pairwise_distance
#
#     # Do I need to return a scalar...?
#     return objective
#
# def objective_jacobian(x):
#     return grad(objective)(x)
#
#
# def objective_hessian(x):
#     return jacfwd(grad(objective))(x)
#
#
#
#
#
# def constraint_component_component(positions, radii):
#     """
#     ...
#     Applies hierarchical collision detection to both components
#
#     :param positions:
#     :param radii:
#     :return:
#     """
#     pass
#
# def constraint_component_component_jacobian(x):
#     return grad(constraint_component_component)(x)
#
# def constraint_component_component_hessian(x):
#     return jacfwd(constraint_component_component_jacobian)(x)
#
# def constraint_component_interconnect(positions, radii):
#     """
#     ...
#     Applies hierarchical collision detection to component
#     :param positions:
#     :param radii:
#     :return:
#     """
#     pass
#
# def constraint_component_interconnect_jacobian(x):
#     return grad(constraint_component_interconnect)(x)
#
# def constraint_component_interconnect_hessian(x):
#     return jacfwd(constraint_component_interconnect_jacobian)(x)
#
# def constraint_interconnect_interconnect(positions, radii):
#     pass
#
# def constraint_interconnect_interconnect_jacobian(x):
#     return grad(constraint_interconnect_interconnect)(x)
#
# def constraint_interconnect_interconnect_hessian(x):
#     return jacfwd(constraint_interconnect_interconnect_jacobian)(x)
#
# def constraint_structure_all(positions, radii):
#     pass
#
# def constraint_structure_all_jacobian(x):
#     return grad(constraint_structure_all)(x)
#
# def constraint_structure_all_hessian(x):
#     return jacfwd(constraint_structure_all_jacobian)(x)
#
# def optimize(fun, x0):
#     """
#     Constrained optimization...
#
#     There are plans to use the JAX-wrapped implementation of scipy.optimize.minimize;
#     however, it currently only supports the BFGS method for unconstrained optimization.
#
#     For now objective function and constraints are hard-coded into this solver. Future versions
#     may seek more flexibility.
#     :return:
#     """
#
#     # Initialize the objective function and its gradient information
#     fun = objective
#     jac = objective_jacobian
#     hess = objective_hessian
#
#     # Define the bounds of the design vector
#     # bounds = Bounds()
#
#     # Initialize the geometric constraints
#     nlc_component_component = NonlinearConstraint(constraint_component_component,-jnp.inf,0.5,
#                                                   jac=constraint_component_component_jacobian,
#                                                   hess=constraint_component_component_hessian)
#
#     nlc_component_interconnect = NonlinearConstraint(constraint_component_interconnect,-jnp.inf,0.5,
#                                                      jac=constraint_component_interconnect_jacobian,
#                                                      hess=constraint_component_interconnect_hessian)
#
#     nlc_interconnect_interconnect = NonlinearConstraint(constraint_interconnect_interconnect,-jnp.inf,0.5,
#                                                         jac=constraint_interconnect_interconnect_jacobian,
#                                                         hess=constraint_interconnect_interconnect_hessian)
#
#     nlc_structure_all = NonlinearConstraint(constraint_structure_all,-jnp.inf,0.5,
#                                             jac=constraint_structure_all_jacobian,
#                                             hess=constraint_structure_all_hessian)
#
#     nlcs = [nlc_component_component, nlc_component_interconnect, nlc_interconnect_interconnect, nlc_structure_all]
#
#
#
#
#     options = {}
#
#     res = minimize(fun, x0, method='trust-constr', constraints = nlcs, jac=jac, hess=hess)
#
#     return res
#
#
#
#
#
#
