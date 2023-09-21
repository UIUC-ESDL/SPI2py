import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, approx_fprime, Bounds
import torch
from torch.autograd.functional import jacobian

np.random.seed(0)

# def objective(positions):
#     """Mock bounding box volume"""
#
#     positions = positions.reshape((-1, 3))
#
#     min_x = np.min(positions[:, 0])
#     max_x = np.max(positions[:, 0])
#     min_y = np.min(positions[:, 1])
#     max_y = np.max(positions[:, 1])
#     min_z = np.min(positions[:, 2])
#     max_z = np.max(positions[:, 2])
#
#     volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)
#
#     return volume
#
# def constraint(positions):
#     """Minimum distance between each point"""
#
#     positions = positions.reshape((-1, 3))
#
#     radius = 0.1
#     radii = radius * np.ones((positions.shape[0], 1))
#
#     position_pairs = []
#
#     return 0
#
# x0 = np.random.rand(9)


def objective(x):
    """Minimize the distance between each point"""

    f = 0
    for i in range(len(x)-1):
        f += (x[i] - x[i+1])**2

    return f

def constraint(x):

    g = []
    for i in range(len(x)-1):
        gi = 0.1 - (x[i] - x[i+1])**2
        g.append(gi)

    return g

# def constant(x, c):
#
#     h = []
#     for i in range(len(c)):
#         hi =

def run_minimize(x):

    num_constraints = len(x) - 1
    lb = -np.inf * np.ones(num_constraints)
    ub = np.zeros(num_constraints)
    nlcs = NonlinearConstraint(constraint, lb, ub)

    sol = minimize(objective, x, method='SLSQP', constraints=nlcs)

    return sol

def run_minimize_eq(x, num_eq_constraints):

    num_constraints = len(x) - 1
    lb = -np.inf * np.ones(num_constraints)
    ub = np.zeros(num_constraints)
    nlcs = NonlinearConstraint(constraint, lb, ub)

    c = np.arange(num_eq_constraints)
    eq_nlcs = NonlinearConstraint(lambda xs: [xs[i] for i in c], c, c)

    sol = minimize(objective, x, method='SLSQP', constraints=[nlcs, eq_nlcs])

    return sol

def run_minimize_bound(x, num_bounds):

    num_constraints = len(x) - 1
    glb = -np.inf * np.ones(num_constraints)
    gub = np.zeros(num_constraints)
    nlcs = NonlinearConstraint(constraint, glb, gub)

    c = np.arange(num_bounds)
    other = len(x) - num_bounds
    c_lb = c
    c_ub = c
    other_lb = -np.inf * np.ones(other)
    other_ub = np.inf * np.ones(other)

    lb = np.hstack((c_lb, other_lb))
    ub = np.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(objective, x, method='SLSQP', constraints=nlcs, bounds=bounds)

    return sol




x0 = np.random.rand(10)
x1 = np.random.rand(100)
x2 = np.random.rand(1000)
x3 = np.random.rand(10000)

# sol1 = run_minimize(x1)
#
# sol2 = run_minimize_eq(x1, 50)


def state_constraint(x):

    a = np.ones((1000, 1))

    return x * a


# dg = approx_fprime(x0, state_constraint, 1e-6)
# print('Shape:', dg.shape)

sol3 = run_minimize_bound(x1, 100)
