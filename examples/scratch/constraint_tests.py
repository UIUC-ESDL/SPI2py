import numpy as np
import torch
from torch.autograd.functional import jacobian
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, approx_fprime, Bounds


def f(x):
    """Minimize the distance between each point"""

    x = torch.tensor(x, dtype=torch.float64)

    return torch.sum((x[1:None] - x[0:-1])**2)

def f_jac(x):

    x = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    def fi(xi): return torch.sum((xi[1:None] - xi[0:-1])**2)

    f_val = fi(x)
    grad_f_val = jacobian(fi, x)

    f_val = f_val.detach().numpy()
    grad_f_val = grad_f_val.detach().numpy()

    return f_val, grad_f_val

def g(x):

    x = torch.tensor(x, dtype=torch.float64)

    g_val = 0.1 - (x[0:-1] - x[1:None]) ** 2

    return g_val

def an_jac_g(x):

    dg_dxi = np.zeros((len(x) - 1, len(x)))
    dg_dxip1 = np.zeros((len(x) - 1, len(x)))

    xi_indices = np.arange(0, len(x)-1)
    xip1_indices = np.arange(1, len(x))

    dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
    dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
    dg_dxi[xi_indices, xi_indices] = -2*x[xi_indices] + 2*x[xip1_indices]
    dg_dxip1[xi_indices, xip1_indices] = 2*x[xi_indices] - 2*x[xip1_indices]

    jac = dg_dxi + dg_dxip1

    return jac

def jac_g(x):

    x = torch.tensor(x, dtype=torch.float64, requires_grad=True)

    def gi(xi): return 0.00001 - (xi[0:-1] - xi[1:None]) ** 2

    grad_g_val = jacobian(gi, x)

    grad_g_val = grad_g_val.detach().numpy()

    return grad_g_val


def run_minimize(x):

    num_constraints = len(x) - 1
    lb = -torch.inf * torch.ones(num_constraints)
    ub = torch.zeros(num_constraints)
    nlcs = NonlinearConstraint(g, lb, ub)

    sol = minimize(f, x, method='SLSQP', constraints=nlcs)

    return sol

def run_minimize_grad(x):
    num_constraints = len(x) - 1
    lb = -torch.inf * torch.ones(num_constraints)
    ub = torch.zeros(num_constraints)
    nlcs = NonlinearConstraint(g, lb, ub, jac=jac_g)

    sol = minimize(f_jac, x, method='SLSQP', jac=True, constraints=nlcs)

    return sol



# def run_minimize_eq(x, num_eq_constraints):
#
#     num_constraints = len(x) - 1
#     lb = -np.inf * np.ones(num_constraints)
#     ub = np.zeros(num_constraints)
#     nlcs = NonlinearConstraint(constraint, lb, ub)
#
#     c = np.arange(num_eq_constraints)
#     eq_nlcs = NonlinearConstraint(lambda xs: [xs[i] for i in c], c, c)
#
#     sol = minimize(objective, x, method='SLSQP', constraints=[nlcs, eq_nlcs])
#
#     return sol
#
def run_minimize_bound(x, num_bounds):

    num_constraints = len(x) - 1
    glb = -torch.inf * torch.ones(num_constraints)
    gub = torch.zeros(num_constraints)
    nlcs = NonlinearConstraint(g, glb, gub)

    c = torch.arange(num_bounds)
    other = len(x) - num_bounds
    c_lb = c
    c_ub = c
    other_lb = -torch.inf * torch.ones(other)
    other_ub = torch.inf * torch.ones(other)

    lb = torch.hstack((c_lb, other_lb))
    ub = torch.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(f, x, method='SLSQP', constraints=nlcs, bounds=bounds)

    return sol

def run_minimize_bound_grad(x, num_bounds):

    num_constraints = len(x) - 1
    glb = -torch.inf * torch.ones(num_constraints)
    gub = torch.zeros(num_constraints)
    nlcs = NonlinearConstraint(g, glb, gub, jac=jac_g)

    c = torch.arange(num_bounds)
    other = len(x) - num_bounds
    c_lb = c
    c_ub = c
    other_lb = -torch.inf * torch.ones(other)
    other_ub = torch.inf * torch.ones(other)

    lb = torch.hstack((c_lb, other_lb))
    ub = torch.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(f_jac, x, method='SLSQP', jac=True, constraints=nlcs, bounds=bounds)

    return sol

def run_minimize_bound_no_con(x, num_bounds):

    fi = torch.sum((x[1:None] - x[0:-1])**2)

    c = np.arange(num_bounds)
    other = len(x) - num_bounds
    c_lb = c
    c_ub = c
    other_lb = -np.inf * np.ones(other)
    other_ub = np.inf * np.ones(other)

    lb = np.hstack((c_lb, other_lb))
    ub = np.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(f, x, method='SLSQP', bounds=bounds)

    return sol

# xe1 = torch.linspace(0, 999, 10)
# xe2 = torch.linspace(0, 999, 100)
# x2e2 = torch.linspace(0, 999, 200)
# x3e2 = torch.linspace(0, 999, 300)
# x4e2 = torch.linspace(0, 999, 400)
# x5e2 = torch.linspace(0, 999, 500)

xe1 = np.linspace(0, 999, 10)
xe2 = np.linspace(0, 999, 100)
x2e2 = np.linspace(0, 999, 200)

x3e2 = np.linspace(0, 999, 300)
x4e2 = np.linspace(0, 999, 400)
x5e2 = np.linspace(0, 999, 500)
xe3 = np.linspace(0, 999, 1000)

# a = np.zeros((9,10))
# xi_indices = np.arange(0, len(xe1)-1)
# xip1_indices = np.arange(1, len(xe1))
#
# a[xi_indices, xi_indices] = xe1[xi_indices]
# a[xi_indices, xip1_indices] = xe1[xip1_indices]


# run_minimize(xe2)
# run_minimize_bound(x2e2,100)



x = xe1

dg_dxi = np.zeros((len(x) - 1, len(x)))
dg_dxip1 = np.zeros((len(x) - 1, len(x)))

xi_indices = np.arange(0, len(x)-1)
xip1_indices = np.arange(1, len(x))

dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
dg_dxi[xi_indices, xi_indices] = -2*x[xi_indices] + 2*x[xip1_indices]
dg_dxip1[xi_indices, xip1_indices] = 2*x[xi_indices] - 2*x[xip1_indices]

jac = dg_dxi + dg_dxip1