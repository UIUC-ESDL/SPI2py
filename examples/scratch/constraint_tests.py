import autograd.numpy as np
from autograd import grad
# import numpy as np
from scipy.sparse import csr_matrix, csc_matrix, coo_matrix, lil_matrix, dok_matrix, dia_matrix, bsr_matrix
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint, approx_fprime, Bounds


def f(x):
    """Minimize the distance between each point"""

    f_val = np.sum((x[1:None] - x[0:-1])**2)

    return f_val


def jac_f(x):

    jac = np.zeros((1, len(x)))

    jac[0][0] = -2*x[1] + 2*x[0]
    jac[0][-1] = 2*x[-1] - 2*x[-2]

    jac[0][1:-1] = 2*x[1:-1] - 2*x[0:-2] - 2*x[2:None] + 2*x[1:-1]

    return jac

def f_jac(x):

    f_val = f(x)

    jac_f_val = jac_f(x)

    return f_val, jac_f_val

def g(x):

    g_val = 0.1 - (x[0:-1] - x[1:None]) ** 2

    return g_val

def jac_g(x):

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

def g_agg(x):

    g_val = 0.1 - (x[0:-1] - x[1:None]) ** 2

    rho = 100
    g_bar_ks = np.max(g_val) + 1 / rho * np.log(np.sum(np.exp(rho * (g_val - np.max(g_val)))))

    return g_bar_ks

jac_g_agg = grad(g_agg)

# def jac_g_agg(x):
#
#     dg_dxi = np.zeros((len(x) - 1, len(x)))
#     dg_dxip1 = np.zeros((len(x) - 1, len(x)))
#
#     xi_indices = np.arange(0, len(x)-1)
#     xip1_indices = np.arange(1, len(x))
#
#     dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
#     dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
#     dg_dxi[xi_indices, xi_indices] = -2*x[xi_indices] + 2*x[xip1_indices]
#     dg_dxip1[xi_indices, xip1_indices] = 2*x[xi_indices] - 2*x[xip1_indices]
#
#     jac = dg_dxi + dg_dxip1
#
#     return jac

# def sjac_g(x):
#
#     dg_dxi = np.zeros((len(x) - 1, len(x)))
#     dg_dxip1 = np.zeros((len(x) - 1, len(x)))
#
#     xi_indices = np.arange(0, len(x)-1)
#     xip1_indices = np.arange(1, len(x))
#
#     dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
#     dg_dxip1[xi_indices, xip1_indices] = x[xip1_indices]
#     dg_dxi[xi_indices, xi_indices] = -2*x[xi_indices] + 2*x[xip1_indices]
#     dg_dxip1[xi_indices, xip1_indices] = 2*x[xi_indices] - 2*x[xip1_indices]
#
#     jac = dg_dxi + dg_dxip1
#
#     sjac = dia_matrix(jac)
#
#     return sjac

def run_minimize(x):

    num_constraints = len(x) - 1
    lb = -np.inf * np.ones(num_constraints)
    ub = np.zeros(num_constraints)
    nlcs = NonlinearConstraint(g, lb, ub, jac=jac_g)

    sol = minimize(f_jac, x, method='SLSQP', jac=True, constraints=nlcs)

    return sol

# def run_minimize_sparse(x):
#
#     num_constraints = len(x) - 1
#     lb = -np.inf * np.ones(num_constraints)
#     ub = np.zeros(num_constraints)
#     nlcs = NonlinearConstraint(g, lb, ub, jac=sjac_g)
#
#     sol = minimize(f_jac, x, method='SLSQP', jac=True, constraints=nlcs)
#
#     return sol

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


def run_minimize_bound(x, num_bounds):

    c = x[0:num_bounds]
    other = len(x) - len(c)
    c_lb = c
    c_ub = c
    other_lb = -np.inf * np.ones(other)
    other_ub = np.inf * np.ones(other)

    lb = np.hstack((c_lb, other_lb))
    ub = np.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(f_jac, x, method='SLSQP', jac=True, bounds=bounds)

    return sol

def run_minimize_bound_con(x, num_bounds):

    # TODO is this larger than bounded-removed variables?
    num_constraints = len(x) - 1
    glb = -np.inf * np.ones(num_constraints)
    gub = np.zeros(num_constraints)
    nlcs = NonlinearConstraint(g, glb, gub, jac=jac_g)

    c = np.arange(num_bounds)
    other = len(x) - num_bounds
    c_lb = c
    c_ub = c
    other_lb = -np.inf * np.ones(other)
    other_ub = np.inf * np.ones(other)

    lb = np.hstack((c_lb, other_lb))
    ub = np.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(f_jac, x, method='SLSQP', jac=True, constraints=nlcs, bounds=bounds)

    return sol


def run_minimize_bound_aggcon(x, num_bounds):

    nlcs = NonlinearConstraint(g_agg, -np.inf, 0, jac=jac_g_agg)

    c = np.arange(num_bounds)
    other = len(x) - num_bounds
    c_lb = c
    c_ub = c
    other_lb = -np.inf * np.ones(other)
    other_ub = np.inf * np.ones(other)

    lb = np.hstack((c_lb, other_lb))
    ub = np.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(f_jac, x, method='SLSQP', jac=True, constraints=nlcs, bounds=bounds)

    return sol


def run_minimize_bound_con_no_grad(x, num_bounds):

    num_constraints = len(x) - 1
    glb = -np.inf * np.ones(num_constraints)
    gub = np.zeros(num_constraints)
    nlcs = NonlinearConstraint(g, glb, gub)

    c = np.arange(num_bounds)
    other = len(x) - num_bounds
    c_lb = c
    c_ub = c
    other_lb = -np.inf * np.ones(other)
    other_ub = np.inf * np.ones(other)

    lb = np.hstack((c_lb, other_lb))
    ub = np.hstack((c_ub, other_ub))

    bounds = Bounds(lb, ub)

    sol = minimize(f, x, method='SLSQP', constraints=nlcs, bounds=bounds)

    return sol



xe1 = np.linspace(0, 999, 10)
xe2 = np.linspace(0, 999, 100)
x2e2 = np.linspace(0, 999, 200)
x3e2 = np.linspace(0, 999, 300)
x4e2 = np.linspace(0, 999, 400)
x5e2 = np.linspace(0, 999, 500)
xe3 = np.linspace(0, 999, 1000)



# run_minimize(x4e2)
# run_minimize_bound(x2e2,100)
# run_minimize_bound_con(x2e2, 100)

run_minimize_bound(xe2, 0)
# run_minimize_bound(xe2, 1)




