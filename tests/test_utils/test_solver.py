import jax.numpy as jnp
from jax import jit, grad, jacfwd
from utils.solver import solver, objective_function, objective_function_gradient
from scipy.optimize import minimize, Bounds, NonlinearConstraint

def test_objective_function():
    x0 = jnp.array([[1.,2.,3.],[4.,3.,1.],[0.,5.,2.]])
    x0 = x0.reshape(-1)

    obj = objective_function(x0)

def test_objective_function_gradient():
    x0 = jnp.array([[1., 2., 3.], [4., 3., 1.], [0., 5., 2.]])
    x0 = x0.reshape(-1)

    g = objective_function_gradient(x0)




def test_unconstrained_optimization():
    """
    Ensures JAX's just-in-time compilation and algorithmic differentiation features
    are working with the scipy.optimize.minimize method.

    As mentioned before the JAX implementation of minimize does not currently support
    necessary constrained optimization methods.
    """

    @jit
    def f(x): return (1 - x[0]) ** 2 + (1 - x[1]) ** 2

    @jit
    def g(x): return grad(f)(x)

    @jit
    def h(x): return jacfwd(g)(x)

    x0 = jnp.array([1., 2.])
    res = minimize(f, x0, jac=g, hess=h)

    assert res.success is True
    assert all(res.x == jnp.array([1.,1.]))


def test_constrained_optimization():
    @jit
    def f(x): return (1 - x[0]) ** 2 + (1 - x[1]) ** 2

    @jit
    def g(x): return grad(f)(x)

    @jit
    def h(x): return jacfwd(g)(x)

    def c(x): return x[0]+x[1]

    def c_jac(x): return grad(c)(x)

    # Why do I get an error specifying the Hessian in the NonlinearConstrain object?
    def c_hess(x): return jacfwd(c_jac)(x)

    nlc = NonlinearConstraint(c, 0, 5, jac=c_jac)

    x0 = jnp.array([1., 2.])
    res = minimize(f, x0, method = 'trust-constr', constraints = nlc,jac=g, hess=h)

    assert res.success is True
    assert all(res.x == jnp.array([1., 1.]))





def test_solver():
    x0 = jnp.array([[1.,2.,3.],[4.,3.,1.],[0.,5.,2.]])
    x0 = x0.reshape(-1)

    def f(x): return x**2
    def g(x): return grad(f)(x)

    def h(x): return jacfwd(g)(x)


    res = solver(f,x0,1)
    print('res', res)