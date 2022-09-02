import jax.numpy as jnp
from utils.solver import solver, objective_function, objective_function_gradient

def test_objective_function():
    x0 = jnp.array([[1.,2.,3.],[4.,3.,1.],[0.,5.,2.]])
    x0 = x0.reshape(-1)

    obj = objective_function(x0)

def test_objective_function_gradient():
    x0 = jnp.array([[1., 2., 3.], [4., 3., 1.], [0., 5., 2.]])
    x0 = x0.reshape(-1)

    g = objective_function_gradient(x0)

def test_solver():
    x0 = jnp.array([[1.,2.,3.],[4.,3.,1.],[0.,5.,2.]])
    x0 = x0.reshape(-1)
    res = solver(x0)
    print('res', res)