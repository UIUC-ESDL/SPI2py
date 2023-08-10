import jax.numpy as np
from jax import grad, jacrev


a = np.array([1., 2., 3.])

def f(x):

    my_dict = {'a': x[0], 'b': x[1], 'c': x[2]}

    x_new = np.array([my_dict['a'], my_dict['b'], my_dict['c']])

    return np.sum(x_new**2)

print(grad(f)(a))

print(jacrev(f)(a))
