import jax.numpy as jnp
from jax import jvp, vjp

def fun_fwd(x):
    return jnp.array([x**2, x**2])

def fun_rev(x):
    return jnp.sum(x)

x_scl = 2.0
x_arr = jnp.array([2.0, 2.0])

vec = jnp.array([1.0, 0.0])


primals_fwd, tangents_fwd = jvp(fun_fwd, (x_arr,), (vec,))
primals_rev, tangents_rev = vjp(fun_rev, x_arr)
