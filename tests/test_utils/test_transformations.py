import jax.numpy as jnp
from utils.transformations import rotate


a = jnp.array([[1,2,3], [1,2,3], [1,2,3]])
ang = jnp.array([1,1.1,2])

def test_rotate():
    rotate(a,ang)