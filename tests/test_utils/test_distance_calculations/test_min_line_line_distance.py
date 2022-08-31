import jax.numpy as jnp
from utils.distance_calculations import min_line_line_distance

# Add test to this and other dist for list vs np array


def test_parallel():

    a0 = jnp.array([0., 0., 0.])
    a1 = jnp.array([1., 0., 0.])
    b0 = jnp.array([0., 0., 1.])
    b1 = jnp.array([1., 0., 1.])

    dist = min_line_line_distance(a0, a1, b0, b1)

    print('dist', dist)

    assert round(dist,2) == 1


def test_skew():

    a0 = jnp.array([0., 0., 0.])
    a1 = jnp.array([1., 0., 0.])
    b0 = jnp.array([0., 0., 2.])
    b1 = jnp.array([1., 0., 1.])

    dist = min_line_line_distance(a0, a1, b0, b1)

    print('dist', dist)

    assert round(dist,2) == 1

