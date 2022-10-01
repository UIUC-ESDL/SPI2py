"""

Here is a helpful checker for test cases: https://www.emathhelp.net/calculators/algebra-2/rotation-calculator/

Use round to avoid small numerical differences
"""

import jax.numpy as jnp
from utils.transformations import rotate


positions = jnp.array([[2.,2.,0.],[4.,2.,0.],[4,0,0]])
current_angle = jnp.array([0.,0.,0.])

def test_rotate_x_cc():

    new_angle_x_90 = jnp.array([jnp.pi / 2, 0., 0.])
    delta_rotation = new_angle_x_90 - current_angle
    new_positions = round(rotate(positions,delta_rotation),0)

    expected = jnp.array([[2.,2.,0.],[4.,2.,0.],[4.,2.,-2.]])

    assert jnp.array_equal(new_positions, expected)


def test_rotate_y_cc():

    new_angle_y_90 = jnp.array([0., jnp.pi / 2, 0.])
    delta_rotation = new_angle_y_90 - current_angle
    new_positions = round(rotate(positions,delta_rotation),0)

    expected = jnp.array([[2.,2.,0.],[2.,2.,-2.],[2.,0.,-2.]])

    assert jnp.array_equal(new_positions, expected)


def test_rotate_z_cc():

    new_angle_z_90 = jnp.array([0., 0., jnp.pi / 2])
    delta_rotation = new_angle_z_90 - current_angle
    new_positions = round(rotate(positions, delta_rotation), 0)

    expected = jnp.array([[2., 2., 0.], [2., 4., 0.], [4., 4., 0.]])

    assert jnp.array_equal(new_positions, expected)