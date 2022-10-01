"""

"""

import jax.numpy as jnp
from jax import jit
from jax.numpy import sin, cos


@jit
def translate(current_positions, current_reference_point, new_reference_point):
    """
    Translates a set of points based on the change in position of a reference point

    :param current_positions:
    :param current_reference_point:
    :param new_reference_point:
    :return:
    """

    delta_x, delta_y, delta_z = new_reference_point - current_reference_point

    new_positions = current_positions + jnp.array([delta_x, delta_y, delta_z])

    return new_positions


@jit
def rotate(reference_position, positions, current_angle, new_angle):
    """
    Rotates a set of points based on the change of rotation of a reference point

    Need to verify the function rotates correctly (e.g., when not about the axis)
    Angles in radians...

    :param reference_position:
    :param positions:
    :param current_angle: Angle in radians!!!
    :param new_angle:
    :return:
    """

    # Shift the object to origin
    origin_positions = positions - reference_position

    delta_theta_x, delta_theta_y, delta_theta_z = new_angle - current_angle

    r_x = jnp.array([[1, 0,                  0                  ],
                     [0, cos(delta_theta_x), -sin(delta_theta_x)],
                     [0, sin(delta_theta_x), cos(delta_theta_x)]])

    r_y = jnp.array([[cos(delta_theta_y),  0, sin(delta_theta_y)],
                     [0,                   1, 0                 ],
                     [-sin(delta_theta_y), 0, cos(delta_theta_y)]])

    r_z = jnp.array([[cos(delta_theta_z), -sin(delta_theta_z), 0],
                     [sin(delta_theta_z), cos(delta_theta_z),  0],
                     [0,                  0,                   1]])

    # Apple rotation matrix with ZYX Euler angle convention
    r = r_z @ r_y @ r_x
    # Transpose positions from [[x1,y1,z1],[x2... ] to [[x1,x2,x3],[y1,... ]
    rotated_origin_positions = (r @ origin_positions.T).T

    # Shift back from origin
    new_positions = rotated_origin_positions + reference_position

    return new_positions



reference_position = jnp.array([2.,2.,0.])
positions = jnp.array([[2.,2.,0.],[4.,2.,0.],[4,0,0]])
current_angle = jnp.array([0.,0.,0.])

new_angle_x_90 = jnp.array([jnp.pi/2,0.,0.])
new_angle_y_90 = jnp.array([0.,jnp.pi/2,0.])
new_angle_z_90 = jnp.array([0.,0.,jnp.pi/2])

print('Old', positions)

new_positions = rotate(reference_position,positions,current_angle,new_angle_x_90)
print('new pos x',new_positions)

new_positions = rotate(reference_position,positions,current_angle,new_angle_y_90)
print('new pos y',new_positions)

new_positions = rotate(reference_position,positions,current_angle,new_angle_z_90)
print('new pos z',new_positions)