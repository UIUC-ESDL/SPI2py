import numpy as np

# # def calculate_position_spline_segment(positions, control_point_1, control_point_2):
# #
# #     num_spheres = positions.shape[0]
# #
# #     # Fx ctrl points
# #     delta_cp_l = control_point_1 - cp_l
# #     delta_cp_r = control_point_2 - cp_r
# #
# #     # Left update
# #     dlx = np.linspace(delta_cp_l[0], 0, num_spheres)
# #     dly = np.linspace(delta_cp_l[1], 0, num_spheres)
# #     dlz = np.linspace(delta_cp_l[2], 0, num_spheres)
# #     dl = np.vstack((dlx, dly, dlz)).T
# #
# #     # Right update
# #     drx = np.linspace(0, delta_cp_r[0], num_spheres)
# #     dry = np.linspace(0, delta_cp_r[1], num_spheres)
# #     drz = np.linspace(0, delta_cp_r[2], num_spheres)
# #     dr = np.vstack((drx, dry, drz)).T
# #
# #     # Update segment
# #     positions_updated = positions + dl + dr
# #
# #     return positions_updated
#
#
#
#
#
#
#
# points_per_segment = 4
#
# # Control points are not spheres? Make sure spheres don't overlap?...
# cp_l = np.array([0, 0, 0])
# cp_r = np.array([3, 0, 0])
# segment_1 = np.linspace(cp_l, cp_r, points_per_segment)
#
# cp_l_new = np.array([-1, 0, 0])
# cp_r_new = np.array([6, 2, 1])
#
# segment_1_new_calculated = calculate_position_spline_segment(segment_1, cp_l_new, cp_r_new)
#
#
#
# # Ground truth
# segment_1_new = np.linspace(cp_l_new, cp_r_new, points_per_segment)
#
#
#
#
# print(np.isclose(segment_1_new, segment_1_new_calculated))

p = np.array([[0, 0, 0, 1]])
p10 = np.repeat(p, 10, axis=0)
p100 = np.repeat(p, 100, axis=0)

p10r = p10.reshape(10,4,1)
p100r = p100.reshape(100,4,1)

t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
t10 = np.tile(t, 10).T.reshape(10, 4, 4)
t100 = np.tile(t, 100).T.reshape(100, 4, 4)

# indices10 = []
# t10 @ p10.reshape(10,4,1)