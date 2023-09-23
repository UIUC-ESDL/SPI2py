import numpy as np

points_per_segment = 4

# Control points are not spheres? Make sure spheres don't overlap?...
cp_l = np.array([0, 0, 0])
cp_r = np.array([3, 0, 0])
segment_1 = np.linspace(cp_l, cp_r, points_per_segment)

cp_l_new = np.array([-1, 0, 0])
cp_r_new = np.array([6, 2, 1])

delta_cp_l = cp_l_new - cp_l
delta_cp_r = cp_r_new - cp_r


# Left update
dlx = np.linspace(delta_cp_l[0], 0, points_per_segment)
dly = np.linspace(delta_cp_l[1], 0, points_per_segment)
dlz = np.linspace(delta_cp_l[2], 0, points_per_segment)
dl = np.vstack((dlx, dly, dlz)).T

# Right update
drx = np.linspace(0, delta_cp_r[0], points_per_segment)
dry = np.linspace(0, delta_cp_r[1], points_per_segment)
drz = np.linspace(0, delta_cp_r[2], points_per_segment)
dr = np.vstack((drx, dry, drz)).T

# Update segment
segment_1_new_calculated = segment_1 + dl + dr



# Ground truth
segment_1_new = np.linspace(cp_l_new, cp_r_new, points_per_segment)




print(np.isclose(segment_1_new, segment_1_new_calculated))
