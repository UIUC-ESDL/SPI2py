"""Module...
...
TODO Add functionality to rotate geometric primitives
"""

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt


def generate_rectangular_prism(origin, dimension):
    """
    Generates...

    This is a rough representation, it doesn't actually package spheres precisely...
    :param origin:
    :param dimension:
    :return:
    """

    # Set the diameter of the packing sphere to the smallest dimension
    diameter = np.min(dimension)
    radius = diameter / 2

    origin_x, origin_y, origin_z = origin
    len_x, len_y, len_z = dimension

    num_x_spheres = int(len_x // diameter)
    num_y_spheres = int(len_y // diameter)
    num_z_spheres = int(len_z // diameter)

    # Add 2 since we want to fit the spheres inside the box, not along its corners
    num_x_nodes = num_x_spheres + 2
    num_y_nodes = num_y_spheres + 2
    num_z_nodes = num_z_spheres + 2

    # [1:-1] Removes the first and last nodes because...
    x = np.linspace(origin_x, origin_x + len_x, num_x_nodes)[1:-1]
    y = np.linspace(origin_y, origin_y + len_y, num_y_nodes)[1:-1]
    z = np.linspace(origin_z, origin_z + len_z, num_z_nodes)[1:-1]

    xx, yy, zz = np.meshgrid(x, y, z)

    positions = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)))

    sphere_count = positions.shape[0]
    radii = np.repeat(radius, sphere_count)

    return positions, radii


def generate_rectangular_prism2(origin, dimension):
    """
    Generates...

    This is a rough representation, it doesn't actually package spheres precisely...
    :param origin:
    :param dimension:
    :return:
    """

    # Recursion depth
    rdepth = 2

    # Set the diameter of the packing sphere to the smallest dimension
    diameter = np.min(dimension)
    radius = diameter / 2

    origin_x, origin_y, origin_z = origin
    len_x, len_y, len_z = dimension

    num_x_spheres = int(len_x // diameter)
    num_y_spheres = int(len_y // diameter)
    num_z_spheres = int(len_z // diameter)

    # Add 2 since we want to fit the spheres inside the box, not along its corners
    num_x_nodes = num_x_spheres + 2
    num_y_nodes = num_y_spheres + 2
    num_z_nodes = num_z_spheres + 2

    # [1:-1] Removes the first and last nodes because...
    x = np.linspace(origin_x, origin_x + len_x, num_x_nodes)[1:-1]
    y = np.linspace(origin_y, origin_y + len_y, num_y_nodes)[1:-1]
    z = np.linspace(origin_z, origin_z + len_z, num_z_nodes)[1:-1]

    xx, yy, zz = np.meshgrid(x, y, z)

    positions = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)))

    sphere_count = positions.shape[0]
    radii = np.repeat(radius, sphere_count)

    return positions, radii


def generate_rectangular_prisms(origins, dimensions):
    positions, radii = np.empty((0, 3)), np.empty(0)

    for origin, dimension in zip(origins, dimensions):
        sphere_positions, sphere_radius = generate_rectangular_prism(origin, dimension)
        positions = np.vstack((positions, sphere_positions))
        radii = np.append(radii, sphere_radius)

    return positions, radii


# TODO Implement a uniformly thick line between two points
# TODO Implement a variably thick line between two points
# TODO Implement MDBD for geometric primitives
# TODO Implement MDBD for convex hulls
# TODO Implement MDBD for STL files

# 3D Points of a rectangular prism
p1 = np.array([[0, 0, 0]])
p2 = np.array([[1, 0, 0]])
p3 = np.array([[1, 1, 0]])
p4 = np.array([[0, 1, 0]])
p5 = np.array([[0, 0, 1]])
p6 = np.array([[1, 0, 1]])
p7 = np.array([[1, 1, 1]])
p8 = np.array([[0, 1, 1]])

rect = np.vstack((p1, p2, p3, p4, p5, p6, p7, p8))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(rect[:, 0], rect[:, 1], rect[:, 2], marker='o', s=100, c='r')

#
#
# def run_optimization(xrange, yrange, zrange, recursion_depth=6, max_iter=20):
#
#     # Generate the initial shape
#     x1, x2 = xrange
#     y1, y2 = yrange
#     z1, z2 = zrange
#
#     shape = np.array([[x1, y1, z1],
#                       [x1, y2, z1],
#                       [x1, y1, z2],
#                       [x1, y2, z2],
#                       [x2, y1, z2],
#                       [x2, y1, z1],
#                       [x2, y2, z1],
#                       [x2, y2, z2]])
#
#     points = np.empty((0, 3))
#     radii = np.empty(0)
#
#     def objective(di): return di[3]
#
#     def constraint_1(di):
#         """Min dis must be > 0 """
#
#         x, y, z, r = di
#
#         point = np.array([[x, y, z]])
#         radius = np.array([r])
#
#         if points.shape[0] == 0:
#             return min([x2-x1, y2-y1, z2-z1])/2
#
#         else:
#
#             gaps = []
#
#             for j in range(len(points)):
#                 gap = np.linalg.norm(point - points[j]) - (radius + radii[j])
#                 gaps.append(gap)
#
#             return min(gaps)
#
#     d0 = np.array([0.5, 0.5, 0.5, 0.25])
#
#     g1 = NonlinearConstraint(constraint_1, 0, np.inf)
#
#     # x > x1
#     g2 = NonlinearConstraint(lambda di: di[0], x1, np.inf)
#
#     # x < x2
#     g3 = NonlinearConstraint(lambda di: di[0], -np.inf, x2)
#
#     # y > y1
#     g4 = NonlinearConstraint(lambda di: di[1], y1, np.inf)
#
#     # y < y2
#     g5 = NonlinearConstraint(lambda di: di[1], -np.inf, y2)
#
#     # z > z1
#     g6 = NonlinearConstraint(lambda di: di[2], z1, np.inf)
#
#     # z < z2
#     g7 = NonlinearConstraint(lambda di: di[2], -np.inf, z2)
#
#     max_r = min([x2-x1, y2-y1, z2-z1])/2
#
#     # TODO Implement bounds
#     bounds = Bounds([x1, y1, z1, 0], [x2, y2, z2, max_r])
#
#     for i in range(recursion_depth):
#         # d0 is random
#         res = minimize(objective, d0, method='trust-constr', constraints=[g1, g2, g3, g4, g5, g6, g7], bounds=bounds)
#         # print(res)
#         points = np.vstack((points, res.x[:3]))
#         radii = np.append(radii, res.x[3])
#         print('point', res.x[:3])
#         print('radius', res.x[3])
#
#     return points, radii
#
#
# points_opt, radii_opt = run_optimization([0, 1], [0, 1], [0, 1])
#
# ax.scatter(points_opt[:, 0], points_opt[:, 1], points_opt[:, 2], marker='o', s=100, c='b')
#
# plt.show()





xrange = [0, 1]
yrange = [0, 1]
zrange = [0, 1]



# Generate the initial shape
x1, x2 = xrange
y1, y2 = yrange
z1, z2 = zrange

max_r = min([x2-x1, y2-y1, z2-z1])/2

shape = np.array([[x1, y1, z1],
                  [x1, y2, z1],
                  [x1, y1, z2],
                  [x1, y2, z2],
                  [x2, y1, z2],
                  [x2, y1, z1],
                  [x2, y2, z1],
                  [x2, y2, z2]])

points = np.empty((0, 3))
radii = np.empty(0)

def objective(di): return 1/di[3]

def constraint_1(di):
    """Min dis must be > 0 """

    x, y, z, r = di

    point = np.array([[x, y, z]])
    radius = np.array([r])

    if points.shape[0] == 0:
        return float(min([x2-x1, y2-y1, z2-z1])/2)

    else:

        gaps = []

        for j in range(len(points)):
            gap = np.linalg.norm(point - points[j]) - (radius + radii[j])
            gaps.append(gap)

        return float(min(gaps))

g1 = NonlinearConstraint(constraint_1, 0, np.inf)

# x > x1
g2 = NonlinearConstraint(lambda di: di[0], x1, np.inf)

# x < x2
g3 = NonlinearConstraint(lambda di: di[0], -np.inf, x2)

# y > y1
g4 = NonlinearConstraint(lambda di: di[1], y1, np.inf)

# y < y2
g5 = NonlinearConstraint(lambda di: di[1], -np.inf, y2)

# z > z1
g6 = NonlinearConstraint(lambda di: di[2], z1, np.inf)

# z < z2
g7 = NonlinearConstraint(lambda di: di[2], -np.inf, z2)



# TODO Implement bounds
bounds = Bounds([x1, y1, z1, 0], [x2, y2, z2, max_r])

for i in range(20):

    # TODO Make this work for other than 0 to 1
    d0 = np.random.rand(4)

    # d0 = np.array([0.5, 0.5, 0.5, 0.25])

    # d0 is random
    res = minimize(objective, d0, method='trust-constr', constraints=[g1, g2, g3, g4, g5, g6, g7], bounds=bounds)
    # print(res)
    points = np.vstack((points, res.x[:3]))
    radii = np.append(radii, res.x[3])
    print('point', res.x[:3])
    print('radius', res.x[3])

ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', s=300, c='b')
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)


plt.show()
