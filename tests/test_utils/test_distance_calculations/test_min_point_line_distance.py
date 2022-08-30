import numpy as np
from utils.distance_calculations import min_point_line_distance


def test_first():
    p = np.array([1., 0., 1.])
    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])

    min_dist = min_point_line_distance(p, a, b)


# Speed test saved...
# p = np.array([0., 1., 1.])
# a = np.array([0., 0., 0.])
# b = np.array([0., 0., 1.])
#
# aa = np.array([[0., 0., 0.]])
# ab = np.array([[0., 0., 0.],
#                [0., 0., 1.],
#                [0., 0., 1.],
#                [0., 0., 1.],
#                [0., 0., 1.]])
#
# start = perf_counter_ns()
# print(min_point_line_distance(p, a, b))
# stop = perf_counter_ns()
# print('Time', stop-start)
#
# start = perf_counter_ns()
# print(min_point_line_distance(p, a, b))
# stop = perf_counter_ns()
# print('Time', stop-start)
#
# start = perf_counter_ns()
# print(min_cdist(aa, ab))
# stop = perf_counter_ns()
# print('Time', stop-start)
#
# # print(min_cdist(p, ab))