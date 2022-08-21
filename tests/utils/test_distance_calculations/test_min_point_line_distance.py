import numpy as np
from utils.distance_calculations import min_point_line_distance


def test_first():
    p = np.array([1, 0, 1])
    a = np.array([0, 0, 0])
    b = np.array([1, 0, 0])

    min_dist = min_point_line_distance(p, a, b)
