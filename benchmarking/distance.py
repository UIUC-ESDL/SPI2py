import numpy as np
from time import time_ns
from SPI2py.analysis.distance import minimum_distance_segment_segment, distances_points_points
from scipy.spatial.distance import cdist
from autograd import grad
from scipy.optimize import minimize

# Benchmark single time

a = np.random.rand(200,3)
b = np.random.rand(200,3)

# t1_1 = time_ns()
# dist = distances_points_points(a, b)
# t1_2 = time_ns()
# c_dist = cdist(a, b)
# t1_3 = time_ns()
#
# print(f"SPI2py: {t1_2 - t1_1} ns")
# print(f"SciPy: {t1_3 - t1_2} ns")



x0 = np.random.rand(200, 3)
x0 = x0.reshape(-1)

def f(x):
    x = x.reshape(-1, 3)
    a = x[0:100]
    b = x[100:200]

    return np.min(distances_points_points(a, b))


df = grad(f)


def f1(x):
    x = x.reshape(-1, 3)
    a = x[0:100]
    b = x[100:200]
    return np.min(cdist(a, b))

# t1_1 = time_ns()
# ans1 = minimize(f, x0)
# t1_2 = time_ns()
# ans2 = minimize(f1, x0)
# t1_3 = time_ns()
# ans3 = minimize(f, x0, jac=df)
# t1_4 = time_ns()

# print(f"SPI2py: {t1_2 - t1_1} ns")
# print(f"SciPy: {t1_3 - t1_2} ns")
# print(f"SPI2py with grad: {t1_4 - t1_3} ns")



