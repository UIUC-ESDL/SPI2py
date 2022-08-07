import numpy as np
from scipy.optimize import minimize, show_options


def callback(xk):
    print('xk', xk)


def fun(x): return x ** 2 + 2


x0 = np.array([1, 2, 3])
res = minimize(fun, x0, method='COBYLA', callback=callback)
