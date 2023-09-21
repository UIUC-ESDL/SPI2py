import numpy as np
import torch
from torch.autograd.functional import jacobian

def euclidean_distance(point_a, point_b):
    """
    Calculates the Euclidean distance between two points.

    :param point_a: 3D point, (3,) ndarray
    :param point_b: 3D point, (3,) ndarray
    :return: Euclidean distance, float
    """

    delta_position = point_a - point_b

    distance = torch.linalg.norm(delta_position)

    return distance

a = torch.tensor([0, 0, 0], dtype=torch.float64, requires_grad=True)
b = torch.tensor([1, 1, 1], dtype=torch.float64, requires_grad=True)

dist = euclidean_distance(a, b)
print('dist: ', dist)

jac = jacobian(euclidean_distance, (a, b))

print('jac_a: ', jac[0])
print('jac_b: ', jac[1])

def jac_euclidean_distance(point_a, point_b):
    """
    Calculates the Jacobian of the Euclidean distance between two points.

    :param point_a: 3D point, (3,) ndarray
    :param point_b: 3D point, (3,) ndarray
    :return: Gradient of Euclidean distance, (3,) ndarray
    """

    grad_a = (point_a - point_b) / torch.linalg.norm(point_a - point_b)
    grad_b = (point_b - point_a) / torch.linalg.norm(point_a - point_b)

    jac = torch.vstack((grad_a, grad_b))

    return jac

jac_an = jac_euclidean_distance(a, b)
print('grad: ', jac_an)

