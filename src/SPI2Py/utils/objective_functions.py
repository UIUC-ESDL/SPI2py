"""



"""
import numpy as np
from itertools import combinations


def objective(x, layout):
    """
    Temp-For now calculate the pairwise distances between all design vectors

    Given a flat list and reshape
    jit
    vectorize
    :param x0:
    :param layout:
    :return:
    """

    # Fix this comment for actual reshape...
    # Reshape flattened design vector from 1D to 2D
    # [x1,y1,z1,x2,... ] to [[x1,y1,z1],[x2... ]
    # positions = positions.reshape(-1, 3)

    #
    pairwise_distance_pairs = list(combinations(layout.positions, 2))

    objective = 0
    for point1, point2 in pairwise_distance_pairs:
        pairwise_distance = np.linalg.norm(point2 - point1)

        objective += pairwise_distance

    # Do I need to return a scalar...?
    return objective
