"""



"""
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist


def objective_1(x, layout):
    """
    Combinatorial check

    Temp-For now calculate the pairwise distances between all design vectors

    Given a flat list and reshape
    jit
    vectorize
    :param x:
    :param layout:
    :return:
    """

    positions_dict = layout.get_positions(x)

    # Fix this comment for actual reshape...
    # Reshape flattened design vector from 1D to 2D
    # [x1,y1,z1,x2,... ] to [[x1,y1,z1],[x2... ]
    # positions = positions.reshape(-1, 3)

    object_pairs = list(combinations(positions_dict.keys(), 2))

    objective = 0
    for object_pair in object_pairs:
        object_1 = object_pair[0]
        object_2 = object_pair[1]

        positions_1 = positions_dict[object_1]
        positions_2 = positions_dict[object_2]

        objective += sum(sum(cdist(positions_1, positions_2)))

    return objective
