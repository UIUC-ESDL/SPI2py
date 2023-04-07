"""Objective functions for the layout optimization problem

This module contains the objective functions for the layout optimization problem.
"""
import numpy as np
from itertools import combinations
from scipy.spatial.distance import cdist


def aggregate_pairwise_distance(x, layout):
    """
    Aggregates the distance between each 2-pair of classes

    This function does not work well because its value becomes very large very quickly.
    The gradient-based solver tends to "ignore" constraint functions that produce consraints
    several orders of magnitude smaller, even when you force feasibility.

    :param x:
    :param layout:
    :return:
    """

    # Calculate the position of every sphere based on design vector x
    positions_dict = layout.calculate_positions(x)

    # Create a list of object pairs
    object_pairs = list(combinations(positions_dict.keys(), 2))

    objective = 0
    for object_pair in object_pairs:
        object_1 = object_pair[0]
        object_2 = object_pair[1]

        positions_1 = positions_dict[object_1][0]
        positions_2 = positions_dict[object_2][0]

        objective += sum(sum(cdist(positions_1, positions_2)))

    # Adding a scaling factor to the objective function helps the solver
    # objective = objective / 100000

    return objective


def normalized_aggregate_gap_distance(x, layout):
    """
    Returns the normalized gap

    :param x:
    :param layout:
    :return:
    """

    # Calculate the position of every sphere based on design vector x
    positions_dict = layout.calculate_positions(x)

    # Create a list of object pairs
    object_pairs = list(combinations(positions_dict.keys(), 2))

    objective = []
    for object_pair in object_pairs:
        object_1 = object_pair[0]
        object_2 = object_pair[1]

        positions_1 = positions_dict[object_1][0]
        positions_2 = positions_dict[object_2][0]

        objective.append(sum(sum(cdist(positions_1, positions_2))))

    # Divide by number of components
    objective = np.sum(objective) / len(objective)


    # TODO Add a scaling factor to the objective function to see if it helps
    objective = objective / 150

    return objective