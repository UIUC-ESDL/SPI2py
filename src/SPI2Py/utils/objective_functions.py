"""



"""

from itertools import combinations
from scipy.spatial.distance import cdist


def aggregate_pairwise_distance(x, layout):
    """
    Aggregates the distiance between each 2-pair of objects

    :param x:
    :param layout:
    :return:
    """

    # Calculate the position of every sphere based on design vector x
    positions_dict = layout.get_positions(x)

    # Create a list of object pairs
    object_pairs = list(combinations(positions_dict.keys(), 2))

    objective = 0
    for object_pair in object_pairs:
        object_1 = object_pair[0]
        object_2 = object_pair[1]

        positions_1 = positions_dict[object_1]
        positions_2 = positions_dict[object_2]

        objective += sum(sum(cdist(positions_1, positions_2)))

    return objective
