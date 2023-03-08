"""



"""
import numpy as np
from .distance import max_spheres_spheres_interference


def max_interference(x, layout, pairs):
    """
    Calculates the maximum interference between two objects.

    A positive interference indicates that the objects are in collision.
    A negative interference indicates that the objects are not in collision.

    TODO Implement constraint aggregation methods in lieu of min/max as described in MDO book Chapter 5.7

    :param x:
    :param layout:
    :param pairs:
    :return:
    """

    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = layout.calculate_positions(x)

    # Calculate the interferences between each sphere of each object pair
    interferences = []

    for obj1, obj2 in pairs:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = positions_dict[str(obj1)][1]

        positions_b = positions_dict[str(obj2)][0]
        radii_b = positions_dict[str(obj2)][1]

        dist = max_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)

        interferences.append(dist)

    max_interference = max(interferences)

    return max_interference



