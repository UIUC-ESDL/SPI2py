"""



"""
import numpy as np

from src.SPI2Py.utils.distance_calculations import min_points_points_distance, min_spheres_spheres_interference


def constraint_component_component(x, layout):
    """
    ...
    Applies hierarchical collision detection to both components

    TODO Write this function

    TODO Fix this to work with variable radii

    :param layout:
    :param x:
    :return:
    """

    positions_dict = layout.get_positions(x)

    # TODO relabel everything as interferences and maxes...
    distances = []
    for obj1, obj2 in layout.component_component_pairs:

        positions_a = positions_dict[obj1]
        radii_a = obj1.radii.reshape(-1, 1)

        positions_b = positions_dict[obj2]
        radii_b = obj2.radii.reshape(-1, 1)

        dist = min_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)

        distances.append(dist)

        print('interference',max(distances))

    return max(distances)



def constraint_component_interconnect(positions, radii):
    """

    TODO Write this function

    Applies hierarchical collision detection to component
    :param positions:
    :param radii:
    :return:
    """
    pass


def constraint_interconnect_interconnect(positions, radii):
    # TODO Write this function
    pass


def constraint_structure_all(positions, radii):
    # TODO Write this function
    pass

