"""



"""


from .distance import max_spheres_spheres_interference


def interference(x, layout, pairs):
    """
    Checks for the maximum collision between two ojects

    :param x:
    :param layout:
    :return:
    """

    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = layout.calculate_positions(x)

    # Calculate the interferences between each sphere of each object pair
    interferences = []
    # for (positions_1, radius_1, position_2, radius_2) in zip(positions_1, radii_1, positions_2, radii_2):
    #
    #     dist = min_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)
    #
    #     interferences.append(dist)
    #
    #     max_interference = max(interferences)

    for obj1, obj2 in pairs:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = positions_dict[str(obj1)][1]

        positions_b = positions_dict[str(obj2)][0]
        radii_b = positions_dict[str(obj2)][1]

        dist = max_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)

        interferences.append(dist)

        max_interference = max(interferences)

    return max_interference
