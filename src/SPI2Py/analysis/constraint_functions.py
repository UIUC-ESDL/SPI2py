"""



"""


from .distance import min_spheres_spheres_interference


# def interference(positions_1, radii_1, positions_2, radii_2):
#     """
#     Checks for the maximum collision between two ojects
#
#     :param x:
#     :param layout:
#     :return:
#     """
#
#     # Calculate the interferences between each sphere of each object pair
#     interferences = []
#     for (positions_1, radius_1, position_2, radius_2) in zip(positions_1, radii_1, positions_2, radii_2):
#
#         dist = min_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)
#
#         interferences.append(dist)
#
#         max_interference = max(interferences)
#
#     return max_interference


def interference_component_component(x, layout):
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
    for obj1, obj2 in layout.component_component_pairs:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = positions_dict[str(obj1)][1]

        positions_b = positions_dict[str(obj2)][0]
        radii_b = positions_dict[str(obj2)][1]

        dist = min_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)

        interferences.append(dist)

        max_interference = max(interferences)

    return max_interference


def interference_component_interconnect(x, layout):

    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = layout.calculate_positions(x)

    # Calculate the interferences between each sphere of each object pair
    interferences = []
    for obj1, obj2 in layout.component_interconnect_pairs:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = positions_dict[str(obj1)][1]

        positions_b = positions_dict[str(obj2)][0]
        radii_b = positions_dict[str(obj2)][1]

        dist = min_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)

        interferences.append(dist)

        max_interference = max(interferences)

    return max_interference


def interference_interconnect_interconnect(x, layout):

    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = layout.calculate_positions(x)

    # Calculate the interferences between each sphere of each object pair
    interferences = []
    for obj1, obj2 in layout.interconnect_interconnect_pairs:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = positions_dict[str(obj1)][1]

        positions_b = positions_dict[str(obj2)][0]
        radii_b = positions_dict[str(obj2)][1]

        dist = min_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)

        interferences.append(dist)

        max_interference = max(interferences)

    return max_interference


def interference_structure_all(x, layout):
    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = layout.calculate_positions(x)

    # Calculate the interferences between each sphere of each object pair
    interferences = []
    for obj1, obj2 in layout.structure_moving_object_pairs:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = obj1.radii.reshape(-1, 1)

        positions_b = positions_dict[str(obj2)][0]
        radii_b = obj2.radii.reshape(-1, 1)

        dist = min_spheres_spheres_interference(positions_a, radii_a, positions_b, radii_b)

        interferences.append(dist)

        max_interference = max(interferences)

    return max_interference

