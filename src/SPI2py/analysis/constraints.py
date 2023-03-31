"""Constraint Functions

This module contains functions that are used to calculate the constraint functions for a given spatial configuration.
"""

import numpy as np
from .distance import distances_points_points, signed_distances_spheres_spheres
from scipy.optimize import NonlinearConstraint


def signed_distances(x, spatial_configuration, pairs):
    """
    Returns the signed distances between all pairs of objects in the layout.

    To be consistent with constraint function notation, this function returns negative values
    for objects that are not interfering with each other, and positive values for objects that
    are interfering with each other.

    TODO Preallocate array and vectorize this function
    TODO Write unit tests

    :param x: Design vector (1D array)
    :param spatial_configuration: The SpatialConfiguration object used to query positions at x
    :param pairs: The list of object pairs to calculate the signed distance between
    :return: An array of signed distances between each object pair
    """
    # Calculate the positions of all spheres in layout given design vector x
    positions_dict = spatial_configuration.calculate_positions(x)

    # Calculate the interferences between each sphere of each object pair
    all_signed_distances = []

    for obj1, obj2 in pairs:
        positions_a = positions_dict[str(obj1)][0]
        radii_a = positions_dict[str(obj1)][1]

        positions_b = positions_dict[str(obj2)][0]
        radii_b = positions_dict[str(obj2)][1]

        # # TODO Reformat Radii shape so we don't have to keep reshaping it
        # dPositions = distances_points_points(positions_a, positions_b)
        # dRadii     = distances_points_points(radii_a.reshape(-1, 1), radii_b.reshape(-1, 1))
        #
        # all_signed_distances.append(dRadii - dPositions)
        signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b).flatten()


        all_signed_distances.append(signed_distances)

    all_signed_distances = np.concatenate(all_signed_distances, axis=0)

    return all_signed_distances

def format_constraints(layout,
                       constraint_function,
                       constraint_aggregation_function,
                       config):

    object_pairs = layout.system.object_pairs

    # Unpack the config dictionary
    check_collisions      = list(config['analysis']['check collisions'].values())
    collision_tolerances  = list(config['analysis']['collision tolerance'].values())

    # Add the applicable interference constraints
    nlcs = []
    if check_collisions[0] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_aggregation_function(constraint_function(x, layout, object_pairs[0])), -np.inf,
                                        collision_tolerances[0]))
    if check_collisions[1] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_aggregation_function(constraint_function(x, layout, object_pairs[1])), -np.inf,
                                        collision_tolerances[1]))
    if check_collisions[2] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_aggregation_function(constraint_function(x, layout, object_pairs[2])), -np.inf,
                                        collision_tolerances[2]))
    if check_collisions[3] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_aggregation_function(constraint_function(x, layout, object_pairs[3])), -np.inf,
                                        collision_tolerances[3]))
    if check_collisions[4] is True:
        nlcs.append(NonlinearConstraint(lambda x: constraint_aggregation_function(constraint_function(x, layout, object_pairs[4])), -np.inf,
                                        collision_tolerances[4]))

    return nlcs