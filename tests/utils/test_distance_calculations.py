import numpy as np
from scipy.spatial.kdtree import
from scipy.spatial.distance import cdist, pdist

def min_cdist():
    pass


def min_kdtree_distance(tree, positions):
    """
    Returns the minimum distance between a KD-Tree and object

    :param tree:
    :param positions:
    :return:
    """

    dist, _  = tree.query(positions)
    min_dist = np.min(dist)

    return min_dist


def min_point_line_distance():
    pass


def min_line_line_distance():
    pass


def calculate_gap(radius1, radius2, min_dist):
    """
    Calculate the gap between two spheres

    gap<0 means no overlap
    gap=0 means tangent
    gap>0 means overlap

    :param radius1:
    :param radius2:
    :param min_dist:
    :return:
    """

    return radius1 + radius2 - min_dist

