import numpy as np
import networkx as nx


def generate_random_layout(layout):
    """
    Generates random layouts using a force-directed algorithm

    Initially assumes 0 rotation and 1x6 design vector
    TODO Create a separate file for topology enumeration, etc.


    :return:
    """

    g = nx.MultiGraph()
    g.add_nodes_from(layout.nodes)
    g.add_edges_from(layout.edges)

    # Optimal distance between nodes
    k = 1

    scale = 6

    # TODO remove this random number seed for actual problems
    seed = 6

    # Dimension of layout
    dim = 3

    positions = nx.spring_layout(g, k=k, dim=dim, scale=scale, seed=seed)

    # Generate random angles too?

    # Temporarily pad zeros for rotations
    design_vectors = []
    rotation = np.array([0, 0, 0])

    for i in positions:
        position = positions[i]
        design_vector = np.concatenate((position, rotation))
        design_vectors.append(design_vector)

    # Flatten design vectors
    # TODO Make more efficient?
    design_vector = np.concatenate(design_vectors)

    return design_vector