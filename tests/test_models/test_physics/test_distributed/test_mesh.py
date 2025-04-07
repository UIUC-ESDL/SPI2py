"""

Unit test for the mesh generation and stiffness matrix assembly. Some details are taken from the paper:

Brent Goplen and Sachin Sapatnekar.
Efficient Thermal Placement of Standard Cells in 3D ICs using a Force Directed Approach.
DOI 10.5555/996070.1009873
https://www.ece.umn.edu/~sachin/conf/iccad03bg.pdf

Note: This paper contains a typographic error in the stiffness matrix. Terms "F" and "E" are swapped in the matrix.
This is first noticeable when enumerated as A, B, C, D, F, E, G, H; and the error can be confirmed when comparing
with the local stiffness matrix in SPI2py.

Node numbering for a single element:

          3--------------2
        / |            / |
      /   |          /   |
    7--------------6     |
    |     0--------|-----1
    |   /          |   /
    | /            | /
    4--------------5

Local stiffness matrix for two elements:

          3--------------2--------------9
        / |            / |            / |
      /   |          /   |          /   |
    7--------------6--------------11    |
    |     0--------|-----1--------|-----8
    |   /          |   /          |   /
    | /            | /            | /
    4--------------5--------------10

"""

# Standard imports
import jax.numpy as jnp

# Local imports
from SPI2py.models.physics.distributed.mesh import generate_mesh


def test_mesh_generation_1_element():

    w = 1  # x
    h = 1  # y
    d = 1  # Z

    nodes_ex_1e = jnp.array([[0, 0, 0],  # 0
                             [w, 0, 0],  # 1
                             [w, h, 0],  # 2
                             [0, h, 0],  # 3
                             [0, 0, d],  # 4
                             [w, 0, d],  # 5
                             [w, h, d],  # 6
                             [0, h, d]])  # 7

    nodes_1e, elements_1e, _, _, _, _, _, _, _ = generate_mesh(0, w, 0, h, 0, d, element_size=1.0)

    # Map the nodes to their natural coordinate ordering
    nodes_1e_mapped = nodes_1e[elements_1e]

    assert jnp.all(jnp.isclose(nodes_ex_1e, nodes_1e_mapped))


def test_mesh_generation_2_elements():

    w = 1  # x
    h = 1  # y
    d = 1  # Z

    nodes_ex_el_1 = jnp.array([[0, 0, 0],   # 0
                               [w, 0, 0],   # 1
                               [w, h, 0],   # 2
                               [0, h, 0],   # 3
                               [0, 0, d],   # 4
                               [w, 0, d],   # 5
                               [w, h, d],   # 6
                               [0, h, d]])  # 7

    nodes_ex_el_2 = jnp.array([[w, 0, 0],    # 0
                               [2*w, 0, 0],  # 1
                               [2*w, h, 0],  # 2
                               [w, h, 0],    # 3
                               [w, 0, d],    # 4
                               [2*w, 0, d],  # 5
                               [2*w, h, d],  # 6
                               [w, h, d]])   # 7

    nodes_2e, elements_2e, _, _, _, _, _, _, _ = generate_mesh(0, 2*w, 0, h, 0, d, element_size=1.0)

    # GLOBAL ORDERING SCHEME


    # LOCAL ORDERING SCHEME


    element_1 = elements_2e[0]
    element_2 = elements_2e[1]
    element_1_expected_ordering = jnp.array([0, 1, 4, 3, 6, 7, 10, 9])
    element_2_expected_ordering = jnp.array([1, 2, 5, 4, 7, 8, 11, 10])

    # Verify that elements correctly map nodes to their natural coordinate ordering
    assert jnp.all(jnp.isclose(element_1_expected_ordering, element_1))
    assert jnp.all(jnp.isclose(element_2_expected_ordering, element_2))

    # Verify that the nodes are correctly mapped to their natural coordinate ordering
    nodes_el_1 = nodes_2e[element_1]
    nodes_el_2 = nodes_2e[element_2]

    assert jnp.all(jnp.isclose(nodes_ex_el_1, nodes_el_1))
    assert jnp.all(jnp.isclose(nodes_ex_el_2, nodes_el_2))
