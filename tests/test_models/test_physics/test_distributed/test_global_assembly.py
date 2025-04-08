"""

Unit test for the mesh generation and stiffness matrix assembly. Some details are taken from the paper:

Brent Goplen and Sachin Sapatnekar.
Efficient Thermal Placement of Standard Cells in 3D ICs using a Force Directed Approach.
DOI 10.5555/996070.1009873
https://www.ece.umn.edu/~sachin/conf/iccad03bg.pdf

Note: This paper contains a typographic error in the stiffness matrix. Terms "F" and "E" are swapped in the matrix.
This is first noticeable when enumerated as A, B, C, D, F, E, G, H; and the error can be confirmed when comparing
with the local stiffness matrix in SPI2py.

Node ordering scheme for the natural coordinates of an element:

         (eta/y)
            |
            *----(xi/x)
           /
      (zeta/z)

            3--------------2
          / |            / |
        /   |          /   |
      7--------------6     |
      |     0--------|-----1
      |   /          |   /
      | /            | /
      4--------------5

A two-element mesh:

             Element 1       Element 2

          3--------------4--------------5
        / |            / |            / |
      /   |          /   |          /   |
    9--------------10-------------11    |
    |     0--------|-----1--------|-----2
    |   /          |   /          |   /
    | /            | /            | /
    6--------------7--------------8

"""

# Standard imports
import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO

# Local imports
from SPI2py.models.physics.distributed.mesh import generate_mesh
from SPI2py.models.physics.distributed.assembly import assemble_sparse_global_stiffness


@pytest.fixture
def fixture_hex8_scalar():

    """
    Eight-node hexahedral element for scalar fields as described in the following paper:

    Brent Goplen and Sachin Sapatnekar. Efficient Thermal Placement of Standard Cells in 3D ICs using a Force Directed
    Approach. DOI 10.5555/996070.1009873

    Note: This paper contains an obvious and easily verifiable typo, swapping terms "F" and "E", which this code fixes.
    """

    w = 1  # x
    h = 1  # y
    d = 1  # Z

    K = 1  # Base stiffness

    A =  4*h*d + 4*w*d + 4*w*h
    B = -4*h*d + 2*w*d + 2*w*h
    C = -2*h*d - 2*w*d +   w*h
    D =  2*h*d - 4*w*d + 2*w*h
    E =  2*h*d + 2*w*d - 4*w*h
    F = -2*h*d +   w*d - 2*w*h
    G =   -h*d -   w*d -   w*h
    H =    h*d - 2*w*d - 2*w*h

    k_ex = K/36 * jnp.array([[A, B, C, D, E, F, G, H],
                             [B, A, D, C, F, E, H, G],
                             [C, D, A, B, G, H, E, F],
                             [D, C, B, A, H, G, F, E],
                             [E, F, G, H, A, B, C, D],
                             [F, E, H, G, B, A, D, C],
                             [G, H, E, F, C, D, A, B],
                             [H, G, F, E, D, C, B, A]])

    return k_ex


def test_global_assembly(fixture_hex8_scalar):


    # Generate the mesh
    nodes_2e, elements_2e, _, _, _, _, _, _, _ = generate_mesh(0, 2, 0, 1, 0, 1, element_size=1.0)

    # Assemble the sparse global stiffness matrix
    Ke_flat, elem_indices, rows_flat, cols_flat, n_nodes, n_elem = assemble_sparse_global_stiffness(nodes_2e, elements_2e, base_k=1.0)
    indices = jnp.stack([rows_flat, cols_flat], axis=-1)
    K_global = BCOO((Ke_flat, indices), shape=(n_nodes, n_nodes))

    # Convert the sparse matrix to a dense matrix for easier comparison
    K_global_dense = K_global.todense()

    # Assemble the expected local stiffness matrices
    k_ex = fixture_hex8_scalar


    # Confirm that all entries that should be zeros are zeros

    zeros_i = jnp.array([[0, 0, 0, 0],
                         [2, 2, 2, 2],
                         [3, 3, 3, 3],
                         [5, 5, 5, 5],
                         [6, 6, 6, 6],
                         [8, 8, 8, 8],
                         [9, 9, 9, 9],
                         [11, 11, 11, 11]])

    zeros_j = jnp.array([[2, 5, 8, 11],
                         [0, 3, 6, 9],
                         [2, 5, 8, 11],
                         [0, 3, 6, 9],
                         [2, 5, 8, 11],
                         [0, 3, 6, 9],
                         [2, 5, 8, 11],
                         [0, 3, 6, 9]])

    assert jnp.all(K_global_dense[zeros_i, zeros_j] == 0.0)

    # And confirm that all other entries are not zeros

    non_zeros_i = jnp.array([0, 0, 0, 0, 0, 0, 0, 0,
                             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                             2, 2, 2, 2, 2, 2, 2, 2,
                             3, 3, 3, 3, 3, 3, 3, 3,
                             4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                             5, 5, 5, 5, 5, 5, 5, 5,
                             6, 6, 6, 6, 6, 6, 6, 6,
                             7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
                             8, 8, 8, 8, 8, 8, 8, 8,
                             9, 9, 9, 9, 9, 9, 9, 9,
                             10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
                             11, 11, 11, 11, 11, 11, 11, 11])

    non_zeros_j = jnp.array([0, 1, 3, 4, 6, 7, 9, 10,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                             1, 2, 4, 5, 7, 8, 10, 11,
                             0, 1, 3, 4, 6, 7, 9, 10,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                             1, 2, 4, 5, 7, 8, 10, 11,
                             0, 1, 3, 4, 6, 7, 9, 10,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                             1, 2, 4, 5, 7, 8, 10, 11,
                             0, 1, 3, 4, 6, 7, 9, 10,
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                             1, 2, 4, 5, 7, 8, 10, 11])

    assert jnp.all(K_global_dense[non_zeros_i, non_zeros_j] != 0.0)

    # Now spot check some values; K_e1 == K_e2 == K_ex

    # K[0,0] == K_e1[0,0]
    assert jnp.isclose(K_global_dense[0, 0], k_ex[0, 0])

    # K[1,1] == K_e1[1,1] + K_e2[0,0]
    assert jnp.isclose(K_global_dense[1, 1], k_ex[1, 1] + k_ex[0, 0])

    # K[0,9] == K_e1[0,7] and K[9,0] == K_e2[7,0]
    assert jnp.isclose(K_global_dense[0, 9], k_ex[0, 7])

    # K[11,11] == K_e2[6,6]
    assert jnp.isclose(K_global_dense[11, 11], k_ex[6, 6])

def test_apply_bc():

    # test_mesh.py already verifies that find_face_nodes works, so we don't need to test it here for getting BC nodes.


    pass