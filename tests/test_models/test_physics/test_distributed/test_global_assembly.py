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
from SPI2py.models.physics.distributed.assembly import construct_global_stiffness_matrix, partition_sparse_matrix
from SPI2py.models.physics.distributed.assembly import (partition_vector,
                                                        assemble_global_system_partition)


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


def test_construct_global_Stiffness_matrix(fixture_hex8_scalar):


    # Generate the mesh
    nodes_2e, elements_2e, _, _, _, _, _, _, _ = generate_mesh(0, 2, 0, 1, 0, 1, element_size=1.0)

    # Assemble the global stiffness matrix and convert it to a dense matrix
    K_global = construct_global_stiffness_matrix(nodes_2e, elements_2e,
                                                 k=1.0,
                                                 pseudo_densities=jnp.ones(elements_2e.shape[0]))

    # Convert the sparse matrix to a dense matrix for easier comparison
    K_global_dense = K_global.todense()

    # Assemble the expected local stiffness matrices
    k_ex = fixture_hex8_scalar

    # Manually define global stiffness matrix indices that should be zero
    # (due to the mesh topology, not from zero terms in local stiffness matrices)
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

    # Confirm that all entries that should be zero (due to mesh topology) are zero
    assert jnp.all(jnp.isclose(K_global_dense[zeros_i, zeros_j], 0.0))

    # And for sanity, confirm that all other entries are not zeros

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

    # Kept for reference.
    # Don't try the reverse check because a 8x8 local stiffness matrix might contain zero terms, so asserting that
    # all "non-zero" terms are non-zero can raise an error if you don't account for this.
    # assert not jnp.any(jnp.isclose(K_global_dense[non_zeros_i, non_zeros_j], 0.0))

    # Verify the sparse matrix
    sparse_indices = K_global.indices
    sparse_indices_expected = jnp.hstack((non_zeros_i.reshape(-1, 1), non_zeros_j.reshape(-1, 1)))

    assert sparse_indices.shape == sparse_indices_expected.shape
    assert jnp.all(sparse_indices.sort() == sparse_indices_expected.sort())


    # Now spot check some values; K_e1 == K_e2 == K_ex

    # K[0,0] == K_e1[0,0]
    assert jnp.isclose(K_global_dense[0, 0], k_ex[0, 0])

    # K[0,1] == K_e1[0,1] == 0 (or near-zero)
    assert jnp.isclose(K_global_dense[0, 1], k_ex[0, 1])

    # K[1,1] == K_e1[1,1] + K_e2[0,0]
    assert jnp.isclose(K_global_dense[1, 1], k_ex[1, 1] + k_ex[0, 0])

    # K[0,9] == K_e1[0,7] and K[9,0] == K_e2[7,0]
    assert jnp.isclose(K_global_dense[0, 9], k_ex[0, 7])

    # K[11,11] == K_e2[6,6]
    assert jnp.isclose(K_global_dense[11, 11], k_ex[6, 6])


def test_partition_global_stiffness_matrix():

    # Generate the mesh
    nodes_2e, elements_2e, _, _, _, _, _, _, _ = generate_mesh(0, 2, 0, 1, 0, 1, element_size=1.0)

    # Assemble the global stiffness matrix and convert it to a dense matrix
    K_global = construct_global_stiffness_matrix(nodes_2e, elements_2e,
                                                 k=1.0,
                                                 pseudo_densities=jnp.ones(elements_2e.shape[0]))

    # Convert the sparse matrix to a dense matrix for easier comparison
    K_global_dense = K_global.todense()

    # Verify the partitioning scheme
    idx_f = jnp.array([3, 4, 5, 9, 10, 11])
    idx_p = jnp.array([0, 1, 2, 6, 7, 8])
    idx_f_2d = jnp.array([[3, 4, 5, 9, 10, 11]])
    idx_p_2d = jnp.array([[0, 1, 2, 6, 7, 8]])
    n_f = idx_f_2d.shape[1]
    n_p = idx_p_2d.shape[1]

    idx_ff = jnp.stack((jnp.repeat(idx_f_2d, n_f).flatten(), jnp.repeat(idx_f_2d, n_f, axis=0).flatten()), axis=1)
    idx_fp = jnp.stack((jnp.repeat(idx_f_2d, n_p).flatten(), jnp.repeat(idx_p_2d, n_f, axis=0).flatten()), axis=1)
    idx_pf = jnp.stack((jnp.repeat(idx_p_2d, n_f).flatten(), jnp.repeat(idx_f_2d, n_p, axis=0).flatten()), axis=1)
    idx_pp = jnp.stack((jnp.repeat(idx_p_2d, n_p).flatten(), jnp.repeat(idx_p_2d, n_p, axis=0).flatten()), axis=1)

    idx_ff_expected = jnp.array([[3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 10, 10, 10,
                                  10, 10, 10, 11, 11, 11, 11, 11, 11],
                                 [3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4,
                                  5, 9, 10, 11, 3, 4, 5, 9, 10, 11]]).T

    idx_fp_expected = jnp.array([[3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 9, 9, 9, 9, 9, 9, 10, 10, 10,
                                  10, 10, 10, 11, 11, 11, 11, 11, 11],
                                 [0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7,
                                  8, 0, 1, 2, 6, 7, 8]]).T

    idx_pf_expected = jnp.array(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
         [3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4, 5, 9, 10, 11, 3, 4, 5,
          9, 10, 11]]).T

    idx_pp_expected = jnp.array(
        [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8],
         [0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7, 8, 0, 1, 2, 6, 7,
          8]]).T

    assert jnp.all(idx_ff == idx_ff_expected)
    assert jnp.all(idx_fp == idx_fp_expected)
    assert jnp.all(idx_pf == idx_pf_expected)
    assert jnp.all(idx_pp == idx_pp_expected)

    # And verify
    K_ff_data_expected = K_global_dense[idx_ff[:, 0], idx_ff[:, 1]]
    K_fp_data_expected = K_global_dense[idx_fp[:, 0], idx_fp[:, 1]]
    K_pf_data_expected = K_global_dense[idx_pf[:, 0], idx_pf[:, 1]]
    K_pp_data_expected = K_global_dense[idx_pp[:, 0], idx_pp[:, 1]]

    K_ff, K_fp, K_pf, K_pp = partition_sparse_matrix(K_global, idx_f, idx_p)

    # We've already verified sparsity indices, so converting partitions to their dense forms makes comparisons easier.
    k_ff_data = K_ff.todense().flatten()
    k_fp_data = K_fp.todense().flatten()
    k_pf_data = K_pf.todense().flatten()
    k_pp_data = K_pp.todense().flatten()

    # Verify the partitions match up as expected.
    assert jnp.all(K_ff_data_expected.shape == k_ff_data.shape)
    assert jnp.all(K_fp_data_expected.shape == k_fp_data.shape)
    assert jnp.all(K_pf_data_expected.shape == k_pf_data.shape)
    assert jnp.all(K_pp_data_expected.shape == k_pp_data.shape)

    assert jnp.all(jnp.isclose(K_ff_data_expected, k_ff_data))
    assert jnp.all(jnp.isclose(K_fp_data_expected, k_fp_data))
    assert jnp.all(jnp.isclose(K_pf_data_expected, k_pf_data))
    assert jnp.all(jnp.isclose(K_pp_data_expected, k_pp_data))


def test_partition_vector():
    pass


def test_apply_bc():

    # test_mesh.py already verifies that find_face_nodes works, so we don't need to test it here for getting BC nodes.


    pass