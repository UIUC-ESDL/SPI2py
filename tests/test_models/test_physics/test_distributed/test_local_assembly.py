"""

Unit test for the mesh generation and stiffness matrix assembly. Some details are taken from the paper:

Brent Goplen and Sachin Sapatnekar.
Efficient Thermal Placement of Standard Cells in 3D ICs using a Force Directed Approach.
DOI 10.5555/996070.1009873
https://www.ece.umn.edu/~sachin/conf/iccad03bg.pdf

Note: This paper contains a typographic error in the stiffness matrix. Terms "F" and "E" are swapped in the matrix.
This is first noticeable when enumerated as A, B, C, D, F, E, G, H; and the error can be confirmed when comparing
with the local stiffness matrix in SPI2py.

Local stiffness matrix for a single element:

          3--------------2
        / |            / |
      /   |          /   |
    7--------------6     |
    |     0--------|-----1
    |   /          |   /
    | /            | /
    4--------------5

"""

import jax.numpy as jnp
from SPI2py.models.physics.distributed.mesh import generate_mesh
from SPI2py.models.physics.distributed.quadrature import shape_functions, gauss_quad
from SPI2py.models.physics.distributed.element import assemble_local_stiffness_matrix_scalar


def test_hex8_scalar():

    """
    Construct and test an 8-node hexahedral element for scalar fields such as temperature.
    """

    # Example from the paper

    w = 1  # x
    h = 1  # y
    d = 1  # Z

    K = 1

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

    # Now our assembly

    gauss_pts, gauss_wts = gauss_quad()

    nodes_1e, elements_1e, _, _, _, _, _, _, _ = generate_mesh(0, w, 0, h, 0, d, element_size=1.0)
    element_1 = elements_1e[0]

    k_SPI2py = assemble_local_stiffness_matrix_scalar(nodes_1e[element_1], K, gauss_pts, gauss_wts)

    assert jnp.all(jnp.isclose(k_ex, k_SPI2py))

    # TODO Same number of nonzero terms



def test_hex8_vector():
    """
    Construct and test an 8-node hexahedral element for vector fields such as displacement.
    """
    pass
