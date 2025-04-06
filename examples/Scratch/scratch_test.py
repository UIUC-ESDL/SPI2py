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

import numpy as np
import jax.numpy as jnp
from SPI2py.models.physics.distributed.mesh import generate_mesh
from SPI2py.models.physics.distributed.quadrature import shape_functions, gauss_quad
from SPI2py.models.physics.distributed.element import assemble_local_stiffness_matrix_scalar
from SPI2py.models.physics.distributed.assembly import assemble_base_global_stiffness


# Verify a single-element mesh


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

assert jnp.all(jnp.isclose(nodes_ex_1e, nodes_1e[elements_1e]))


# Verify a multi-element mesh (2 elements in x-direction)


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

element_1 = elements_2e[0]
element_2 = elements_2e[1]

nodes_el_1 = nodes_2e[jnp.array([0, 1, 4, 3, 6, 7, 10, 9])]
nodes_el_2 = nodes_2e[jnp.array([1, 2, 5, 4, 7, 8, 11, 10])]

assert jnp.all(jnp.isclose(nodes_ex_el_1, nodes_el_1))
assert jnp.all(jnp.isclose(nodes_ex_el_2, nodes_el_2))


# Verify local stiffness matrix


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


gauss_pts, gauss_wts = gauss_quad()

# k_SPI2py_ex_nodes = assemble_local_stiffness_matrix_scalar(nodes_ex_el_1, K, gauss_pts, gauss_wts)
k_SPI2py = assemble_local_stiffness_matrix_scalar(nodes_1e[elements_1e[0]], K, gauss_pts, gauss_wts)

# Assert
assert jnp.all(jnp.isclose(k_ex, k_SPI2py))


# Verify Global stiffness matrix...


# Ke_flat, elem_indices, rows_flat, cols_flat, n_nodes, _ = assemble_base_global_stiffness