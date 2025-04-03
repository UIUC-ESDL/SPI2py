"""

Brent Goplen and Sachin Sapatnekar.
Efficient Thermal Placement of Standard Cells in 3D ICs using a Force Directed Approach.
DOI 10.5555/996070.1009873
https://www.ece.umn.edu/~sachin/conf/iccad03bg.pdf

Note: This paper contains a typographic error in the stiffness matrix. Terms "F" and "E" are swapped in the matrix.
This is first noticeable when enumerated as A, B, C, D, F, E, G, H; and the error can be confirmed when comparing
with the local stiffness matrix in SPI2py.

"""

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from SPI2py.models.physics.distributed.mesh import generate_mesh
from SPI2py.models.physics.distributed.quadrature import shape_functions, gauss_quad
from SPI2py.models.physics.distributed.element import assemble_local_stiffness_matrix_scalar
from SPI2py.models.physics.distributed.assembly import assemble_base_global_stiffness


# Verify single-element mesh

w = 1
h = 1
d = 1
nodes_ex = jnp.array([[0, 0, 0],
                      [w, 0, 0],
                      [w, h, 0],
                      [0, h, 0],
                      [0, 0, d],
                      [w, 0, d],
                      [w, h, d],
                      [0, h, d]])

nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh(0, w, 0, h, 0, d, element_size=1.0)

jnp.isclose(nodes_ex, nodes[elements])

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

k_SPI2py = assemble_local_stiffness_matrix_scalar(nodes, K, gauss_pts, gauss_wts)

# Assert
jnp.isclose(k_ex, k_SPI2py)


# Verify Global stiffness matrix...


# Ke_flat, elem_indices, rows_flat, cols_flat, n_nodes, n_elem = assemble_base_global_stiffness