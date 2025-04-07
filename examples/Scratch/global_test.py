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

* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7
----------------------------------
0 |   |   |   |   |   |   |   |
----------------------------------
1 |   |   |   |   |   |   |   |
----------------------------------
2 |   |   |   |   |   |   |   |
----------------------------------
3 |   |   |   |   |   |   |   |
----------------------------------
4 |   |   |   |   |   |   |   |
----------------------------------
5 |   |   |   |   |   |   |   |
----------------------------------
6 |   |   |   |   |   |   |   |
----------------------------------
7 |   |   |   |   |   |   |   |

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


* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11
---------------------------------------------------
0 |e1 |e1 |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
1 |e1 |e12|   |   |   |   |   |   |   |   |    |
---------------------------------------------------
2 |   |   |e2 |   |   |   |   |   |   |   |    |
---------------------------------------------------
3 |   |   |   |e1 |   |   |   |   |   |   |    |
---------------------------------------------------
4 |   |   |   |   |e12|   |   |   |   |   |    |
---------------------------------------------------
5 |   |   |   |   |   |e2 |   |   |   |   |    |
---------------------------------------------------
6 |   |   |   |   |   |   |e1 |   |   |   |    |
---------------------------------------------------
7 |   |   |   |   |   |   |   |e12|   |   |    |
---------------------------------------------------
8 |   |   |   |   |   |   |   |   |e2 |   |    |
---------------------------------------------------
9 |   |   |   |   |   |   |   |   |   |e1 |    |
---------------------------------------------------
10|   |   |   |   |   |   |   |   |   |   |e12 |
---------------------------------------------------
11|   |   |   |   |   |   |   |   |   |   |    |e2

"""

# Standard imports
import jax.numpy as jnp

# Local imports
from SPI2py.models.physics.distributed.mesh import generate_mesh

w = 1  # x
h = 1  # y
d = 1  # Z

nodes_2e, elements_2e, _, _, _, _, _, _, _ = generate_mesh(0, 2*w, 0, h, 0, d, element_size=1.0)

element_1 = elements_2e[0]
element_2 = elements_2e[1]

# Assemble the expected global and local stiffness matrices

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


K_global_expected = jnp.zeros((12, 12))
K_global_expected[]


