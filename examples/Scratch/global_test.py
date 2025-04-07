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

    y
    |
    0----x
  /
 z

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


* | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11
---------------------------------------------------
0 | - |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
1 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
2 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
3 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
4 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
5 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
6 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
7 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
8 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
9 |   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
10|   |   |   |   |   |   |   |   |   |   |    |
---------------------------------------------------
11|   |   |   |   |   |   |   |   |   |   |    |



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

