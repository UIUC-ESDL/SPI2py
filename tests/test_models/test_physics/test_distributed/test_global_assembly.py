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