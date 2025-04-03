"""
Heat and Mass Transfer, 5th Edition
Example 5-4 Heat Loss through Chimneys
"""

import numpy as np
import matplotlib.pyplot as plt
from SPI2py.models.physics.distributed.element import assemble_local_stiffness_matrix_scalar
from SPI2py.models.physics.distributed.quadrature import shape_functions

# Parameters
k = 1.4  # thermal conductivity, concrete (W * m * K)
size_c = 0.20  # size of chimney (m)
size_w = 0.60  # size of wall (m)
T_i = 300  # temperature, hot gas in chimney (C)
T_o = 20  # temperature, ambient air (C)
h_i = 70  # convection coefficient, inside chimney (W / m^2 * K)
h_o = 21  # convection coefficient, outside wall (W / m^2 * K)

# Convert temperatures to Kelvin
T_i += 273
T_o += 273


n_elements = 20  # number of elements along one side
k = 200  # thermal conductivity (W / m * K)
h_c = 10  # convection coefficient (W / m^2 * K)
T_inf = 293  # ambient temperature (K)
q = 1000  # heat generation (W/m³)
T_fixed = 300  # fixed temperature on bottom boundary (K)

# # Mesh generation
# n_nodes = n_elements + 1
# dx = length / n_elements
# x = np.linspace(0, length, n_nodes)
# y = np.linspace(0, length, n_nodes)
# X, Y = np.meshgrid(x, y)  # Create 2D grid
# nodes = np.column_stack((X.ravel(), Y.ravel()))  # Flatten into list of (x, y)
#
# # nodes = [(i, j) for j in y for i in x]
# n_total_nodes = len(nodes)
#
# # Create elements (connectivity of nodes)
# elements = []
# for j in range(n_elements):  # Loop over rows of elements
#     for i in range(n_elements):  # Loop over columns of elements
#         n1 = j * (n_elements + 1) + i  # Bottom-left node
#         n2 = n1 + 1  # Bottom-right node
#         n3 = n1 + n_elements + 1  # Top-left node
#         n4 = n3 + 1  # Top-right node
#         elements.append([n1, n2, n4, n3])
#
# # Initialize global stiffness matrix and force vector
# K = np.zeros((n_total_nodes, n_total_nodes))
# F = np.zeros(n_total_nodes)
#
# # Helper functions for 2D shape functions
# def local_stiffness(dx, dy, k):
#     """2D bilinear element stiffness matrix for heat conduction."""
#     coeff = k / (dx * dy)
#     return coeff * np.array([
#         [ 2, -1, -1,  0],
#         [-1,  2,  0, -1],
#         [-1,  0,  2, -1],
#         [ 0, -1, -1,  2]
#     ])
#
# def local_force(dx, dy, q):
#     """2D bilinear element force vector for heat generation."""
#     coeff = q * dx * dy / 4
#     return coeff * np.ones(4)
#
# # Assemble global stiffness matrix and force vector
# for element in elements:
#     n1, n2, n3, n4 = element
#     stiffness = local_stiffness(dx, dx, k)
#     force = local_force(dx, dx, q)
#     global_nodes = [n1, n2, n3, n4]
#
#     for i in range(4):
#         F[global_nodes[i]] += force[i]
#         for j in range(4):
#             K[global_nodes[i], global_nodes[j]] += stiffness[i, j]
#
# # Apply boundary conditions (Dirichlet BC on bottom surface)
# bottom_nodes = [i for i in range(n_nodes)]
# for node in bottom_nodes:
#     K[node, :] = 0
#     K[node, node] = 1
#     F[node] = T_fixed
#
# # Apply convection (Neumann BCs) on other boundaries
# top_nodes = [i + n_nodes * n_elements for i in range(n_nodes)]
# left_nodes = [i * n_nodes for i in range(n_nodes)]
# right_nodes = [(i + 1) * n_nodes - 1 for i in range(n_nodes)]
# convection_nodes = set(top_nodes + left_nodes + right_nodes)
#
# for node in convection_nodes:
#     K[node, node] += h_c * dx
#     F[node] += h_c * T_inf * dx
#
#
# # Define the size and position of the fixed-temperature square
# square_size = 4  # Number of elements along one side of the square
# square_start_x = int(n_elements / 2 - square_size / 2)  # Center square
# square_start_y = int(n_elements / 2 - square_size / 2)
# square_end_x = square_start_x + square_size
# square_end_y = square_start_y + square_size
#
# # Identify nodes inside the fixed-temperature square
# fixed_square_nodes = []
# for j in range(square_start_y, square_end_y + 1):
#     for i in range(square_start_x, square_end_x + 1):
#         node = j * n_nodes + i
#         fixed_square_nodes.append(node)
#
# # Apply Dirichlet boundary conditions for the fixed-temperature square
# T_fixed_square = 350  # Fixed temperature for the square
# for node in fixed_square_nodes:
#     K[node, :] = 0
#     K[node, node] = 1
#     F[node] = T_fixed_square
#
# # Optional: Adjust heat generation for the square region if necessary
# # For simplicity, assume no heat generation in the square
# for element in elements:
#     n1, n2, n3, n4 = element
#     if any(node in fixed_square_nodes for node in element):
#         local_f = np.zeros(4)  # No heat generation in the fixed-temperature square
#         global_nodes = [n1, n2, n3, n4]
#         for i in range(4):
#             F[global_nodes[i]] += local_f[i]
#
# # Solve for temperature distribution
# T = np.linalg.solve(K, F)
#
# # Post-process and visualize results
# T_grid = T.reshape((n_nodes, n_nodes))
# plt.figure(figsize=(8, 6))
# plt.contourf(x, y, T_grid, levels=50, cmap="inferno")
# plt.colorbar(label="Temperature (K)")
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.title("Temperature Distribution")
#
# # Plot elements
# for element in elements:
#     coords = nodes[element]  # Get coordinates of element nodes
#     x_coords = np.append(coords[:, 0], coords[0, 0])  # Close the square
#     y_coords = np.append(coords[:, 1], coords[0, 1])
#     plt.plot(x_coords, y_coords, 'k-', lw=0.5)
#
# # Highlight the fixed-temperature nodes
# fixed_square_x = [nodes[n][0] for n in fixed_square_nodes]
# fixed_square_y = [nodes[n][1] for n in fixed_square_nodes]
# plt.scatter(fixed_square_x, fixed_square_y, c='red', label='Fixed Temperature (350K)')
# # plt.legend()
#
#
# plt.show()



