# Standard imports
import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jax.scipy.sparse.linalg import cg

# Local imports
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_face_nodes
from SPI2py.models.physics.distributed.assembly import assemble_global_system_partition
from SPI2py.models.physics.distributed.solver import solve_system_partition
from SPI2py.models.physics.distributed.assembly import construct_global_stiffness_matrix

"""
A simple 1x1x1 metal cube with its bottom face held at a fixed temperature (Dirichlet BC) and its top face subject
to convection (Robin BC).
"""

T_inf = 293  # ambient temperature (K)
T_fixed = 573  # fixed temperature on bottom boundary (K)

h_c = 21  # convection coefficient (W / (m^2 * K))
k = 56.00  # thermal conductivity (W / (m * K))
heat_source = 25000.00  # Heat generation (W / m^2)

# # Mesh generation
# x_min, x_max = (0, 1)
# y_min, y_max = (0, 1)
# z_min, z_max = (0, 1)
# element_size = 0.25

# Generate the mesh
w = 1  # x
h = 1  # y
d = 1  # Z
# element_size = 1.0
element_size = 0.1
nodes, elements, _, _, _, _, _, _, _ = generate_mesh(0, 2 * w, 0, h, 0, d, element_size=element_size)

densities = jnp.ones(elements.shape[0])  # uniform density
heat_loads = jnp.zeros(elements.shape[0])  # no heat generation

# Find active nodes and face nodes
top_normal = jnp.array([0, 1, 0])
bottom_normal = jnp.array([0, -1, 0])
right_normal = jnp.array([1, 0, 0])

robin_nodes = find_face_nodes(nodes, top_normal)
dirichlet_nodes = find_face_nodes(nodes, bottom_normal)
heat_nodes = find_face_nodes(nodes, right_normal)

idx = jnp.arange(nodes.shape[0])
idx_p = dirichlet_nodes
idx_f = jnp.setdiff1d(idx, idx_p)

# r_nodes_expected = jnp.array([3, 4, 5, 9, 10, 11])
# d_nodes_expected = jnp.array([0, 1, 2, 6, 7, 8])

robin_area = element_size * element_size

dirichlet_temperature = T_fixed * jnp.ones(dirichlet_nodes.shape[0])

# Face is 1 m^2. If there were for nodes, nodal contribution would be w/4.
# heat (W / m^2) * area (m^2) = total heat (W) --> nodal contribution (W) / n_nodes
face_heat_loads = heat_source * jnp.ones(heat_nodes.shape[0]) / heat_nodes.size

# Initialize global stiffness matrix and force vector
K, f, u = assemble_global_system_partition(nodes, elements,
                                            base_k=k,
                                            r_nodes=robin_nodes,
                                            r_h=h_c,
                                            r_T_inf=T_inf,
                                            r_area=robin_area,
                                            heat_nodes=heat_nodes,
                                            heat_loads=face_heat_loads,
                                            d_nodes=dirichlet_nodes,
                                            d_T=dirichlet_temperature)


u = solve_system_partition(K, f, u, idx_f, idx_p)


# K_ff, K_fp, K_pf, K_pp = K
# f_f, f_p = f
# _, u_p = u


# Solve the global system using a sparse solver.
# rhs = (f_f - K_fp @ u_p)
# u_f, _ = cg(K_ff, rhs)

# u_unordered = jnp.vstack((u_f.reshape(-1, 1), u_p.reshape(-1, 1)))

print("Maximum temperature: ", jnp.max(u))
print("Minimum temperature: ", jnp.min(u))

print("Done")
