import jax.numpy as jnp
import matplotlib.pyplot as plt
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_active_nodes, find_face_nodes
from SPI2py.models.physics.distributed.element import assemble_local_stiffness_matrix_scalar
from SPI2py.models.physics.distributed.assembly import assemble_base_global_system_partition, apply_bc_partition
from SPI2py.models.physics.distributed.quadrature import shape_functions
from SPI2py.models.physics.distributed.solver import solve_system_partition



"""
A simple 1x1x1 metal cube with its bottom face held at a fixed temperature (Dirichlet BC) and its top face subject
to convection (Robin BC).
"""

T_inf = 293  # ambient temperature (K)
T_fixed = 573  # fixed temperature on bottom boundary (K)

h_c = 21  # convection coefficient (W / (m^2 * K))
k = 56.00  # thermal conductivity (W / (m * K))

# Mesh generation
x_min, x_max = (0, 1)
y_min, y_max = (0, 1)
z_min, z_max = (0, 1)

element_size = 0.25

nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max,
                                                                 element_size=element_size)

densities = jnp.ones(elements.shape[0])  # uniform density
heat_loads = jnp.zeros(elements.shape[0])  # no heat generation

# Find active nodes and face nodes
dirichlet_nodes = find_face_nodes(nodes, jnp.array([0.0, 0.0, -1.0]))
robin_nodes = find_face_nodes(nodes, jnp.array([0.0, 0.0, 1.0]))
robin_area = element_size * element_size

dirichlet_temperature = T_fixed * jnp.ones(dirichlet_nodes.shape[0])

# Initialize global stiffness matrix and force vector
K_base, f_base, elem_indices = assemble_base_global_system_partition(nodes, elements,
                                                                     base_k=k,
                                                                     r_nodes=robin_nodes,
                                                                     r_h=h_c,
                                                                     r_T_inf=T_inf,
                                                                     r_area=robin_area,
                                                                     d_nodes=dirichlet_nodes,
                                                                     d_T=dirichlet_temperature)

idx = jnp.arange(nodes.shape[0])
idx_p = dirichlet_nodes
idx_f = jnp.setdiff1d(idx, idx_p)

u_p = dirichlet_temperature


K_bc, f_bc = apply_bc_partition(K_base, f_base,
                                r_nodes=robin_nodes,
                                r_h=h_c,
                                r_area=robin_area,
                                r_T_inf=T_inf,
                                idx_f=idx_f, idx_p=idx_p)

# And test update...
u = solve_system_partition(densities, heat_loads,
                           nodes, elements,
                           K_bc, f_bc, u_p,
                           elem_indices, idx_f, idx_p)