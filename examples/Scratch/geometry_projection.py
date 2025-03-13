from time import time_ns
import numpy as np
import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, vjp
import pyvista as pv
# from SPI2py.models.projection.grid import create_grid
from SPI2py.models.mechanics.homogenous_transformation import transform_points
from SPI2py.models.projection.projection import project_component, combine_densities
from SPI2py.models.utilities.visualization import plot_grid, plot_spheres, plot_AABB_spheres, plot_stl_file, plot_nodes
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.mesh import generate_mesh_vec, find_active_nodes, find_face_nodes
from SPI2py.models.physics.distributed.assembly import DirichletBC, RobinBC
from SPI2py.models.physics.distributed.mesh import generate_mesh_vec
from SPI2py.models.physics.distributed.solver import solve_system
from SPI2py.models.utilities.visualization import plot_temperature_distribution

import jax
jax.config.update("jax_enable_x64", True)

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
# jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
# jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
# jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")

# Create grid
# el_size = 0.5
el_size = 0.25
bounds = (0, 7.5, 0, 5, 0, 7.5)

x_min, x_max, y_min, y_max, z_min, z_max = bounds
nodes, elements, el_centers, nx, ny, nz, lx, ly, lz = generate_mesh_vec(x_min, x_max, y_min, y_max, z_min, z_max, element_size=el_size)
el_centers = el_centers.reshape(nx, ny, nz, 1, 3)

# Read the mesh kernel
kernel_pos, kernel_rad = create_uniform_kernel(1, mode='circumscription')
kernel_pos = kernel_pos.reshape(-1, 3)
kernel_rad = kernel_rad.reshape(-1, 1)

# Read the part model (Numbers for part 1)
# Slice by minimum radius instead of length to maintain kernel symmetry
S_c = 3.0e-2

# Part 1
xyzr_be = np.loadtxt('csvs/Bot_Eye_5k_300s.csv', delimiter=',')
xyzr_be = xyzr_be[xyzr_be[:, 3] >= S_c]
pos_be = xyzr_be[:, :3]
rad_be = xyzr_be[:, 3:4]
pos_be_center = jnp.mean(pos_be, axis=0, keepdims=True)
pos_be = transform_points(pos_be, pos_be_center.reshape(-1), translation=jnp.array([0.625, 0.625, 0.125]), rotation=jnp.array([0, 0, 0]))

# Calculate the pseudo-densities
densities_be, densities_pen_be = project_component(el_centers, jnp.array([el_size]), pos_be, rad_be, kernel_pos, kernel_rad)
densities_combined = combine_densities(densities_be, min_density=1e-2, penalty_factor=3)

# Heat-generating component
heat_load_nodes = find_active_nodes(densities_be, elements, threshold=1e-3)
heat_load_per_element = 0.5


# Define boundary conditions

# Fixed-temperature surface
dirichlet_nodes = find_face_nodes(nodes, jnp.array([0.0, 0.0, -1.0]))
dirichlet_bc_1 = DirichletBC(dirichlet_nodes, T=200)

# Convection surface
conv_surface_area = el_size**2
robin_nodes = find_face_nodes(nodes, jnp.array([0.0, 0.0, 1.0]))
robin_bc_1 = RobinBC(robin_nodes, h=10.0, T_inf=200, area=conv_surface_area)



# Run the FEA pipeline.
t1 = time_ns()
nodes, elements, T = solve_system(densities_combined.flatten(),
                                  heat_load_per_element,
                                  nodes,
                                  elements,
                                  base_k=1.0,
                                  boundary_conditions=[dirichlet_bc_1, robin_bc_1])
t2 = time_ns()

print("Time taken:", (t2 - t1) / 1e9, "s")


# from jax import jacfwd, jacrev
#
# t3 = time_ns()
# jf = jacfwd(solve_system, argnums=1)
# jf_val = jf(densities_combined.flatten(),
#                                   heat_load_per_element,
#                                   nodes,
#                                   elements,
#                                   base_k=1.0,
#                                   boundary_conditions=[dirichlet_bc_1, robin_bc_1])
# t4 = time_ns()
#
# print("Time taken for forward Jacobian:", (t4 - t3) / 1e9, "s")


T_np = np.array(T)
print("Computed nodal temperatures (sample):", T_np[:10])
nodes_plot = np.array(nodes)
T_plot = np.array(T)

el_centers = np.array(el_centers)

# # Plot
# plotter = pv.Plotter(shape=(2, 3), window_size=(1500, 500))
#
# # Plot the grid without the kernel
# plot_grid(plotter, (0, 0), el_centers, el_size, densities=None)
# plot_stl_file(plotter, (0, 0), 'models/Bot_Eye_scaled.stl', translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0))
# plot_AABB_spheres(plotter, (0, 0), pos_be, rad_be, color='blue')
#
#
# # Plot the grid with the kernel
# # plot_AABB_spheres(plotter, (1, 1), sample_positions_be, sample_radii_be, color='black', opacity=0.0)
# # plot_spheres(plotter, (1, 1), sample_positions_be, sample_radii_be, 'blue', opacity=0.5)
#
# plot_spheres(plotter, (0, 1), pos_be, rad_be, 'blue', opacity=0.5)
# plot_stl_file(plotter, (0, 1), 'models/Bot_Eye_scaled.stl', translation=(0.625, 0.625, 0.125), rotation=(0, 0, 0), opacity=0.5)
#
#
#
# # Plot the grid without the kernel
# plot_grid(plotter, (0, 2), el_centers, el_size, densities=densities_combined)
# plot_grid(plotter, (1, 1), el_centers, el_size, densities=densities_be)
# plot_grid(plotter, (1, 2), el_centers, el_size, densities=densities_combined)
#
#
# # plot_nodes(plotter, (0, 2), np.array(nodes), np.array(heat_load_nodes), label='Heat Load', color='red', point_size=20)
#
# plot_temperature_distribution(plotter,
#                               (0, 2),
#                               nodes_plot,
#                               T_plot,
#                               heat_load_nodes,
#                               robin_nodes,
#                               dirichlet_nodes,
#                               dims=(nx + 1, ny + 1, nz + 1),
#                               cmap='jet')

# plotter.show_axes()
# plotter.link_views()
# plotter.show()

