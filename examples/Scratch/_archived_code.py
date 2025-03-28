# model.connect('system.components.comp_1.transformed_sphere_positions', 'bbv.sphere_positions')
# model.connect('system.components.comp_1.transformed_sphere_radii', 'bbv.sphere_radii')

# @staticmethod
    # def _compute_primal(density, heat_loads, nodes, elements,
    #                     r_nodes, r_h, r_T_inf, r_area,
    #                     d_nodes, d_T):
    #
    #     # Flatten the inputs
    #     density = density.flatten()
    #     heat_loads = heat_loads.flatten()
    #
    #     # Set the base thermal conductivity
    #     # TODO What?
    #     base_k = 1.0
    #
    #
    #     # Assemble the global stiffness matrix and load vector.
    #     K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k)
    #
    #     # Apply the heat loads to the system.
    #     nodes_per_elem = 8
    #     element_contrib = (heat_loads * density) / nodes_per_elem
    #     node_contrib = jnp.repeat(element_contrib, nodes_per_elem)
    #     f = f.at[elements.flatten()].add(node_contrib)
    #
    #     # Apply the boundary conditions and partition the system.
    #     K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, idx_f, idx_p = apply_boundary_conditions(K, f,
    #                                                                                     r_nodes, r_h, r_T_inf, r_area,
    #                                                                                     d_nodes, d_T)
    #
    #     # Solve the partitioned system for the unknown displacements.
    #     # K_ff @ u_f + K_fp @ u_p = f_f
    #     # K_ff @ u_f = f_f - K_fp @ u_p
    #     # u_f = K_ff^-1 @ (f_f - K_fp @ u_p)
    #     # u_f = jnp.linalg.solve(K_ff, f_f - K_fp @ u_p)
    #
    #     # Convert K_ff to a sparse format for efficient solving
    #     # K_ff = BCOO.from_scipy_sparse(coo_matrix(K_ff))
    #     # K_ff = coo_fromdense(K_ff)
    #     K_ff = BCOO.fromdense(K_ff)
    #
    #     # Solve the partitioned system for the unknown displacements using Conjugate Gradient (CG)
    #     def fea_solve(rhs):
    #         u_f, _ = cg(K_ff, rhs, tol=1e-8, maxiter=500)
    #         return u_f
    #
    #     u_f = fea_solve(f_f - K_fp @ u_p)  # Solving K_ff @ u_f = (f_f - K_fp @ u_p)
    #
    #     # Reassemble the full solution.
    #     n_nodes = K.shape[0]
    #     u = jnp.zeros(n_nodes)
    #     u = u.at[idx_f].set(u_f)
    #     u = u.at[idx_p].set(u_p)
    #
    #     # Calculate the max temp
    #     # u_max = kreisselmeier_steinhauser_max(u)
    #     # TODO Reset
    #     u_max = kreisselmeier_steinhauser_min(u)
    #
    #     return u, u_max


#                 interconnects.add_subsystem(f'int_{i}', interconnect)
#
#                 # Connect the interconnects to the components
#                 self.connect(f'components.comp_{component_1}.transformed_ports',
#                              f'interconnects.int_{i}.start_point',
#                              src_indices=om.slicer[port_1, :])
#                 self.connect(f'components.comp_{component_2}.transformed_ports',
#                              f'interconnects.int_{i}.end_point',
#                              src_indices=om.slicer[port_2, :])
#
#             self.add_subsystem('interconnects', interconnects)
#
#         # Create the system
#         system = System(n_projections=len(components_dict) + len(interconnects_dict), rho_min=1e-3)
#         self.add_subsystem('system', system)
#
#
#         # Connect the components to the system
#         i = 0
#         for j in range(len(components_dict)):
#             self.connect(f'components.comp_{j}.pseudo_densities',
#                           f'system.pseudo_densities_{i}')
#             i += 1
#
#         # Connect the interconnects to the system
#         for j in range(len(interconnects_dict)):
#             self.connect(f'interconnects.int_{j}.pseudo_densities',
#                           f'system.pseudo_densities_{i}')
#             i += 1

# model.add_subsystem('system', SpatialConfiguration(input_dict=input_file))
# model.add_subsystem('mesh', Mesh(bounds=bounds,
#                                  n_elements_per_unit_length=n_elements_per_unit_length))

# model.add_subsystem('projections', Projections(n_comp_projections=n_components,
#                                                n_int_projections=m_interconnects))
#
# model.add_subsystem('aggregator', ProjectionAggregator(n_projections=n_projections))
#
#
# model.add_subsystem('mux_all_sphere_positions', Multiplexer(n_i=n_points_per_object, m=3))
# model.add_subsystem('mux_all_sphere_radii', Multiplexer(n_i=n_points_per_object, m=1))
# model.add_subsystem('bbv', BoundingBoxVolume())
#
# # Connect the system to the projections
# i = 0
# for j in range(n_components):
#     model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'projections.projection_{i}.sphere_positions')
#     model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'projections.projection_{i}.sphere_radii')
#     model.connect(f'system.components.comp_{j}.volume', f'projections.projection_{i}.volume')
#     model.connect(f'projections.projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
#     i += 1
#
# for j in range(m_interconnects):
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', f'projections.projection_{i}.sphere_positions')
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_radii', f'projections.projection_{i}.sphere_radii')
#     # model.connect(f'system.interconnects.int_{j}.volume', f'projections.projection_{i}.volume')
#     model.connect(f'projections.projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
#     i += 1
#
# # Connect the mesh to the projections
# model.connect('mesh.element_length', 'aggregator.element_length')
# for i in range(n_projections):
#     model.connect('mesh.element_length', f'projections.projection_{i}.element_length')
#     model.connect('mesh.centers', f'projections.projection_{i}.centers')
#     model.connect('mesh.element_bounds', f'projections.projection_{i}.element_bounds')
#     model.connect('mesh.sample_points', f'projections.projection_{i}.element_sphere_positions')
#     model.connect('mesh.sample_radii', f'projections.projection_{i}.element_sphere_radii')
#
# # Connect the system to the bounding box
# i = 0
# for j in range(n_components):
#     model.connect(f'system.components.comp_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
#     model.connect(f'system.components.comp_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
#     i += 1
#
# for j in range(m_interconnects):
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', f'mux_all_sphere_positions.input_{i}')
#     model.connect(f'system.interconnects.int_{j}.transformed_sphere_radii', f'mux_all_sphere_radii.input_{i}')
#     i += 1
#
# model.connect('mux_all_sphere_positions.stacked_output', 'bbv.sphere_positions')
# model.connect('mux_all_sphere_radii.stacked_output', 'bbv.sphere_radii')
#




# Archived code
        # self.add_input('volume', val=0.0)
        # self.add_output('volume_estimation_error', val=0.0, desc='How accurately the projection represents the object')
        # volume_kernel = jnp.sum(4/3 * jnp.pi * kernel_radii ** 3)
        # volume_approximation_error = abs((volume_kernel - volume_element) / volume_element)


# def plot_problem(prob):
#     """
#     Plot the model at a given state.
#     """
#
#     # Create the plotter
#     plotter = pv.Plotter(shape=(1, 2), window_size=(1000, 500))
#
#     # Plot 1: Objects
#     plotter.subplot(0, 0)
#
#     # Plot the components
#     components = []
#     component_colors = []
#     for subsystem in prob.model.spatial_config.components._subsystems_myproc:
#         positions = prob.get_val(f'spatial_config.components.{subsystem.name}.transformed_sphere_positions')
#         radii = prob.get_val(f'spatial_config.components.{subsystem.name}.transformed_sphere_radii')
#         color = subsystem.options['color']
#
#         spheres = []
#         for position, radius in zip(positions, radii):
#             spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
#
#         merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
#
#         components.append(merged)
#         component_colors.append(color)
#
#     for comp, color in zip(components, component_colors):
#         plotter.add_mesh(comp, color=color, opacity=0.5)
#
#     # Plot the interconnects
#     if 'interconnects' in prob.model.spatial_config._subsystems_allprocs:
#
#         interconnects = []
#         interconnect_colors = []
#         for subsystem in prob.model.spatial_config.interconnects._subsystems_myproc:
#
#             positions = prob.get_val(f'spatial_config.interconnects.{subsystem.name}.transformed_sphere_positions')
#             radii = prob.get_val(f'spatial_config.interconnects.{subsystem.name}.transformed_sphere_radii')
#             color = subsystem.options['color']
#
#             # Plot the spheres
#             spheres = []
#             for position, radius in zip(positions, radii):
#                 spheres.append(pv.Sphere(radius=radius, center=position, theta_resolution=30, phi_resolution=30))
#
#             # Plot the cylinders
#             cylinders = []
#             for i in range(len(positions) - 1):
#                 start = positions[i]
#                 stop = positions[i + 1]
#                 radius = radii[i]
#                 length = np.linalg.norm(stop - start)
#                 direction = (stop - start) / length
#                 center = (start + stop) / 2
#                 cylinder = pv.Cylinder(center=center, direction=direction, radius=radius, height=length)
#                 cylinders.append(cylinder)
#
#             # merged = pv.MultiBlock(spheres).combine().extract_surface().clean()
#             merged_spheres = pv.MultiBlock(spheres).combine().extract_surface().clean()
#             merged_cylinders = pv.MultiBlock(cylinders).combine().extract_surface().clean()
#             merged = merged_spheres + merged_cylinders
#
#             interconnects.append(merged)
#             interconnect_colors.append(color)
#
#         for inter, color in zip(interconnects, interconnect_colors):
#             plotter.add_mesh(inter, color=color, lighting=False)
#
#     # Plot 2: The combined density with colored spheres
#     plotter.subplot(0, 1)
#
#     # Plot grid
#     bounds = prob.model.mesh.options['bounds']
#     nx = int(prob.get_val('mesh.n_el_x'))
#     ny = int(prob.get_val('mesh.n_el_y'))
#     nz = int(prob.get_val('mesh.n_el_z'))
#     spacing = float(prob.get_val('mesh.element_length'))
#     plot_grid(plotter, nx, ny, nz, bounds, spacing)
#
#
#     # Plot projections
#     pseudo_densities = prob.get_val(f'spatial_config.system.pseudo_densities')
#     centers = prob.get_val(f'mesh.centers')
#
#     # Plot the projected pseudo-densities of each element (speed up by skipping near-zero densities)
#     density_threshold = 1e-3
#     above_threshold_indices = np.argwhere(pseudo_densities > density_threshold)
#     for idx in above_threshold_indices:
#         n_i, n_j, n_k = idx
#
#         # Calculate the center of the current box
#         center = centers[n_i, n_j, n_k]
#         density = pseudo_densities[n_i, n_j, n_k]
#
#         if density > 1:
#             # Create the box
#             box = pv.Cube(center=center, x_length=2*spacing, y_length=2*spacing, z_length=2*spacing)
#             plotter.add_mesh(box, color='red', opacity=0.5)
#         else:
#             # Create the box
#             box = pv.Cube(center=center, x_length=spacing, y_length=spacing, z_length=spacing)
#             plotter.add_mesh(box, color='black', opacity=density)
#
#
#     # Configure the plot
#     plotter.link_views()
#     plotter.view_xy()
#     # plotter.view_isometric()
#     plotter.show_axes()
#     # plotter.show_bounds(color='black')
#     # p.background_color = 'white'
#
#     plotter.show()



# # Sweep component and plot derivatives
# import matplotlib.pyplot as plt
#
# # prob.set_val('system.components.comp_2.translation', [-1.5, 0.75, 0.5])
# x_values = np.linspace(-1.5, 1.5, 100)
# # f_vals = []
# # df_dx = []
# c_vals = []
# dc_dx = []
# for xi in x_values:
#     prob.set_val('system.components.comp_2.translation', [xi, 0, 0])
#     # prob.set_val('system.components.comp_2.translation', [0, xi, 1])
#     # prob.set_val('system.components.comp_2.translation', [0, 0, xi])
#     prob.run_model()
#     # fval = copy(prob.get_val('bbv.volume'))
#     cval = copy(prob.get_val('projections.aggregator.max_density'))
#     # f_vals.append(fval)
#     c_vals.append(cval)
#     # totalsf = prob.compute_totals(of=['bbv.volume'], wrt=['system.components.comp_2.translation'])
#     totalsc = prob.compute_totals(of=['projections.aggregator.max_density'], wrt=['system.components.comp_2.translation'])
#
#     # Extract the scalar derivative
#     # derivf = copy(totalsf[('bbv.volume', 'system.components.comp_2.translation')][0][0])
#     derivc = copy(totalsc[('projections.aggregator.max_density', 'system.components.comp_2.translation')][0][0])
#     # df_dx.append(derivf)
#     dc_dx.append(derivc)
#
# # Apply finite difference
# prob.model.approx_totals(method='fd')  # Use finite differencing
# df_dx_approx = []
# dc_dx_approx = []
# for xi in x_values:
#     prob.set_val('system.components.comp_2.translation', [xi, 0, 0])
#     # prob.set_val('system.components.comp_2.translation', [0, xi, 1])
#     # prob.set_val('system.components.comp_2.translation', [0, 0, xi])
#     prob.run_model()
#
#     # totalsf = prob.compute_totals(of=['bbv.volume'], wrt=['system.components.comp_2.translation'])
#     totalsc = prob.compute_totals(of=['projections.aggregator.max_density'], wrt=['system.components.comp_2.translation'])
#
#     # Extract the scalar derivative
#     # derivf = copy(totalsf[('bbv.volume', 'system.components.comp_2.translation')][0][0])
#     derivc = copy(totalsc[('projections.aggregator.max_density', 'system.components.comp_2.translation')][0][0])
#     # df_dx_approx.append(derivf)
#     dc_dx_approx.append(derivc)
#
#
# # # Plot results
# # plt.figure(figsize=(8, 6))
# # plt.plot(x_values, f_vals, label='BBV', color='blue', linestyle='-')
# # plt.plot(x_values, df_dx, label='Computed Derivative', color='red', linestyle='--')
# # plt.plot(x_values, df_dx_approx, label='FD Derivative', color='green', linestyle='-.')
# # plt.xlabel('Translation (x)')
# # plt.ylabel('Value')
# # plt.title('BBV Objective & Its Derivatives')
# # plt.legend()
# # plt.grid(True)
# # plt.show()
#
#
# # Plot results
# plt.figure(figsize=(8, 6))
# plt.plot(x_values, c_vals, label='Max Density Constraint', color='blue', linestyle='-')
# plt.plot(x_values, dc_dx, label='Computed Derivative', color='red', linestyle='--')
# plt.plot(x_values, dc_dx_approx, label='FD Derivative', color='green', linestyle='-.')
# plt.xlabel('Translation (x)')
# plt.ylabel('Value')
# plt.title('Max Density Constraint & Its Derivatives')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # prob.set_val('system.components.comp_2.translation', [1, 1, 1])
# # prob.run_model()
#
# prob.set_val('system.components.comp_2.translation', [2, 0.75, 0.5])
# prob.run_model()


# @jit
# def assemble_global_stiffness_matrix_dense(nodes, elements, density, base_k):
#     """
#     Assemble the global stiffness matrix K in a sparse format.
#
#     Parameters:
#       nodes:    (n_nodes, 3) array of coordinates.
#       elements: (n_elem, 8) connectivity array (indices into nodes).
#       density:  (n_elem,) array of material densities.
#       base_k:   Base conductivity.
#
#     Returns:
#       K_sparse: Sparse global stiffness matrix (BCOO format).
#       f_global: Load vector (sparse).
#     """
#     n_nodes = nodes.shape[0]
#     n_elem = elements.shape[0]
#
#     # Compute effective conductivity for each element
#     k_eff_all = base_k * density
#
#     # Gather nodal coordinates for all elements
#     element_nodes_all = nodes[elements]
#
#     # Get Gauss quadrature points and weights
#     gauss_pts, gauss_wts = gauss_quad()
#
#     # Compute the local stiffness matrix for each element
#     Ke_all = vmap(lambda el_nodes, k_eff: assemble_local_stiffness_matrix(el_nodes, k_eff, gauss_pts, gauss_wts))(element_nodes_all, k_eff_all)
#
#     # Create index arrays for assembling global stiffness matrix
#     rows = jnp.repeat(elements, repeats=8, axis=1).reshape(-1)
#     cols = jnp.tile(elements, reps=(1, 8)).reshape(-1)
#
#     rows_flat = rows.reshape(-1)
#     cols_flat = cols.reshape(-1)
#     Ke_flat = Ke_all.reshape(-1)
#
#     # Initialize and assemble the global stiffness matrix.
#     K_global = jnp.zeros((n_nodes, n_nodes))
#     K_global = K_global.at[rows_flat, cols_flat].add(Ke_flat)
#
#     # Initialize sparse global load vector
#     f_global = jnp.zeros(n_nodes)
#
#     return K_global, f_global

# @jit
# def assemble_global_stiffness_matrix(nodes, elements, density, base_k):
#     """
#     Assemble the global stiffness matrix K in a sparse format.
#
#     Parameters:
#       nodes:    (n_nodes, 3) array of coordinates.
#       elements: (n_elem, 8) connectivity array (indices into nodes).
#       density:  (n_elem,) array of material densities.
#       base_k:   Base conductivity.
#
#     Returns:
#       K_sparse: Sparse global stiffness matrix (BCOO format).
#       f_global: Load vector (sparse).
#     """
#     n_nodes = nodes.shape[0]
#     n_elem = elements.shape[0]
#
#     # Compute effective conductivity for each element
#     k_eff_all = base_k * density
#
#     # Gather nodal coordinates for all elements
#     element_nodes_all = nodes[elements]
#
#     # Get Gauss quadrature points and weights
#     gauss_pts, gauss_wts = gauss_quad()
#
#     # Compute the local stiffness matrix for each element
#     Ke_all = vmap(lambda el_nodes, k_eff: assemble_local_stiffness_matrix(el_nodes, k_eff, gauss_pts, gauss_wts))(element_nodes_all, k_eff_all)
#
#     # Create global index arrays.
#     # For each element (row in elements, shape (8,)), we want to generate all 64 (i,j) pairs.
#     # rows: (n_elem, 8, 8) where each row is a repetition of the element connectivity along axis=1.
#     # cols: (n_elem, 8, 8) where each column is the connectivity repeated along axis=2.
#     rows = jnp.broadcast_to(elements[:, :, None], (n_elem, 8, 8))
#     cols = jnp.broadcast_to(elements[:, None, :], (n_elem, 8, 8))
#
#     # Flatten local stiffness contributions and the index arrays.
#     Ke_flat = Ke_all.reshape(-1)  # (n_elem*64,)
#     rows_flat = rows.reshape(-1)
#     cols_flat = cols.reshape(-1)
#
#     # BCOO expects the indices to have shape (nnz, ndims); so stack rows and cols along last axis.
#     indices = jnp.stack([rows_flat, cols_flat], axis=-1)  # shape: (n_elem*64, 2)
#
#     # Create the sparse global stiffness matrix.
#     K_sparse = BCOO((Ke_flat, indices), shape=(n_nodes, n_nodes))
#
#     # Initialize load vector as zeros.
#     f_global = jnp.zeros(n_nodes)
#
#     return K_sparse, f_global

# def assemble_global_stiffness_matrix(nodes, elements, density, base_k):
#     """
#     Assemble the global stiffness matrix K and load vector f in a fully vectorized way without using vmap.
#
#     Parameters:
#       nodes:      (n_nodes, 3) array of coordinates.
#       elements:   (n_elem, 8) connectivity array (indices into nodes).
#       density:    (n_elem,) array of material densities.
#       base_k:     Base thermal conductivity.
#       gauss_pts:  1D array of Gauss quadrature points.
#       gauss_wts:  1D array of Gauss quadrature weights.
#
#     Returns:
#       K_global:   (n_nodes, n_nodes) global stiffness matrix.
#       f_global:   (n_nodes,) load vector (zero in this example).
#     """
#     n_nodes = nodes.shape[0]
#     n_elem = elements.shape[0]
#
#     # 1. Compute effective conductivity for each element.
#     #    (Assume a simple linear interpolation: k_eff = base_k * density)
#     k_eff_all = base_k * density  # shape: (n_elem,)
#
#     # 2. Gather nodal coordinates for all elements.
#     #    element_nodes_all will have shape (n_elem, 8, 3)
#     element_nodes_all = nodes[elements]
#
#     # Get Gauss quadrature points and weights
#     gauss_pts, gauss_wts = gauss_quad()
#
#     # 3. Build the quadrature grid for the element integration.
#     #    Create a tensor grid from the 1D gauss_pts and gauss_wts.
#     xi_grid, eta_grid, zeta_grid = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing='ij')
#     xi = xi_grid.flatten()  # shape: (n_qp,)
#     eta = eta_grid.flatten()  # shape: (n_qp,)
#     zeta = zeta_grid.flatten()  # shape: (n_qp,)
#     n_qp = xi.shape[0]
#
#     # 4. Build the corresponding quadrature weights.
#     wx, wy, wz = jnp.meshgrid(gauss_wts, gauss_wts, gauss_wts, indexing='ij')
#     w_total = (wx * wy * wz).flatten()  # shape: (n_qp,)
#
#     # 5. Evaluate the shape functions and their natural derivatives at all quadrature points.
#     #    shape_functions_vec should accept arrays of quadrature points and return:
#     #      N_all: (n_qp, 8)
#     #      dN_dxi_all: (n_qp, 8, 3)
#     N_all, dN_dxi_all = shape_functions(xi, eta, zeta)
#
#     # 6. Compute the Jacobian for each element at each quadrature point.
#     #    For each element e and quadrature point q:
#     #       J_all[e, q, j, k] = sum_{i=0}^{7} element_nodes_all[e, i, j] * dN_dxi_all[q, i, k]
#     #    This can be done with einsum:
#     J_all = jnp.einsum('eij,qik->eqjk', element_nodes_all, dN_dxi_all)
#     # J_all has shape: (n_elem, n_qp, 3, 3)
#
#     # 7. Compute the determinant and inverse of each Jacobian.
#     detJ_all = jnp.abs(jnp.linalg.det(J_all))  # shape: (n_elem, n_qp)
#     J_inv_all = jnp.linalg.inv(J_all)  # shape: (n_elem, n_qp, 3, 3)
#
#     # 8. Map the shape function derivatives to physical coordinates.
#     #    For each element and quadrature point:
#     #       dN_dx = J_inv * dN_dxi.
#     #    First, broadcast dN_dxi_all from shape (n_qp, 8, 3) to (n_elem, n_qp, 8, 3):
#     dN_dxi_all_b = jnp.broadcast_to(dN_dxi_all, (n_elem, n_qp, 8, 3))
#     # Now compute dN_dx_all with einsum:
#     dN_dx_all = jnp.einsum('eqjk,eqik->eqij', J_inv_all, dN_dxi_all_b)
#     # dN_dx_all has shape: (n_elem, n_qp, 8, 3)
#
#     # 9. Compute the element stiffness contributions for each element and quadrature point.
#     #    For each element e and quadrature point q, the contribution is:
#     #      contrib[e, q] = k_eff_all[e] * (dN_dx_all[e,q] @ dN_dx_all[e,q]^T) * detJ_all[e,q] * w_total[q]
#     contrib_all = jnp.einsum('eqik,eqjk->eqij', dN_dx_all, dN_dx_all)  # shape: (n_elem, n_qp, 8, 8)
#     contrib_all = k_eff_all[:, None, None, None] * contrib_all \
#                   * detJ_all[:, :, None, None] * w_total[None, :, None, None]
#
#     # 10. Sum contributions over quadrature points for each element to obtain element stiffness matrices.
#     Ke_all = jnp.sum(contrib_all, axis=1)  # shape: (n_elem, 8, 8)
#
#     # 11. Assemble the global stiffness matrix via vectorized scatter-add.
#     #     For each element, we need to add its 8x8 block to the global K.
#     #     Build global index arrays for rows and columns:
#     #       rows: for each element, repeat its 8 node indices 8 times.
#     #       cols: for each element, tile its 8 node indices 8 times.
#     rows = jnp.repeat(elements, repeats=8, axis=1)  # shape: (n_elem, 64)
#     cols = jnp.tile(elements, reps=(1, 8))  # shape: (n_elem, 64)
#
#     # Flatten these index arrays.
#     rows_flat = rows.reshape(-1)  # shape: (n_elem*64,)
#     cols_flat = cols.reshape(-1)
#
#     # Flatten the element stiffness matrices.
#     Ke_flat = Ke_all.reshape(-1)
#
#     # Initialize the global stiffness matrix and scatter-add contributions.
#     K_global = jnp.zeros((n_nodes, n_nodes))
#     K_global = K_global.at[rows_flat, cols_flat].add(Ke_flat)
#
#     # For this simple example, the load vector is zero.
#     f_global = jnp.zeros(n_nodes)
#
#     return K_global, f_global

# def assemble_global_stiffness_matrix(nodes, elements, density, base_k):
#     """
#     Assemble the global stiffness matrix K and load vector f in a fully vectorized way,
#     initializing the global stiffness matrix as a sparse matrix (BCOO format).
#
#     Parameters:
#       nodes:      (n_nodes, 3) array of coordinates.
#       elements:   (n_elem, 8) connectivity array (indices into nodes).
#       density:    (n_elem,) array of material densities.
#       base_k:     Base thermal conductivity.
#
#     Returns:
#       K_global:   Sparse global stiffness matrix in BCOO format (shape: n_nodes x n_nodes).
#       f_global:   (n_nodes,) global load vector (zero in this simple example).
#     """
#     n_nodes = nodes.shape[0]
#     n_elem = elements.shape[0]
#
#     # 1. Compute effective conductivity for each element.
#     k_eff_all = base_k * density  # shape: (n_elem,)
#
#     # 2. Gather nodal coordinates for all elements.
#     element_nodes_all = nodes[elements]  # shape: (n_elem, 8, 3)
#
#     # Get Gauss quadrature points and weights.
#     gauss_pts, gauss_wts = gauss_quad()
#
#     # 3. Build the quadrature grid for element integration.
#     xi_grid, eta_grid, zeta_grid = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing='ij')
#     xi = xi_grid.flatten()  # shape: (n_qp,)
#     eta = eta_grid.flatten()
#     zeta = zeta_grid.flatten()
#     n_qp = xi.shape[0]
#
#     # 4. Build corresponding quadrature weights.
#     wx, wy, wz = jnp.meshgrid(gauss_wts, gauss_wts, gauss_wts, indexing='ij')
#     w_total = (wx * wy * wz).flatten()  # shape: (n_qp,)
#
#     # 5. Evaluate shape functions and their natural derivatives.
#     #    shape_functions returns:
#     #      N_all: (n_qp, 8)
#     #      dN_dxi_all: (n_qp, 8, 3)
#     N_all, dN_dxi_all = shape_functions(xi, eta, zeta)
#
#     # 6. Compute the Jacobian for each element at each quadrature point.
#     #    J_all[e, q, j, k] = sum_{i=0}^{7} element_nodes_all[e, i, j] * dN_dxi_all[q, i, k]
#     J_all = jnp.einsum('eij,qik->eqjk', element_nodes_all, dN_dxi_all)
#     # J_all has shape: (n_elem, n_qp, 3, 3)
#
#     # 7. Compute determinant and inverse of each Jacobian.
#     detJ_all = jnp.abs(jnp.linalg.det(J_all))  # shape: (n_elem, n_qp)
#     J_inv_all = jnp.linalg.inv(J_all)  # shape: (n_elem, n_qp, 3, 3)
#
#     # 8. Map shape function derivatives to physical coordinates.
#     dN_dxi_all_b = jnp.broadcast_to(dN_dxi_all, (n_elem, n_qp, 8, 3))
#     dN_dx_all = jnp.einsum('eqjk,eqik->eqij', J_inv_all, dN_dxi_all_b)
#     # dN_dx_all has shape: (n_elem, n_qp, 8, 3)
#
#     # 9. Compute element stiffness contributions for each quadrature point.
#     contrib_all = jnp.einsum('eqik,eqjk->eqij', dN_dx_all, dN_dx_all)
#     contrib_all = k_eff_all[:, None, None, None] * contrib_all \
#                   * detJ_all[:, :, None, None] * w_total[None, :, None, None]
#     # contrib_all has shape: (n_elem, n_qp, 8, 8)
#
#     # 10. Sum contributions over quadrature points to get the element stiffness matrix.
#     Ke_all = jnp.sum(contrib_all, axis=1)  # shape: (n_elem, 8, 8)
#
#     # 11. Assemble the global stiffness matrix via vectorized scatter-add.
#     #     For each element, build index arrays for its 8x8 block.
#     rows = jnp.repeat(elements, repeats=8, axis=1)  # shape: (n_elem, 64)
#     cols = jnp.tile(elements, reps=(1, 8))  # shape: (n_elem, 64)
#     rows_flat = rows.reshape(-1)  # shape: (n_elem*64,)
#     cols_flat = cols.reshape(-1)
#     Ke_flat = Ke_all.reshape(-1)  # shape: (n_elem*64,)
#
#     # 12. Build the sparse global stiffness matrix using BCOO.
#     indices = jnp.stack([rows_flat, cols_flat], axis=-1)  # shape: (n_elem*64, 2)
#     K_global = BCOO((Ke_flat, indices), shape=(n_nodes, n_nodes))
#
#     # 13. For this simple example, the load vector is zero.
#     f_global = jnp.zeros(n_nodes)
#
#     return K_global, f_global