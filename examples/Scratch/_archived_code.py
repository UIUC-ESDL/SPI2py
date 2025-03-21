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