# model.connect('system.components.comp_1.transformed_sphere_positions', 'bbv.sphere_positions')
# model.connect('system.components.comp_1.transformed_sphere_radii', 'bbv.sphere_radii')




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