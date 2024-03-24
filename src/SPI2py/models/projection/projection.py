def mdbd_cube(center, length, n_recursions=0):
    positions = []
    radii = []

    # Define the initial cube
    positions.append(center)
    radii.append(length / 2)


# From Mohammad's code
# rho_min = 1e-3;
# rho_d = torch.zeros((constants['num_models'], nelx * nely * nelz)).to(device)
# rho = torch.zeros((nelx * nely * nelz)).to(device)
#
# for i in range(0, constants['num_models']):
#     sph_inds = torch.arange(2 * i * constants['num_sph'],
#                             (2 * i + 1) * constants['num_sph'])  ## index of spheres for each component
#
#     radii = torch.tile(mdbds[sph_inds, -1].reshape((-1, 1)), (1, MESH['centers'].shape[0])).reshape(
#         (-1, 1))  # Radius of circle
#     x0v = torch.tile(cur_pos[sph_inds, :3], (1, MESH['centers'].shape[0])).reshape(
#         (-1, 3))  # The current location of each component
#     mesh_cent = torch.tile(MESH['centers'], (constants['num_sph'], 1))  # Center of each mesh
#     dd = torch.sqrt(torch.sum((mesh_cent - x0v) ** 2, axis=1)).reshape(
#         (-1, 1))  # Distance between the center of spheres and meshes
#     phi_q = dd - radii  # Distance between each mesh and sphere
#     phi_d = (1 - rho_min) / (1 + torch.exp(20 * phi_q)) + rho_min  # Map the distance to [0,1] value for meshes
#     rho_x = torch.reshape(phi_d, (constants['num_sph'], nelx * nely * nelz))
#     rho_d[i, :] = torch.sum(rho_x, axis=0)
#     rho += rho_d[i, :]
# rho = (1 - rho_min) / (1 + torch.exp(-1 * (rho - 0.5))) + rho_min
# rho_d = (1 - rho_min) / (1 + torch.exp(-1 * (rho_d - 0.5))) + rho_min
# rho[rho > 1] = 1
# rho_d[rho_d > 1] = 1
