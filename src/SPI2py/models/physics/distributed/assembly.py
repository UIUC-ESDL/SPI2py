import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax import jit
from jax.experimental.sparse import BCOO
from dataclasses import dataclass, field
from chex import assert_shape, assert_type
from .element import assemble_local_stiffness_matrix
from .quadrature import gauss_quad
from scipy.sparse import coo_matrix, csr_matrix, diags


def assemble_local_stiffness_matrix_hex(el_size, base_k):
    """
    Assemble the local stiffness matrix for an 8-node linear hexahedral element
    using two-point Gauss quadrature in each direction.

    Parameters:
      el_size : the edge length of the hexahedral element (assumed uniform)
      base_k  : the base conductivity (or stiffness constant)

    Returns:
      Ke : 8x8 local stiffness matrix.

    Notes:
      - Natural coordinates: (xi,eta,zeta) in [-1,1]^3.
      - The 8 nodes have natural coordinates:
            (-1,-1,-1), (1,-1,-1), (1,1,-1), (-1,1,-1),
            (-1,-1, 1), (1,-1, 1), (1,1, 1), (-1,1, 1)
      - Two-point Gauss quadrature in each direction uses points ±1/√3 with weight 1.
      - For a uniform brick, the Jacobian is constant with |J| = (el_size/2)^3 = el_size^3/8,
        and the mapping gradients scale by 2/el_size.
      - Thus, each quadrature contribution carries a factor: (2/el_size)^2 * (el_size^3/8) = el_size/2.
    """
    # Define Gauss points and weights.
    gauss_pts = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    weights = np.array([1.0, 1.0])

    # Natural coordinates for 8-node hexahedron.
    nat_coords = np.array([
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])

    n_nodes = 8
    Ke = np.zeros((n_nodes, n_nodes))

    # The constant Jacobian determinant.
    detJ = el_size ** 3 / 8.0
    # The transformation factor for derivatives: physical derivative = (2/el_size) * natural derivative.
    transform = 2.0 / el_size

    # Loop over all 8 integration points.
    for xi in range(2):
        for eta in range(2):
            for zeta in range(2):
                gp = np.array([gauss_pts[xi], gauss_pts[eta], gauss_pts[zeta]])
                w = weights[xi] * weights[eta] * weights[zeta]

                # For each node, compute the derivatives of the shape functions with respect to xi, eta, zeta.
                dN_nat = np.zeros((n_nodes, 3))  # each row for a node
                for i in range(n_nodes):
                    xi_i, eta_i, zeta_i = nat_coords[i]
                    # Standard formulas for 8-node brick element.
                    dN_dxi = 1 / 8 * xi_i * (1 + gp[1] * eta_i) * (1 + gp[2] * zeta_i)
                    dN_deta = 1 / 8 * (1 + gp[0] * xi_i) * eta_i * (1 + gp[2] * zeta_i)
                    dN_dzeta = 1 / 8 * (1 + gp[0] * xi_i) * (1 + gp[1] * eta_i) * zeta_i
                    dN_nat[i, :] = np.array([dN_dxi, dN_deta, dN_dzeta])

                # Transform to physical derivatives.
                dN_phys = transform * dN_nat  # each derivative multiplied by 2/h

                # Compute contribution at this integration point.
                # Local stiffness contribution: Ke_ij += weight * detJ * (gradN_i dot gradN_j)
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        Ke[i, j] += w * detJ * np.dot(dN_phys[i, :], dN_phys[j, :])

    # Multiply by the base conductivity.
    Ke *= base_k
    return Ke


def assemble_global_stiffness_matrix(nodes, elements, densities, base_k, el_size):
    """
    Assemble the global stiffness matrix for a mesh of uniform hexahedral elements.

    Parameters:
      nodes      : (n_nodes x dim) array of node coordinates.
      elements   : (n_elements x 8) array of connectivity (indices into nodes).
      densities  : (n_elements,) array with ersatz densities for each element.
      base_k     : base conductivity.
      el_size    : edge length of each hexahedral element.

    Returns:
      K_global : global stiffness matrix in CSR format.
      f_global : global load vector (initialized to zeros).
    """
    n_nodes = nodes.shape[0]
    data = []
    row = []
    col = []
    # Initialize load vector.
    f_global = np.zeros(n_nodes)

    # Precompute the constant local stiffness matrix (for a unit element)
    Ke_const = assemble_local_stiffness_matrix_hex(el_size, base_k)
    # Element volume.
    el_vol = el_size ** 3

    # Loop over each element.
    for e in range(elements.shape[0]):
        elem_nodes = elements[e, :]  # indices (length 8)
        rho = densities[e]
        # Scale local stiffness by ersatz density.
        Ke_e = rho * Ke_const
        # Assemble into global matrix.
        for i_local, i_global in enumerate(elem_nodes):
            for j_local, j_global in enumerate(elem_nodes):
                data.append(Ke_e[i_local, j_local])
                row.append(i_global)
                col.append(j_global)
        # Optionally, assemble the element load vector.
        # Here, assume each element has a heat load Q (e.g. from inputs) that is distributed equally.
        # (For example, if heat_loads[e] is the load per element, then each node gets Q/8.)
        # We postpone applying heat loads until later (see below).

    K_global = coo_matrix((data, (row, col)), shape=(n_nodes, n_nodes))
    return K_global.tocsr(), f_global


def assemble_global_load_vector(nodes, elements, heat_loads, densities, el_size):
    """
    Assemble the global load vector by distributing the element heat load
    to the nodes. We assume that for each element, the load is scaled by the
    ersatz density and distributed equally among its 8 nodes.

    Parameters:
      nodes      : (n_nodes x dim) array.
      elements   : (n_elements x 8) connectivity.
      heat_loads : (n_elements,) array with heat load per element.
      densities  : (n_elements,) array with ersatz densities.
      el_size    : element edge length.

    Returns:
      f_global : global load vector (dense numpy array).
    """
    n_nodes = nodes.shape[0]
    f_global = np.zeros(n_nodes)
    nodes_per_elem = 8
    el_vol = el_size ** 3
    for e in range(elements.shape[0]):
        elem_nodes = elements[e, :]
        Qe = heat_loads[e] * densities[e] * el_vol  # total load in the element
        node_load = Qe / nodes_per_elem
        f_global[elem_nodes] += node_load
    return f_global


def apply_boundary_conditions(K, f, robin_nodes, r_h, r_T_inf, r_area,
                              dirichlet_nodes, dirichlet_values, beta):
    """
    Apply boundary conditions using a penalty method for Dirichlet BCs and
    adding a Robin (convective) contribution.

    For each Robin node, add to the diagonal:
      K[ii, ii] += r_h * r_area,
    and add to the load vector:
      f[ii] += r_h * r_area * r_T_inf.

    For each Dirichlet node, add a penalty term:
      K[ii, ii] += beta,
    and f[ii] += beta * (prescribed value).
    """
    n = K.shape[0]

    # Robin contribution.
    robin_diag = np.zeros(n)
    robin_diag[robin_nodes] = r_h * r_area
    R = diags(robin_diag, offsets=0, shape=(n, n), format='csr')
    f_robin = np.zeros(n)
    f_robin[robin_nodes] = r_h * r_area * r_T_inf

    # Dirichlet penalty.
    dirichlet_diag = np.zeros(n)
    dirichlet_diag[dirichlet_nodes] = beta
    P = diags(dirichlet_diag, offsets=0, shape=(n, n), format='csr')
    f_penalty = np.zeros(n)
    f_penalty[dirichlet_nodes] = beta * dirichlet_values

    K_new = K + R + P
    f_new = f + f_robin + f_penalty
    return K_new, f_new






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
#
#
# def apply_boundary_conditions(K, f, r_nodes, r_h, r_T_inf, r_area,
#                                      d_nodes, d_T, beta=1e10):
#     """
#     Apply Robin and Dirichlet BCs via sparse additions to the global system.
#
#     Robin (convective) BC:
#       For nodes in r_nodes, add a contribution to the diagonal and right-hand side:
#         K[ii, ii] += (r_h * r_area)
#         f[ii]      += (r_h * r_area * r_T_inf)
#
#     Dirichlet BC (penalty method):
#       For nodes in d_nodes, add a large penalty to force the solution toward d_T:
#         K[ii, ii] += beta
#         f[ii]      += beta * d_T
#
#     Parameters:
#       K : sparse global stiffness matrix (BCOO format)
#       f : global load vector (dense jnp.array)
#       r_nodes : 1D array of node indices for Robin BC
#       r_h, r_T_inf, r_area : scalars for the Robin condition
#       d_nodes : 1D array of node indices for Dirichlet BC
#       d_T : Dirichlet values at d_nodes (same shape as d_nodes)
#       beta : penalty coefficient (large positive scalar)
#
#     Returns:
#       K_new, f_new : the updated sparse system (BCOO and dense vector)
#     """
#     n = K.shape[0]
#
#     # --- Robin BC contribution ---
#     # For every Robin node, add a diagonal value = r_h * r_area.
#     robin_diag_val = r_h * r_area
#     # Build a sparse diagonal matrix R with nonzeros only at r_nodes.
#     # Here, we assume r_nodes is a jnp.array of indices.
#     data_R = jnp.full(r_nodes.shape, robin_diag_val)
#     # Indices as 2 x N_R array: each column is [i, i] for i in r_nodes.
#     # indices_R = jnp.stack([r_nodes, r_nodes], axis=0)
#     indices_R = jnp.stack([r_nodes, r_nodes], axis=-1)  # or .T if you prefer
#
#     R = BCOO((data_R, indices_R), shape=K.shape)
#
#     # --- Dirichlet BC penalty contribution ---
#     data_P = jnp.full(d_nodes.shape, beta)
#     # indices_P = jnp.stack([d_nodes, d_nodes], axis=0)
#     indices_P = jnp.stack([d_nodes, d_nodes], axis=-1)
#
#     P = BCOO((data_P, indices_P), shape=K.shape)
#
#     # Update f with Robin and penalty contributions:
#     # Robin: f[i] += r_h * r_area * r_T_inf for each i in r_nodes.
#     f_robin = jnp.zeros_like(f)
#     f_robin = f_robin.at[r_nodes].set(r_h * r_area * r_T_inf)
#     # Dirichlet penalty: f[i] += beta * d_T[i] for i in d_nodes.
#     f_penalty = jnp.zeros_like(f)
#     f_penalty = f_penalty.at[d_nodes].set(beta * d_T)
#
#     # Combine contributions with K and f
#     K_new = K + R + P
#     f_new = f + f_robin + f_penalty
#
#     return K_new, f_new




# def apply_boundary_conditions(K, f, r_nodes, r_h, r_T_inf, r_area,
#                               d_nodes, d_T):
#     """
#     A central function to apply boundary conditions to the global stiffness matrix and load vector.
#
#     This includes modifying and partitioning the system. This also provides a means to control and
#     investigate the superimposition of boundary conditions. For example, if we are optimizing the
#     layout of two pipes with fixed but different temperatures, we can see how selecting one Dirichlet
#     condition over the other, averaging those conditions, reformulating them as high heat loads rather than
#     fixed temperature, etc., impact the optimization process.
#     """
#
#     # Add the Robin (convective) contribution to the diagonal entries.
#     # Add the corresponding contribution to the load vector.
#     K_add = (r_h * r_area)
#     # K = K.at[r_nodes, r_nodes].add(K_add)
#     f_add = (r_h * r_area * r_T_inf)
#     # f = f.at[r_nodes].add(f_add)
#
#     K, f = append_global_system(K, f, r_nodes, K_add, f_add)
#
#     # TODO Replace with identify
#     idx_p, u_p = d_nodes, d_T
#
#     # Obtain the number of nodes and all node indices.
#     n_nodes = K.shape[0]
#     idx = jnp.arange(n_nodes)
#
#     # Find the free indices by subtracting the fixed indices from all indices.
#     idx_f = jnp.setdiff1d(idx, idx_p)
#
#     # Partition the stiffness matrix and load vector.
#     K_ff, K_fp, K_pf, K_pp, f_f, f_p = partition_global_system(K, f, idx_f, idx_p)
#
#     return K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, idx_f, idx_p


# def append_global_system(K, f, append_indices, K_add, f_add):
#     """
#     Modify the global stiffness matrix K and load vector f by updating values at specified nodes.
#
#     Parameters:
#       K: Global stiffness matrix (n_nodes x n_nodes).
#       f: Global load vector (n_nodes,).
#       append_indices: 1D array of node indices to be modified.
#       K_add: Stiffness matrix to add at these nodes.
#       f_add: Load vector to add at these nodes.
#
#     Returns:
#       K_new: Modified stiffness matrix.
#       f_new: Modified load vector.
#     """
#     K_new = K.at[append_indices, append_indices].add(K_add)
#     f_new = f.at[append_indices].add(f_add)
#     return K_new, f_new


# def partition_global_system(K, f, idx_f, idx_p):
#     """Optimized partitioning using JAX advanced indexing."""
#
#     idx = jnp.concatenate([idx_f, idx_p])  # Concatenate once to avoid multiple re-indexing
#     K_sub = K[idx][:, idx]  # One slicing operation
#
#     n_f = len(idx_f)  # Number of free DOFs
#     K_ff, K_fp = K_sub[:n_f, :n_f], K_sub[:n_f, n_f:]
#     K_pf, K_pp = K_sub[n_f:, :n_f], K_sub[n_f:, n_f:]
#
#     f_f, f_p = f[idx_f], f[idx_p]
#
#     return K_ff, K_fp, K_pf, K_pp, f_f, f_p