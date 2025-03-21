import numpy as np
from jax import vmap
import jax.numpy as jnp
from jax import jit
from jax.experimental.sparse import BCOO
from dataclasses import dataclass, field
from chex import assert_shape, assert_type
from .element import assemble_local_stiffness_matrix
from .quadrature import gauss_quad, shape_functions
from scipy.sparse import coo_matrix, csr_matrix, diags


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

@jit
def assemble_base_global_stiffness(nodes, elements, base_k):
    """
    Assemble the global base stiffness matrix (with density=1) and an auxiliary mapping,
    using vmap to call the local stiffness function.

    Parameters:
      nodes:    (n_nodes, 3) array of coordinates.
      elements: (n_elem, 8) connectivity array (indices into nodes).
      base_k:   Base thermal conductivity.

    Returns:
      K_base:      Sparse global stiffness matrix in BCOO format (for density = 1).
      elem_indices: (nnz,) integer array mapping each nonzero entry in K_base to its element index.
    """
    n_elem = elements.shape[0]
    n_nodes = nodes.shape[0]

    # For the base case, use density=1 for every element.
    k_eff_all = base_k * jnp.ones((n_elem,))

    # Gather nodal coordinates for each element.
    element_nodes_all = nodes[elements]  # shape: (n_elem, 8, 3)

    # Get quadrature points and weights.
    gauss_pts, gauss_wts = gauss_quad()  # Assumed to return 1D arrays

    # Use vmap to compute the 8x8 local stiffness matrix for each element.
    Ke_all = vmap(lambda el_nodes, k_eff: assemble_local_stiffness_matrix(el_nodes, k_eff, gauss_pts, gauss_wts))(
        element_nodes_all, k_eff_all
    )  # shape: (n_elem, 8, 8)

    # Build global index arrays for scatter-add:
    # For each element, generate its 8x8 block of indices.
    # rows: shape (n_elem, 64)
    rows = jnp.repeat(elements, repeats=8, axis=1)
    # cols: shape (n_elem, 64)
    cols = jnp.tile(elements, reps=(1, 8))
    rows_flat = rows.reshape(-1)
    cols_flat = cols.reshape(-1)
    Ke_flat = Ke_all.reshape(-1)  # shape: (n_elem*64,)

    # Build auxiliary mapping: for each element, its index repeats 64 times.
    elem_indices = jnp.repeat(jnp.arange(n_elem), 64)  # shape: (n_elem*64,)

    # Stack rows and cols to form an index array of shape (n_elem*64, 2).
    indices = jnp.stack([rows_flat, cols_flat], axis=-1)

    # Create the sparse matrix in BCOO format.
    K_base = BCOO((Ke_flat, indices), shape=(n_nodes, n_nodes))
    return K_base, elem_indices


@jit
def update_global_stiffness(K_base, elem_indices, density, penal=1.0):
    """
    Update the base global stiffness matrix using the current densities.

    Parameters:
      K_base:      Sparse base stiffness matrix (for density = 1) in BCOO format.
      elem_indices: (nnz,) array mapping each nonzero entry in K_base to its element index.
      density:     (n_elem,) array of current element densities (or broadcastable to that shape).
      penal:       Penalization exponent.

    Returns:
      K_updated:   Updated global stiffness matrix in BCOO format.
    """
    # Ensure density is a 1D vector.
    density = density.flatten()  # now shape is (n_elem,)

    # Compute scaling factors for each element (e.g., density**penal).
    scaling_factors = density ** penal  # shape: (n_elem,)

    # Use the element mapping to broadcast the scaling to each nonzero.
    scaling = scaling_factors[elem_indices]  # Expected shape: (nnz,)

    # Multiply the data in the base matrix by the scaling factors.
    new_data = K_base.data * scaling
    K_updated = BCOO((new_data, K_base.indices), shape=K_base.shape)
    return K_updated

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


def apply_boundary_conditions(K, f, r_nodes, r_h, r_T_inf, r_area,
                                     d_nodes, d_T, beta=1e10):
    """
    Apply Robin and Dirichlet BCs via sparse additions to the global system.

    Robin (convective) BC:
      For nodes in r_nodes, add a contribution to the diagonal and right-hand side:
        K[ii, ii] += (r_h * r_area)
        f[ii]      += (r_h * r_area * r_T_inf)

    Dirichlet BC (penalty method):
      For nodes in d_nodes, add a large penalty to force the solution toward d_T:
        K[ii, ii] += beta
        f[ii]      += beta * d_T

    Parameters:
      K : sparse global stiffness matrix (BCOO format)
      f : global load vector (dense jnp.array)
      r_nodes : 1D array of node indices for Robin BC
      r_h, r_T_inf, r_area : scalars for the Robin condition
      d_nodes : 1D array of node indices for Dirichlet BC
      d_T : Dirichlet values at d_nodes (same shape as d_nodes)
      beta : penalty coefficient (large positive scalar)

    Returns:
      K_new, f_new : the updated sparse system (BCOO and dense vector)
    """
    n = K.shape[0]

    # --- Robin BC contribution ---
    # For every Robin node, add a diagonal value = r_h * r_area.
    robin_diag_val = r_h * r_area
    # Build a sparse diagonal matrix R with nonzeros only at r_nodes.
    # Here, we assume r_nodes is a jnp.array of indices.
    data_R = jnp.full(r_nodes.shape, robin_diag_val)
    # Indices as 2 x N_R array: each column is [i, i] for i in r_nodes.
    # indices_R = jnp.stack([r_nodes, r_nodes], axis=0)
    indices_R = jnp.stack([r_nodes, r_nodes], axis=-1)  # or .T if you prefer

    R = BCOO((data_R, indices_R), shape=K.shape)

    # --- Dirichlet BC penalty contribution ---
    data_P = jnp.full(d_nodes.shape, beta)
    # indices_P = jnp.stack([d_nodes, d_nodes], axis=0)
    indices_P = jnp.stack([d_nodes, d_nodes], axis=-1)

    P = BCOO((data_P, indices_P), shape=K.shape)

    # Update f with Robin and penalty contributions:
    # Robin: f[i] += r_h * r_area * r_T_inf for each i in r_nodes.
    f_robin = jnp.zeros_like(f)
    f_robin = f_robin.at[r_nodes].set(r_h * r_area * r_T_inf)
    # Dirichlet penalty: f[i] += beta * d_T[i] for i in d_nodes.
    f_penalty = jnp.zeros_like(f)
    f_penalty = f_penalty.at[d_nodes].set(beta * d_T)

    # Combine contributions with K and f
    K_new = K + R + P
    f_new = f + f_robin + f_penalty

    return K_new, f_new




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