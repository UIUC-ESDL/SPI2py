import jax.numpy as jnp
from jax import vmap, jit
from jax.experimental.sparse import BCOO

from .element import assemble_local_stiffness_matrix_scalar
from .quadrature import gauss_quad


# @jit
def construct_global_stiffness_matrix(node_positions,
                                      element_to_node_mapping,
                                      base_k):
    """
        Assemble the global base stiffness matrix.
        Indices are included for mapping element densities to each of their local stiffness matrix contributions.

        Parameters:
          node_positions:    (n_nodes, 3) array of coordinates.
          element_to_node_mapping: (n_elem, 8) connectivity array (indices into nodes).
          base_k:   Base thermal conductivity.

        Returns:
          K_base:      Sparse global stiffness matrix in BCOO format.
        """

    # Get the number of elements and nodes.
    n_elem = element_to_node_mapping.shape[0]
    n_nodes = node_positions.shape[0]

    # For the base case, use density=1 for every element.
    # TODO Use penalized density...
    k_eff_all = base_k * jnp.ones((n_elem,))

    # Gather nodal coordinates for each element.
    # Shape: (n_elem, 8, 3)
    el_nodes_all = node_positions[element_to_node_mapping]

    # Get quadrature points and weights.
    gauss_pts, gauss_wts = gauss_quad()

    # Use vmap to compute the 8x8 local stiffness matrix for each element.
    # Shape: (n_elem, 8, 8)
    Ke_all = vmap(lambda el_nodes, k_eff: assemble_local_stiffness_matrix_scalar(el_nodes, k_eff, gauss_pts, gauss_wts))(
        el_nodes_all, k_eff_all)

    # Flatten the local stiffness contributions.
    # Shape: (n_elem*64,)
    Ke_flat = Ke_all.reshape(-1)

    # Build global index arrays for scatter-add:
    # For each element, generate its 8x8 block of indices.
    # rows: shape (n_elem, 64)
    # cols: shape (n_elem, 64)
    rows = jnp.repeat(element_to_node_mapping, repeats=8, axis=1)
    cols = jnp.tile(element_to_node_mapping, reps=(1, 8))
    rows_flat = rows.reshape(-1)
    cols_flat = cols.reshape(-1)

    # Create the sparse matrix in BCOO format.
    indices = jnp.stack([rows_flat, cols_flat], axis=-1)
    K = BCOO((Ke_flat, indices), shape=(n_nodes, n_nodes))

    # Remove duplicate entries in the sparse matrix.
    # For example, if there is no interaction between nodes A and B, then we must ensure that K[A, B] = K[B, A] = 0.
    K = K.sum_duplicates()

    return K


def partition_sparse_matrix(K, idx_f, idx_p):

    Ke_flat = K.data
    indices = K.indices
    rows_flat = K.indices[:, 0]
    cols_flat = K.indices[:, 1]

    # Partition the contributions into four blocks based on free (idx_f) and prescribed (idx_p) DOF indices.
    mask_ff = jnp.logical_and(jnp.isin(rows_flat, idx_f), jnp.isin(cols_flat, idx_f))
    mask_fp = jnp.logical_and(jnp.isin(rows_flat, idx_f), jnp.isin(cols_flat, idx_p))
    mask_pf = jnp.logical_and(jnp.isin(rows_flat, idx_p), jnp.isin(cols_flat, idx_f))
    mask_pp = jnp.logical_and(jnp.isin(rows_flat, idx_p), jnp.isin(cols_flat, idx_p))

    # Helper function: remap a global index to local indices using the sorted index array.
    def remap_local(global_inds, idx_array):
        return jnp.searchsorted(idx_array, global_inds)

    # Free–Free block.
    data_ff = Ke_flat[mask_ff]
    indices_ff = indices[mask_ff]
    local_rows_ff = remap_local(indices_ff[:, 0], idx_f)
    local_cols_ff = remap_local(indices_ff[:, 1], idx_f)
    indices_ff_local = jnp.stack([local_rows_ff, local_cols_ff], axis=-1)
    K_ff = BCOO((data_ff, indices_ff_local), shape=(len(idx_f), len(idx_f)))

    # Free–Prescribed block.
    data_fp = Ke_flat[mask_fp]
    indices_fp = indices[mask_fp]
    local_rows_fp = remap_local(indices_fp[:, 0], idx_f)
    local_cols_fp = remap_local(indices_fp[:, 1], idx_p)
    indices_fp_local = jnp.stack([local_rows_fp, local_cols_fp], axis=-1)
    K_fp = BCOO((data_fp, indices_fp_local), shape=(len(idx_f), len(idx_p)))

    # Prescribed–Free block.
    data_pf = Ke_flat[mask_pf]
    indices_pf = indices[mask_pf]
    local_rows_pf = remap_local(indices_pf[:, 0], idx_p)
    local_cols_pf = remap_local(indices_pf[:, 1], idx_f)
    indices_pf_local = jnp.stack([local_rows_pf, local_cols_pf], axis=-1)
    K_pf = BCOO((data_pf, indices_pf_local), shape=(len(idx_p), len(idx_f)))

    # Prescribed–Prescribed block.
    data_pp = Ke_flat[mask_pp]
    indices_pp = indices[mask_pp]
    local_rows_pp = remap_local(indices_pp[:, 0], idx_p)
    local_cols_pp = remap_local(indices_pp[:, 1], idx_p)
    indices_pp_local = jnp.stack([local_rows_pp, local_cols_pp], axis=-1)
    K_pp = BCOO((data_pp, indices_pp_local), shape=(len(idx_p), len(idx_p)))

    return K_ff, K_fp, K_pf, K_pp


def partition_vector(v, idx_f, idx_p):

    v_f = v[idx_f]
    v_p = v[idx_p]

    return v_f, v_p


# @jit
def assemble_global_system_partition(nodes, elements,
                                     base_k,
                                     r_nodes, r_h, r_T_inf, r_area,
                                     d_nodes, d_T):

    # Identify the free and prescribed nodes.
    idx = jnp.arange(nodes.shape[0])
    idx_p = d_nodes
    idx_f = jnp.setdiff1d(idx, idx_p)

    # Construct the global stiffness matrix and load vector
    K = construct_global_stiffness_matrix(nodes, elements, base_k)
    f = jnp.zeros_like(idx, dtype=jnp.float64)
    u = jnp.zeros_like(idx, dtype=jnp.float64)

    # Apply Robin BC
    K_robin_indices = jnp.stack([r_nodes, r_nodes], axis=-1)
    K_robin_data = (r_h * r_area) * jnp.ones_like(r_nodes)
    K_robin = BCOO((K_robin_data, K_robin_indices), shape=K.shape)
    K = K + K_robin

    f_robin_data = (r_h * r_area * r_T_inf) * jnp.ones_like(r_nodes)
    f = f.at[r_nodes].add(f_robin_data)

    # Apply Dirichlet BC
    u = u.at[d_nodes].add(d_T)

    # Partition the global stiffness matrix and load vector
    K_ff, K_fp, K_pf, K_pp = partition_sparse_matrix(K, idx_f, idx_p)
    f_f, f_p               = partition_vector(f, idx_f, idx_p)
    u_f, u_p               = partition_vector(u, idx_f, idx_p)

    # Repack the partitioned stiffness matrices and load vector.
    K_base = (K_ff, K_fp, K_pf, K_pp)
    f_base = (f_f, f_p)
    u_base = (u_f, u_p)

    return K_base, f_base, u_base




# @jit
def assemble_base_global_system_penalty(nodes, elements, base_k,
                                        r_nodes, r_h, r_T_inf, r_area,
                                        d_nodes, d_T):

    Ke_flat, elem_indices, rows_flat, cols_flat, n_nodes, n_elem = construct_global_stiffness_matrix(nodes, elements, base_k)

    # Stack rows and cols to form an index array of shape (n_elem*64, 2).
    indices = jnp.stack([rows_flat, cols_flat], axis=-1)

    # Create the sparse matrix in BCOO format.
    K_base = BCOO((Ke_flat, indices), shape=(n_nodes, n_nodes))

    # Define forces
    f_base = jnp.zeros((n_nodes,))

    # Apply boundary conditions (both Robin and Dirichlet).
    K_base, f_base = apply_bc_penalty(K_base, f_base,
                                      r_nodes, r_h, r_T_inf, r_area,
                                      d_nodes, d_T,
                                      beta=1e10)

    return K_base, f_base, elem_indices


def apply_bc_penalty(K, f,
                     r_nodes, r_h, r_T_inf, r_area,
                     d_nodes, d_T,
                     beta=1e10):
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
    # n = K.shape[0]
    #
    # # K--Robin BC contribution
    # # Build a sparse diagonal matrix R with non-zeros only at r_nodes.
    # robin_value = r_h * r_area
    # data_R = jnp.full(r_nodes.shape, robin_value)
    # indices_R = jnp.stack([r_nodes, r_nodes], axis=-1)
    # R = BCOO((data_R, indices_R), shape=K.shape)
    #
    # # K--Dirichlet BC contribution
    # data_P = jnp.full(d_nodes.shape, beta)
    # indices_P = jnp.stack([d_nodes, d_nodes], axis=-1)
    # P = BCOO((data_P, indices_P), shape=K.shape)
    #
    # # f--Robin contribution
    # # f[i] += r_h * r_area * r_T_inf for each i in r_nodes.
    # f_robin = jnp.zeros_like(f)
    # f_robin = f_robin.at[r_nodes].set(r_h * r_area * r_T_inf)
    #
    # # f--Dirichlet penalty
    # # f[i] += beta * d_T[i] for i in d_nodes.
    # f_penalty = jnp.zeros_like(f)
    # f_penalty = f_penalty.at[d_nodes].set(beta * d_T)
    #
    # # Combine contributions with K and f
    # K_new = K + R + P
    # f_new = f + f_robin + f_penalty
    #
    # return K_new, f_new

    # Identify diagonal entries (where row index equals column index).
    diag_mask = (K.indices[:, 0] == K.indices[:, 1])
    diag_nodes = K.indices[:, 0]  # these are the node indices for the diagonal entries

    # Create masks for Robin and Dirichlet BCs on the diagonal.
    robin_mask = diag_mask & jnp.isin(diag_nodes, r_nodes)
    dirichlet_mask = diag_mask & jnp.isin(diag_nodes, d_nodes)

    # Compute the total update for each diagonal entry.
    # If a diagonal entry corresponds to a Robin node, add r_h*r_area.
    # If it corresponds to a Dirichlet node, add beta.
    # (If a node is in both sets, the contributions are summed.)
    update = robin_mask.astype(K.data.dtype) * (r_h * r_area) \
             + dirichlet_mask.astype(K.data.dtype) * (beta)

    # Update the K data array.
    new_data = K.data + update
    K_new = BCOO((new_data, K.indices), shape=K.shape)

    # Update the load vector f.
    f_new = f
    f_new = f_new.at[r_nodes].add(r_h * r_area * r_T_inf)
    f_new = f_new.at[d_nodes].add(beta * d_T)

    return K_new, f_new


# @jit
def update_global_system_penalty(K_base, f_base,
                                 elements, elem_indices,
                                 densities, heat_loads):
    """
    Update the base global stiffness matrix using the current densities.

    Parameters:
      K_base:      Sparse base stiffness matrix (for density = 1) in BCOO format.
      elem_indices: (nnz,) array mapping each nonzero entry in K_base to its element index.
      densities:     (n_elem,) array of current element densities (or broadcastable to that shape).

    Returns:
      K_updated:   Updated global stiffness matrix in BCOO format.
    """

    # Ensure density is a 1D vector.
    densities = densities.flatten()
    heat_loads = heat_loads.flatten()

    # Use the element mapping to broadcast the scaling to each nonzero.
    scaling = densities[elem_indices]

    # Multiply the data in the base matrix by the scaling factors.
    new_data = K_base.data * scaling
    K_updated = BCOO((new_data, K_base.indices), shape=K_base.shape)

    # Assemble the global load vector from heat loads.
    nodes_per_elem = 8
    element_contrib = (heat_loads * densities) / nodes_per_elem
    f_updated = f_base.at[elements.flatten()].add(jnp.repeat(element_contrib, nodes_per_elem))

    return K_updated, f_updated








# def apply_bc_partition_method_dense(K, f,
#                                     r_nodes, r_h, r_T_inf, r_area,
#                                     d_nodes, d_T):
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
#     f_add = (r_h * r_area * r_T_inf)
#     K, f = append_global_system(K, f, r_nodes, K_add, f_add)
#
#     # Label the Dirichlet nodes and their prescribed temperatures.
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
#
#
# def append_global_system_dense(K, f, append_indices, K_add, f_add):
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
#
#
# def partition_global_system_dense(K, f, idx_f, idx_p):
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