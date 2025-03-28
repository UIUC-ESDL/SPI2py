import jax.numpy as jnp
from jax import vmap, jit
from jax.experimental.sparse import BCOO
from chex import assert_shape, assert_type

from .element import assemble_local_stiffness_matrix
from .quadrature import gauss_quad



@jit
def assemble_base_global_stiffness_penalty(nodes, elements, base_k):
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
    n = K.shape[0]

    # K--Robin BC contribution
    # Build a sparse diagonal matrix R with non-zeros only at r_nodes.
    robin_value = r_h * r_area
    data_R = jnp.full(r_nodes.shape, robin_value)
    indices_R = jnp.stack([r_nodes, r_nodes], axis=-1)
    R = BCOO((data_R, indices_R), shape=K.shape)

    # K--Dirichlet BC contribution
    data_P = jnp.full(d_nodes.shape, beta)
    indices_P = jnp.stack([d_nodes, d_nodes], axis=-1)
    P = BCOO((data_P, indices_P), shape=K.shape)

    # f--Robin contribution
    # f[i] += r_h * r_area * r_T_inf for each i in r_nodes.
    f_robin = jnp.zeros_like(f)
    f_robin = f_robin.at[r_nodes].set(r_h * r_area * r_T_inf)

    # f--Dirichlet penalty
    # f[i] += beta * d_T[i] for i in d_nodes.
    f_penalty = jnp.zeros_like(f)
    f_penalty = f_penalty.at[d_nodes].set(beta * d_T)

    # Combine contributions with K and f
    K_new = K + R + P
    f_new = f + f_robin + f_penalty

    return K_new, f_new


@jit
def update_global_stiffness_penalty(K_base,
                                    elem_indices,
                                    densities):
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

    # Use the element mapping to broadcast the scaling to each nonzero.
    scaling = densities[elem_indices]

    # Multiply the data in the base matrix by the scaling factors.
    new_data = K_base.data * scaling
    K_updated = BCOO((new_data, K_base.indices), shape=K_base.shape)

    return K_updated


def assemble_base_global_stiffness_partition(nodes, elements, base_k,
                                             idx_f, idx_p):
    n_elem = elements.shape[0]
    n_nodes = nodes.shape[0]
    n_local = elements.shape[1]  # e.g., 8 for an 8-node element

    # For the base case, use density=1 for every element.
    k_eff_all = base_k * jnp.ones((n_elem,))

    # Assemble local stiffness matrices for each element.
    element_nodes = nodes[elements]  # shape: (n_elem, n_local, dim)
    gauss_pts, gauss_wts = gauss_quad()  # assumed to return 1D arrays
    Ke_all = vmap(lambda el_nodes, k: assemble_local_stiffness_matrix(el_nodes, k, gauss_pts, gauss_wts))(
        element_nodes, k_eff_all
    )  # shape: (n_elem, n_local, n_local)

    # Flatten the local stiffness contributions.
    Ke_flat = Ke_all.reshape(-1)  # shape: (n_elem * n_local*n_local,)

    # Build global index arrays for scatter-add.
    rows = jnp.repeat(elements, repeats=n_local, axis=1)  # shape: (n_elem, n_local*n_local)
    cols = jnp.tile(elements, reps=(1, n_local))  # shape: (n_elem, n_local*n_local)
    rows_flat = rows.reshape(-1)
    cols_flat = cols.reshape(-1)
    indices = jnp.stack([rows_flat, cols_flat], axis=-1)  # shape: (nnz, 2)

    # Build a full element mapping array: each element's contribution repeats n_local*n_local times.
    elem_indices_full = jnp.repeat(jnp.arange(n_elem), n_local * n_local)

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
    ei_ff = elem_indices_full[mask_ff]

    # Free–Prescribed block.
    data_fp = Ke_flat[mask_fp]
    indices_fp = indices[mask_fp]
    local_rows_fp = remap_local(indices_fp[:, 0], idx_f)
    local_cols_fp = remap_local(indices_fp[:, 1], idx_p)
    indices_fp_local = jnp.stack([local_rows_fp, local_cols_fp], axis=-1)
    K_fp = BCOO((data_fp, indices_fp_local), shape=(len(idx_f), len(idx_p)))
    ei_fp = elem_indices_full[mask_fp]

    # Prescribed–Free block.
    data_pf = Ke_flat[mask_pf]
    indices_pf = indices[mask_pf]
    local_rows_pf = remap_local(indices_pf[:, 0], idx_p)
    local_cols_pf = remap_local(indices_pf[:, 1], idx_f)
    indices_pf_local = jnp.stack([local_rows_pf, local_cols_pf], axis=-1)
    K_pf = BCOO((data_pf, indices_pf_local), shape=(len(idx_p), len(idx_f)))
    ei_pf = elem_indices_full[mask_pf]

    # Prescribed–Prescribed block.
    data_pp = Ke_flat[mask_pp]
    indices_pp = indices[mask_pp]
    local_rows_pp = remap_local(indices_pp[:, 0], idx_p)
    local_cols_pp = remap_local(indices_pp[:, 1], idx_p)
    indices_pp_local = jnp.stack([local_rows_pp, local_cols_pp], axis=-1)
    K_pp = BCOO((data_pp, indices_pp_local), shape=(len(idx_p), len(idx_p)))
    ei_pp = elem_indices_full[mask_pp]

    # Assemble load vector f (here zero) and partition.
    f = jnp.zeros((n_nodes,))
    f_f = f[idx_f]
    f_p = f[idx_p]
    # Prescribed values u_p; here zeros (modify as needed).
    u_p = jnp.zeros_like(f_p)

    return K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, (ei_ff, ei_fp, ei_pf, ei_pp)


def update_global_stiffness_partition(K_ff, K_fp, K_pf, K_pp,
                                      f_f, f_p,
                                      u_p,
                                      elem_indices,
                                      densities):

    ei_ff, ei_fp, ei_pf, ei_pp = elem_indices
    densities = densities.flatten()

    new_data_ff = K_ff.data * densities[ei_ff]
    new_data_fp = K_fp.data * densities[ei_fp]
    new_data_pf = K_pf.data * densities[ei_pf]
    new_data_pp = K_pp.data * densities[ei_pp]

    K_ff_new = BCOO((new_data_ff, K_ff.indices), shape=K_ff.shape)
    K_fp_new = BCOO((new_data_fp, K_fp.indices), shape=K_fp.shape)
    K_pf_new = BCOO((new_data_pf, K_pf.indices), shape=K_pf.shape)
    K_pp_new = BCOO((new_data_pp, K_pp.indices), shape=K_pp.shape)

    # f_f, f_p, and u_p remain unchanged.
    return K_ff_new, K_fp_new, K_pf_new, K_pp_new, f_f, f_p, u_p

def apply_bc_partition_method(K_ff, K_fp, K_pf, K_pp,
                              f_f, f_p,
                              r_nodes, r_h, r_area, r_T_inf,
                              idx_f, idx_p):
    # Partition Robin nodes into free and prescribed sets.
    # jnp.intersect1d returns sorted intersections.
    r_nodes_free = jnp.intersect1d(r_nodes, idx_f)
    r_nodes_presc = jnp.intersect1d(r_nodes, idx_p)

    # For free nodes: remap global indices to local indices within idx_f.
    local_free = jnp.searchsorted(idx_f, r_nodes_free)
    # Build a diagonal sparse update for K_ff.
    diag_free = r_h * r_area * jnp.ones_like(local_free)
    indices_free = jnp.stack([local_free, local_free], axis=-1)
    R_free = BCOO((diag_free, indices_free), shape=(len(idx_f), len(idx_f)))
    K_ff_updated = K_ff + R_free

    # Similarly for prescribed nodes: remap global indices to local indices in idx_p.
    local_presc = jnp.searchsorted(idx_p, r_nodes_presc)
    diag_presc = r_h * r_area * jnp.ones_like(local_presc)
    indices_presc = jnp.stack([local_presc, local_presc], axis=-1)
    R_presc = BCOO((diag_presc, indices_presc), shape=(len(idx_p), len(idx_p)))
    K_pp_updated = K_pp + R_presc

    # Update the load vectors: add f_val = r_h * r_area * r_T_inf.
    # For free nodes:
    f_val = r_h * r_area * r_T_inf
    f_f_updated = f_f.at[local_free].add(f_val)
    # For prescribed nodes:
    f_p_updated = f_p.at[local_presc].add(f_val)

    return K_ff_updated, K_fp, K_pf, K_pp_updated, f_f_updated, f_p_updated


# def apply_bc_partition_method(K, f,
#                               r_nodes, r_h, r_T_inf, r_area,
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
#
#
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