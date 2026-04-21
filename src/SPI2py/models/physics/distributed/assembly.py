import jax.numpy as jnp
from jax import vmap, jit
from jax.experimental.sparse import BCOO

from .element import assemble_local_stiffness_matrix_scalar
from .quadrature import gauss_quad


# @jit
def construct_global_stiffness_matrix(node_positions,
                                      element_to_node_mapping,
                                      k,
                                      pseudo_densities):
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

    # Scale the base thermal conductivity by the pseudo-densities.
    k_eff = k * pseudo_densities

    # Gather nodal coordinates for each element.
    # Shape: (n_elem, 8, 3)
    el_nodes_all = node_positions[element_to_node_mapping]

    # Get quadrature points and weights.
    gauss_pts, gauss_wts = gauss_quad()

    # Use vmap to compute the 8x8 local stiffness matrix for each element.
    # Shape: (n_elem, 8, 8)
    Ke_all = vmap(lambda el_nodes, k_eff: assemble_local_stiffness_matrix_scalar(el_nodes, k_eff, gauss_pts, gauss_wts))(
        el_nodes_all, k_eff)

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
    indices = K.indices
    rows_flat = indices[:, 0]
    cols_flat = indices[:, 1]

    n_nodes = K.shape[0]
    n_f = idx_f.shape[0]
    n_p = idx_p.shape[0]

    free_map = -jnp.ones(n_nodes, dtype=indices.dtype)
    free_map = free_map.at[idx_f].set(jnp.arange(n_f, dtype=indices.dtype))

    prescribed_map = -jnp.ones(n_nodes, dtype=indices.dtype)
    prescribed_map = prescribed_map.at[idx_p].set(jnp.arange(n_p, dtype=indices.dtype))

    rows_f = free_map[rows_flat]
    cols_f = free_map[cols_flat]
    rows_p = prescribed_map[rows_flat]
    cols_p = prescribed_map[cols_flat]

    def make_block(local_rows, local_cols, shape):
        mask = (local_rows >= 0) & (local_cols >= 0)
        data = jnp.where(mask, K.data, 0)
        local_indices = jnp.stack([
            jnp.where(mask, local_rows, 0),
            jnp.where(mask, local_cols, 0),
        ], axis=-1)
        return BCOO((data, local_indices), shape=shape).sum_duplicates()

    K_ff = make_block(rows_f, cols_f, (n_f, n_f))
    K_fp = make_block(rows_f, cols_p, (n_f, n_p))
    K_pf = make_block(rows_p, cols_f, (n_p, n_f))
    K_pp = make_block(rows_p, cols_p, (n_p, n_p))

    return K_ff, K_fp, K_pf, K_pp


def partition_vector(v, idx_f, idx_p):

    v_f = v[idx_f]
    v_p = v[idx_p]

    return v_f, v_p


# @jit
def assemble_global_system_partition(nodes, elements,
                                     k,
                                     pseudo_densities,
                                     r_nodes, r_h, r_T_inf, r_area,
                                     heat_nodes, heat_loads,
                                     d_nodes, d_T):

    # Identify the free and prescribed nodes.
    idx = jnp.arange(nodes.shape[0])
    idx_p = d_nodes
    idx_f = jnp.setdiff1d(idx, idx_p)

    # Construct the global stiffness matrix and load vector
    K = construct_global_stiffness_matrix(nodes, elements, k, pseudo_densities)
    f = jnp.zeros_like(idx, dtype=jnp.float64)
    u = jnp.zeros_like(idx, dtype=jnp.float64)

    # Apply Robin BC
    K_robin_indices = jnp.stack([r_nodes, r_nodes], axis=-1)
    K_robin_data = (r_h * r_area) * jnp.ones_like(r_nodes)
    K_robin = BCOO((K_robin_data, K_robin_indices), shape=K.shape)
    K = K + K_robin

    f_robin_data = (r_h * r_area * r_T_inf) * jnp.ones_like(r_nodes)
    f = f.at[r_nodes].add(f_robin_data)

    # Apply heat loads
    if heat_nodes is not None:
        f = f.at[heat_nodes].add(heat_loads)

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
