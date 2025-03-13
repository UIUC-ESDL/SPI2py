from jax import vmap
import jax.numpy as jnp
from dataclasses import dataclass, field
from chex import assert_shape, assert_type
from .element import assemble_local_stiffness_matrix
from .quadrature import gauss_quad

from jax.experimental.sparse import BCOO, bcoo_slice, bcoo_extract
from scipy.sparse import coo_matrix


def assemble_global_stiffness_matrix(nodes, elements, density, base_k):
    """
    Assemble the global stiffness matrix K in a sparse format.

    Parameters:
      nodes:    (n_nodes, 3) array of coordinates.
      elements: (n_elem, 8) connectivity array (indices into nodes).
      density:  (n_elem,) array of material densities.
      base_k:   Base conductivity.

    Returns:
      K_sparse: Sparse global stiffness matrix (BCOO format).
      f_global: Load vector (sparse).
    """
    n_nodes = nodes.shape[0]
    n_elem = elements.shape[0]

    # Compute effective conductivity for each element
    k_eff_all = base_k * density

    # Gather nodal coordinates for all elements
    element_nodes_all = nodes[elements]

    # Get Gauss quadrature points and weights
    gauss_pts, gauss_wts = gauss_quad(n_qp=2)

    # Compute the local stiffness matrix for each element
    Ke_all = vmap(lambda el_nodes, k_eff: assemble_local_stiffness_matrix(el_nodes, k_eff, gauss_pts, gauss_wts))(element_nodes_all, k_eff_all)

    # Create index arrays for assembling global stiffness matrix
    rows = jnp.repeat(elements, repeats=8, axis=1).reshape(-1)
    cols = jnp.tile(elements, reps=(1, 8)).reshape(-1)
    Ke_flat = Ke_all.reshape(-1)

    # Assemble sparse stiffness matrix
    K_sparse = BCOO.from_scipy_sparse(coo_matrix((Ke_flat, (rows, cols)), shape=(n_nodes, n_nodes)))

    # Convert back to dense for debugging
    # K_dense = K_sparse.todense()

    # Initialize sparse global load vector
    f_global = jnp.zeros(n_nodes)

    return K_sparse, f_global


def apply_boundary_conditions(K, f, boundary_conditions):
    """
    A central function to apply boundary conditions to the global stiffness matrix and load vector.

    This includes modifying and partitioning the system. This also provides a means to control and
    investigate the superimposition of boundary conditions. For example, if we are optimizing the
    layout of two pipes with fixed but different temperatures, we can see how selecting one Dirichlet
    condition over the other, averaging those conditions, reformulating them as high heat loads rather than
    fixed temperature, etc., impact the optimization process.
    """
    dirichlet_bcs = []
    robin_bcs = []
    for bc in boundary_conditions:
        if bc.bc_type == "dirichlet":
            dirichlet_bcs.append(bc)
        elif bc.bc_type == "robin":
            robin_bcs.append(bc)
        else:
            raise ValueError(f"Unknown boundary condition type: {bc.bc_type}")

    # Apply Robin boundary conditions.
    r_nodes = [bc.nodes for bc in robin_bcs][0]
    r_h = [bc.h for bc in robin_bcs][0]
    r_T_inf = [bc.T_inf for bc in robin_bcs][0]
    r_area = [bc.area for bc in robin_bcs][0]

    # # Convert to dense
    # K = K.todense()
    # # K = BCOO.fromdense(K)


    # Add the Robin (convective) contribution to the diagonal entries.
    # Construct indices and values for sparse update
    K_bc_values = jnp.full((r_nodes.shape[0],), r_h * r_area, dtype=K.dtype)
    K_bc_indices = jnp.stack([r_nodes, r_nodes], axis=1).astype(jnp.int32)  # Ensure indices are int32

    # Create sparse BCOO update matrix
    K_bc = BCOO((K_bc_values, K_bc_indices), shape=K.shape)

    # Apply the update using sparse addition
    K = K + K_bc

    # Add the corresponding contribution to the load vector.
    f_add = (r_h * r_area * r_T_inf)
    f = f.at[r_nodes].add(f_add)

    # # Convert to dense
    # K = K.todense()
    # # K = BCOO.fromdense(K)

    # Combine and apply the Dirichlet BCs.
    d_nodes = [bc.nodes for bc in dirichlet_bcs]
    d_T = [bc.T for bc in dirichlet_bcs]

    idx_p, u_p = combine_fixed_conditions(d_nodes, d_T)

    # Convert to dense
    # K = K.todense()
    # K = BCOO.fromdense(K)

    # Obtain the number of nodes and all node indices.
    n_nodes = K.shape[0]
    idx = jnp.arange(n_nodes)

    # Find the free indices by subtracting the fixed indices from all indices.
    idx_f = jnp.setdiff1d(idx, idx_p)

    # Convert to dense
    # K = K.todense()
    # K = BCOO.fromdense(K)

    # Partition the stiffness matrix and load vector.
    K_ff, K_fp, K_pf, K_pp, f_f, f_p = partition_global_system(K, f, idx_f, idx_p)

    # Convert to dense
    # K = K.todense()
    # K = BCOO.fromdense(K)
    # K_ff = BCOO.fromdense(K_ff)
    # K_fp = BCOO.fromdense(K_fp)
    # K_pf = BCOO.fromdense(K_pf)
    # K_pp = BCOO.fromdense(K_pp)
    # K_ff = K_ff.todense()
    # K_fp = K_fp.todense()
    # K_pf = K_pf.todense()
    # K_pp = K_pp.todense()

    return K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, idx_f, idx_p



# def apply_boundary_conditions(K, f, boundary_conditions):
#     """
#     A central function to apply boundary conditions to the global stiffness matrix and load vector.
#
#     This includes modifying and partitioning the system. This also provides a means to control and
#     investigate the superimposition of boundary conditions. For example, if we are optimizing the
#     layout of two pipes with fixed but different temperatures, we can see how selecting one Dirichlet
#     condition over the other, averaging those conditions, reformulating them as high heat loads rather than
#     fixed temperature, etc., impact the optimization process.
#     TODO Vectorize?
#     """
#     dirichlet_bcs = []
#     robin_bcs = []
#     for bc in boundary_conditions:
#         if bc.bc_type == "dirichlet":
#             dirichlet_bcs.append(bc)
#         elif bc.bc_type == "robin":
#             robin_bcs.append(bc)
#         else:
#             raise ValueError(f"Unknown boundary condition type: {bc.bc_type}")
#
#     # Apply Robin boundary conditions.
#     r_nodes = [bc.nodes for bc in robin_bcs][0]
#     r_h = [bc.h for bc in robin_bcs][0]
#     r_T_inf = [bc.T_inf for bc in robin_bcs][0]
#     r_area = [bc.area for bc in robin_bcs][0]
#
#     # Add the Robin (convective) contribution to the diagonal entries.
#     K_add = (r_h * r_area)
#
#     # Add the corresponding contribution to the load vector.
#     f_add = (r_h * r_area * r_T_inf)
#
#     K = K.at[r_nodes, r_nodes].add(K_add)
#     f = f.at[r_nodes].add(f_add)
#
#     # Combine and apply the Dirichlet BCs.
#     d_nodes = [bc.nodes for bc in dirichlet_bcs]
#     d_T = [bc.T for bc in dirichlet_bcs]
#
#     idx_p, u_p = combine_fixed_conditions(d_nodes, d_T)
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



def append_global_system(K, f, append_indices, K_add, f_add):
    """
    Modify the global stiffness matrix K and load vector f by updating values at specified nodes.

    Parameters:
      K: Global stiffness matrix (n_nodes x n_nodes).
      f: Global load vector (n_nodes,).
      append_indices: 1D array of node indices to be modified.
      K_add: Stiffness matrix to add at these nodes.
      f_add: Load vector to add at these nodes.

    Returns:
      K_new: Modified stiffness matrix.
      f_new: Modified load vector.
    """
    K_new = K.at[append_indices, append_indices].add(K_add)
    f_new = f.at[append_indices].add(f_add)
    return K_new, f_new


# def partition_global_system(K, f, idx_f, idx_p):
#     """
#     Partition the global stiffness matrix and load vector into free and prescribed components efficiently.
#     Uses sparse matrix slicing to maintain sparsity.
#     """
#
#     # Use sparse slicing for faster access, without converting to dense
#     K_ff = K[idx_f][:, idx_f]  # Stays sparse
#     K_fp = K[idx_f][:, idx_p]  # Stays sparse
#     K_pf = K[idx_p][:, idx_f]  # Stays sparse
#     K_pp = K[idx_p][:, idx_p]  # Stays sparse
#     f_f = f[idx_f]
#     f_p = f[idx_p]
#
#     print(f"K_ff min: {K_ff.min()}, max: {K_ff.max()}")
#     print(f"f_f min: {f_f.min()}, max: {f_f.max()}")
#
#     return K_ff, K_fp, K_pf, K_pp, f_f, f_p

def extract_submatrix(K, row_idx, col_idx):
    """
    Extracts a submatrix from a sparse BCOO matrix using explicit sparse filtering.

    Parameters:
        K (BCOO): Sparse stiffness matrix.
        row_idx (jnp.ndarray or list): Indices of rows to extract.
        col_idx (jnp.ndarray or list): Indices of columns to extract.

    Returns:
        K_sub (BCOO): Extracted submatrix.
    """
    # Ensure indices are JAX arrays
    row_idx = jnp.asarray(row_idx)
    col_idx = jnp.asarray(col_idx)

    # Extract sparse data and indices
    data, indices = K.data, K.indices  # Indices shape: (n_nonzero, 2)

    # Find rows that match row_idx
    row_mask = jnp.isin(indices[:, 0], row_idx)

    # Filter rows
    data = data[row_mask]
    indices = indices[row_mask]

    # Find columns that match col_idx
    col_mask = jnp.isin(indices[:, 1], col_idx)
    data = data[col_mask]
    indices = indices[col_mask]

    # Remap indices for the new submatrix
    row_map = {int(v): i for i, v in enumerate(row_idx.tolist())}
    col_map = {int(v): i for i, v in enumerate(col_idx.tolist())}

    new_indices = jnp.array([[row_map[int(r)], col_map[int(c)]] for r, c in indices], dtype=jnp.int32)

    # Construct new sparse submatrix
    K_sub = BCOO((data, new_indices), shape=(len(row_idx), len(col_idx)))

    return K_sub


def partition_global_system(K, f, idx_f, idx_p):
    """
    Partitions the global stiffness matrix and load vector into free and prescribed components.

    Parameters:
        K (BCOO): Global stiffness matrix (sparse).
        f (jnp.ndarray): Global load vector.
        idx_f (jnp.ndarray): Indices of free nodes.
        idx_p (jnp.ndarray): Indices of prescribed nodes.

    Returns:
        K_ff (BCOO): Submatrix for free DOFs.
        K_fp (BCOO): Submatrix for free-prescribed coupling.
        K_pf (BCOO): Submatrix for prescribed-free coupling.
        K_pp (BCOO): Submatrix for prescribed DOFs.
        f_f (jnp.ndarray): Load vector for free DOFs.
        f_p (jnp.ndarray): Load vector for prescribed DOFs.
    """
    # Extract submatrices using sparse slicing
    K_ff = extract_submatrix(K, idx_f, idx_f)
    K_fp = extract_submatrix(K, idx_f, idx_p)
    K_pf = extract_submatrix(K, idx_p, idx_f)
    K_pp = extract_submatrix(K, idx_p, idx_p)

    # Extract sub-load vectors
    f_f = f[idx_f]
    f_p = f[idx_p]

    return K_ff, K_fp, K_pf, K_pp, f_f, f_p


# def partition_global_system(K, f, idx_f, idx_p):
#     """
#     Set the values of the global stiffness matrix K and load vector f at specified nodes.
#
#     Parameters:
#       K: Global stiffness matrix (n_nodes x n_nodes).
#       f: Global load vector (n_nodes,).
#       idx_f: 1D array of free node indices.
#       idx_p: 1D array of prescribed node indices.
#
#     Returns:
#         K_ff: Reduced stiffness matrix for free DOFs.
#         K_fp: Stiffness matrix coupling free and prescribed DOFs.
#         K_pf: Stiffness matrix coupling prescribed and free DOFs.
#         K_pp: Reduced stiffness matrix for prescribed DOFs.
#         f_f: Modified load vector for free DOFs.
#         f_p: Modified load vector for prescribed DOFs.
#     """
#
#     # Partition the stiffness matrix and load vector.
#     # [K_ff K_fp] {u_f} = {f_f}
#     # [K_pf K_pp] {u_p} = {f_p}
#     K_ff = K[idx_f][:, idx_f]
#     K_fp = K[idx_f][:, idx_p]
#     K_pf = K[idx_p][:, idx_f]
#     K_pp = K[idx_p][:, idx_p]
#     f_f = f[idx_f]
#     f_p = f[idx_p]
#
#     return K_ff, K_fp, K_pf, K_pp, f_f, f_p


def combine_fixed_conditions(idx_p, D_p):
    """
    Combine multiple sets of fixed nodes and their prescribed values into single arrays.

    Parameters:
      idx_p: a list (or tuple) of 1D arrays of prescribed node indices.
      D_p: a list (or tuple) of 1D arrays (or scalars) of prescribed values,
                         corresponding to each set of fixed nodes.

    Returns:
      combined_fixed_nodes: a 1D array containing all fixed node indices.
      combined_fixed_values: a 1D array containing the prescribed value for each fixed node.
    """
    combined_nodes = []
    combined_values = []
    for nodes_i, values_i in zip(idx_p, D_p):
        # Ensure values_i is a 1D array broadcasted to the same length as nodes_i.
        values_i = jnp.broadcast_to(jnp.atleast_1d(values_i), (nodes_i.shape[0],))
        combined_nodes.append(nodes_i)
        combined_values.append(values_i)

    combined_fixed_nodes = jnp.concatenate(combined_nodes)
    combined_fixed_values = jnp.concatenate(combined_values)
    return combined_fixed_nodes, combined_fixed_values



@dataclass
class BoundaryCondition:
    """
    Base class for a boundary condition.
    """

    nodes: jnp.ndarray
    bc_type: str = field(init=False)

    def __post_init__(self):

        # Nodes should be a 1D array.
        assert_shape(self.nodes, (None,))


@dataclass
class DirichletBC(BoundaryCondition):
    """
    Dirichlet boundary condition.
    """

    T: (int, float)

    def __post_init__(self):

        # Nodes should be a 1D array.
        self.bc_type = "dirichlet"

        # Value should be a scalar.
        assert isinstance(self.T, (int, float))


@dataclass
class RobinBC(BoundaryCondition):
    """
    Robin boundary condition.
    """
    h: (int, float)
    T_inf: (int, float)
    area: (int, float)

    def __post_init__(self):
        self.bc_type = "robin"
        assert isinstance(self.h, (int, float))
        assert isinstance(self.T_inf, (int, float))
        assert isinstance(self.area, (int, float))