# Standard imports
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

# Local imports
from .assembly import update_global_stiffness_partition


def solve_system_partition(densities, heat_loads,
                           nodes, elements,
                           K_base, f_base, u_p,
                           elem_indices, idx_f, idx_p):
    """
    Compute the primal (temperature) solution and a measure of the maximum
    temperature using a sparse solver. This function assembles the global system,
    applies heat loads, and then enforces boundary conditions using a penalty
    formulation rather than explicit partitioning.

    Parameters:
      densities    : material density field (array, to be flattened)
      heat_loads   : distributed heat loads (array, to be flattened)
      nodes        : node coordinates (array)
      elements     : element connectivity (array)
      r_nodes      : indices for nodes with Robin BC
      r_h, r_T_inf, r_area : Robin BC parameters
      d_nodes      : indices for nodes with Dirichlet BC (prescribed temperature)
      d_T          : prescribed temperatures at d_nodes

    Returns:
      u    : computed temperature (or displacement) field (dense vector)
      u_max: a scalar computed via a Kreisselmeier–Steinhauser (KS) function (here using a min-version)
    """

    # Solve via the partition method.

    K_updated, f_updated = update_global_stiffness_partition(K_base, f_base,
                                                             element_to_node_mapping=elements,
                                                             idx_vector=(idx_f, idx_p),
                                                             ei_matrix=elem_indices,
                                                             element_densities=densities, element_heat_loads=heat_loads)
    K_ff, K_fp, K_pf, K_pp = K_updated
    f_f, f_p = f_updated

    # Solve the global system using a sparse solver.
    rhs = (f_f - K_fp @ u_p)
    u_f, _ = cg(K_ff, rhs)

    # Reassemble the full solution.
    n_nodes = nodes.shape[0]
    u = jnp.zeros(n_nodes)
    u = u.at[idx_f].set(u_f)
    u = u.at[idx_p].set(u_p)

    return u


def solve_system_penalty():
    # Archived code
    # # Solve via the penalty method.
    #
    # # Update the global stiffness matrix using current densities.
    # K_updated, f_updated = update_global_system_penalty(K_base, f_base,
    #                                                     elements, elem_indices,
    #                                                     densities, heat_loads)
    #
    # # Solve the global system using a sparse solver.
    # u, _ = cg(K_updated, f_updated, tol=1e-8, maxiter=500)
    pass
