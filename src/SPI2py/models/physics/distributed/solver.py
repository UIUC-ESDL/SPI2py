import jax
import jax.numpy as jnp
from .assembly import assemble_global_stiffness_matrix, apply_boundary_conditions

from jax.scipy.sparse.linalg import cg
from jax.experimental.sparse import BCOO
from scipy.sparse import coo_matrix


def solve_system(density,
                 heat_load_per_element,
                 nodes,
                 elements,
                 base_k,
                 boundary_conditions):
    """
    Given a global stiffness matrix K and load vector f, modify them for Robin BCs,
    partition the system for Dirichlet BCs, and solve the reduced system.
    Then reassemble the full solution vector.

    h,
 T_inf,
 fixed_nodes,
 fixed_values,
 robin_nodes,
 conv_area,
 comp_nodes,
 comp_temp
    """

    # Generate the mesh and element connectivity.
    # ...

    # Assemble the global stiffness matrix and load vector.
    K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k)

    # For each element, add (heat_load/number_of_nodes) to each of its 8 nodes.
    nodes_per_elem = 8
    element_contrib = (heat_load_per_element * density) / nodes_per_elem
    elem_contrib_flat = element_contrib.flatten()
    node_contrib = jnp.repeat(elem_contrib_flat, nodes_per_elem)
    f = f.at[elements.flatten()].add(node_contrib)

    # Apply the boundary conditions and partition the system.
    K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, idx_f, idx_p = apply_boundary_conditions(K, f, boundary_conditions)

    # Solve the partitioned system for the unknown displacements.
    # TODO Does sparse solver work with autograd VJP?
    # K_ff @ u_f + K_fp @ u_p = f_f
    # K_ff @ u_f = f_f - K_fp @ u_p
    # u_f = K_ff^-1 @ (f_f - K_fp @ u_p)
    # u_f = jnp.linalg.solve(K_ff, f_f - K_fp @ u_p)

    # Convert K_ff to a sparse format for efficient solving
    # K_ff = BCOO.from_scipy_sparse(coo_matrix(K_ff))
    # K_ff = BCOO.fromdense(K_ff)

    # Solve the partitioned system for the unknown displacements using Conjugate Gradient (CG)
    def fea_solve(rhs):
        u_f, _ = cg(K_ff, rhs, tol=1e-8, maxiter=500)
        return u_f

    u_f = fea_solve(f_f - K_fp @ u_p)  # Solving K_ff @ u_f = (f_f - K_fp @ u_p)

    # Reassemble the full solution.
    n_nodes = K.shape[0]
    u = jnp.zeros(n_nodes)
    u = u.at[idx_f].set(u_f)
    u = u.at[idx_p].set(u_p)

    return nodes, elements, u
