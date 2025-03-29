from jax import jit
import jax.numpy as jnp
from .quadrature import shape_functions


@jit
def assemble_local_stiffness_matrix(nodes, k_eff, gauss_pts, gauss_wts):
    """
    Compute the 8x8 element stiffness matrix for a single element.

    Einstein summation convention is used for clarity.
    q = gaussian quadrature point index, 0-1 (two-point quadrature).
    e = hexahedral element node index, 0-7 (8 nodes).
    g = global coordinate index, 0-2 (x, y, z).
    l = local coordinate index, 0-2 (xi, eta, zeta).

    Parameters:
      nodes: (8, 3)  array with the coordinates of the element's nodes.
      k_eff:         Scalar effective conductivity for the element.
      gauss_pts:     1D array of Gauss quadrature points.
      gauss_wts:     1D array of Gauss quadrature weights.

    Returns:
      Ke: (8,8) element stiffness matrix.
    """

    # Build a 3D quadrature grid.
    # Shape: (n_qp,)
    xi_grid, eta_grid, zeta_grid = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing='ij')
    xi = xi_grid.flatten()
    eta = eta_grid.flatten()
    zeta = zeta_grid.flatten()

    # Build the total quadrature weights.
    # Shape: (n_qp,)
    wx, wy, wz = jnp.meshgrid(gauss_wts, gauss_wts, gauss_wts, indexing='ij')
    w_total = (wx * wy * wz).flatten()

    # Evaluate shape functions and their derivatives at all quadrature points.
    # Shapes (n_qp, 8) and (n_qp, 8, 3)
    Nqp, dNqp_dLocal = shape_functions(xi, eta, zeta)

    # Compute the Jacobian at each quadrature point.
    # J[q, g, l] = sum_{e=0}^{7} nodes[e, g] * dNqp_dLocal[q, e, l]
    # Shape: (n_qp, 3, 3)
    J = jnp.einsum('eg,qel->qgl', nodes, dNqp_dLocal)

    # Compute determinant and inverse of the Jacobian.
    detJ = jnp.abs(jnp.linalg.det(J))
    J_inv = jnp.linalg.inv(J)

    # Map the shape function derivatives to physical coordinates:
    # dNqp_dGlobal[q] = J_inv[q] @ dNqp_dLocal[q] for each quadrature point.
    # Shape (n_qp, 8, 3)
    dNqp_dGlobal = jnp.einsum('qgl,qel->qeg', J_inv, dNqp_dLocal)

    # For each quadrature point, compute the contribution to the local stiffness:
    # Contribution = k_eff * (dNqp_dGlobal @ dNqp_dGlobal^T) * detJ * w_total.
    # Shape: (n_qp, 8, 8)
    contrib = jnp.einsum('qel,qgl->qeg', dNqp_dGlobal, dNqp_dGlobal)
    contrib = k_eff * contrib * (detJ * w_total)[:, None, None]

    # Sum over all quadrature points to obtain the local stiffness matrix.
    # Shape: (8, 8)
    Ke = jnp.sum(contrib, axis=0)

    return Ke
