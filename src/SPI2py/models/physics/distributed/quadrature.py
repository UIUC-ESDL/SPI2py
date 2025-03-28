import jax.numpy as jnp
from jax import jit


@jit
def shape_functions(xi, eta, zeta):
    """
    A vectorized evaluation of the trilinear shape functions and their derivatives.

    Parameters:
      xi, eta, zeta: arrays of shape (n_qp,)

    Returns:
      Nqp: (n_qp, 8) array of shape function values.
      dNqp_dLocal: (n_qp, 8, 3) array of shape function derivatives with respect to local coordinates (xi,eta,zeta).
    """

    # Reference coordinates for the eight nodes, shape (8, 3)
    node_ref = jnp.array([
        [-1, -1, -1],
        [ 1, -1, -1],
        [ 1,  1, -1],
        [-1,  1, -1],
        [-1, -1,  1],
        [ 1, -1,  1],
        [ 1,  1,  1],
        [-1,  1,  1]
    ])

    # Reshape xi, eta, zeta to shape (n_qp, 1) for broadcasting.
    xi = xi[:, None]
    eta = eta[:, None]
    zeta = zeta[:, None]

    # Evaluate shape functions.
    Nqp = 1/8.0 * (1 + xi * node_ref[:, 0]) * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2])

    # Compute derivatives with respect to xi, eta, and zeta.
    dNqp_dxi = 1/8.0 * node_ref[:, 0] * (1 + eta * node_ref[:, 1]) * (1 + zeta * node_ref[:, 2])
    dNqp_deta = 1/8.0 * node_ref[:, 1] * (1 + xi  * node_ref[:, 0]) * (1 + zeta * node_ref[:, 2])
    dNqp_dzeta = 1/8.0 * node_ref[:, 2] * (1 + xi  * node_ref[:, 0]) * (1 + eta  * node_ref[:, 1])

    # Broadcast each derivative to shape (n_qp, 8)
    dNqp_dxi = jnp.broadcast_to(dNqp_dxi, Nqp.shape)
    dNqp_deta = jnp.broadcast_to(dNqp_deta, Nqp.shape)
    dNqp_dzeta = jnp.broadcast_to(dNqp_dzeta, Nqp.shape)

    # Stack to get derivatives of shape (n_qp, 8, 3)
    dNqp_dLocal = jnp.stack([dNqp_dxi, dNqp_deta, dNqp_dzeta], axis=-1)

    return Nqp, dNqp_dLocal


@jit
def gauss_quad():
    """
    Return the Gauss quadrature points and weights for a 3D hexahedron.

    Parameters:
      n_qp: Number of quadrature points in each direction.

    Returns:
      gauss_pts: Array of quadrature points of shape (n_qp,).
      gauss_wts: Array of quadrature weights of shape (n_qp,).
    """

    # Define Gauss quadrature points (2-point rule in each direction)
    gauss_pts = jnp.array([-1.0 / jnp.sqrt(3.0), 1.0 / jnp.sqrt(3.0)])
    gauss_wts = jnp.array([1.0, 1.0])

    return gauss_pts, gauss_wts
