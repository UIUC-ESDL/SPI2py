import math
import jax.numpy as jnp
from jax import jit


def generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max, element_size=1.0):
    """
    Generate a cubic mesh for a rectangular domain.

    The domain is defined by the ranges [x_min, x_max], [y_min, y_max], [z_min, z_max].
    The mesh cells are cubes with side length element_size.

    Returns:
      nodes: An array of shape ((nx+1)*(ny+1)*(nz+1), 3) with the vertex coordinates.
      elements: An array of shape (nx*ny*nz, 8) with the connectivity (indices into nodes)
                for each hexahedral cell.
      centers: An array of shape (nx*ny*nz, 3) with the center coordinates of each element.
    """

    # Calculate the element length
    lx = x_max - x_min
    ly = y_max - y_min
    lz = z_max - z_min

    # Determine number of cells (elements) along each axis.
    nx = math.ceil(lx / element_size)
    ny = math.ceil(ly / element_size)
    nz = math.ceil(lz / element_size)

    # Generate vertex coordinates along each axis.
    x = jnp.linspace(x_min, x_min + lx, nx + 1)
    y = jnp.linspace(y_min, y_min + ly, ny + 1)
    z = jnp.linspace(z_min, z_min + lz, nz + 1)

    # Create a 3D meshgrid of node positions.
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    X = X.ravel()
    Y = Y.ravel()
    Z = Z.ravel()

    nodes = jnp.stack([X, Y, Z], axis=-1)

    # Create the element connectivity.
    # Each element is defined by its "base" indices in the grid of cells.
    i = jnp.arange(nx)
    j = jnp.arange(ny)
    k = jnp.arange(nz)
    I, J, K = jnp.meshgrid(i, j, k, indexing='ij')
    I = I.ravel()
    J = J.ravel()
    K = K.ravel()

    # Node numbering in the grid of vertices:
    # index = i * ((ny+1) * (nz+1)) + j * (nz+1) + k.
    stride_j = (nz + 1)
    stride_i = (ny + 1) * (nz + 1)

    def idx(i, j, k):
        return i * stride_i + j * stride_j + k

    # For each cell, compute the indices of its 8 vertices.
    n0 = idx(I, J, K)
    n1 = idx(I + 1, J, K)
    n2 = idx(I + 1, J + 1, K)
    n3 = idx(I, J + 1, K)
    n4 = idx(I, J, K + 1)
    n5 = idx(I + 1, J, K + 1)
    n6 = idx(I + 1, J + 1, K + 1)
    n7 = idx(I, J + 1, K + 1)
    elements = jnp.stack([n0, n1, n2, n3, n4, n5, n6, n7], axis=-1)

    # Compute centers of each element.
    # The center along each axis is x_min + element_size/2 + i * element_size,
    # for i = 0,...,nx-1 (similarly for y and z).
    centers_x = x_min + element_size / 2 + jnp.arange(nx) * element_size
    centers_y = y_min + element_size / 2 + jnp.arange(ny) * element_size
    centers_z = z_min + element_size / 2 + jnp.arange(nz) * element_size

    C_X, C_Y, C_Z = jnp.meshgrid(centers_x, centers_y, centers_z, indexing='ij')
    C_X = C_X.ravel()
    C_Y = C_Y.ravel()
    C_Z = C_Z.ravel()
    centers = jnp.stack([C_X, C_Y, C_Z], axis=-1)

    return nodes, elements, centers, nx, ny, nz, lx, ly, lz


def find_active_nodes(element_densities, elements, threshold=1e-3):
    """
    Given an array of element densities (one value per element) and an
    elements connectivity array of shape (n_elem, 8) (with each row containing the
    8 node indices for that element), returns a 1D array of unique node indices
    for all elements with density above the threshold.

    Parameters
    ----------
    element_densities : jnp.ndarray, shape (n_elem,)
        The density value for each element.
    elements : jnp.ndarray, shape (n_elem, 8)
        Connectivity information (node indices) for each element.
    threshold : float, optional
        The density threshold. Default is 1e-3.

    Returns
    -------
    active_node_indices : jnp.ndarray, shape (n_active_nodes,)
        A sorted 1D array of node indices that belong to elements
        with density above the threshold.
    """

    # Flatten densities to match elements
    densities_flat = element_densities.flatten()

    # Identify the indices of elements whose density is above the threshold.
    active_elem_idx = jnp.where(densities_flat > threshold)[0]

    # Use these element indices to get the corresponding node indices.
    # This will produce an array of shape (n_active_elems, 8)
    active_nodes = elements[active_elem_idx]

    # Flatten the array to a 1D list of node indices.
    active_nodes_flat = active_nodes.flatten()

    # Get the unique node indices.
    active_node_indices = jnp.unique(active_nodes_flat)

    return active_node_indices


def find_face_nodes(nodes: jnp.ndarray, face_normal: jnp.ndarray, tol: float = 1e-6) -> jnp.ndarray:
    """
    Given an array of nodes (shape: [n_nodes, 3]) and a face normal (e.g., [0, 1, 0] for the top),
    return the indices of nodes on the face corresponding to the maximum projection in that direction.

    Parameters:
      nodes: (n_nodes, 3) array of node coordinates.
      face_normal: A 3-element array indicating the direction of the face normal.
                   For example, [0, 1, 0] for the top face.
      tol: Tolerance for deciding if a node is on the face.

    Returns:
      A 1D array of indices corresponding to nodes on that face.
    """
    # Normalize the face normal.
    n_unit = face_normal / jnp.linalg.norm(face_normal)

    # Compute dot products between each node and the face normal.
    dots = nodes @ n_unit  # shape: (n_nodes,)

    # The face we want is the one with the maximum projection.
    max_dot = jnp.max(dots)

    # Select nodes that are within 'tol' of the maximum dot product.
    face_indices = jnp.where(max_dot - dots < tol)[0]
    return face_indices
