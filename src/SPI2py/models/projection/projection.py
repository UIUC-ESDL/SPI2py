import jax.numpy as jnp
from .encapsulation import overlap_volume_sphere_sphere

def calculate_pseudo_densities(sphere_positions, sphere_radii, element_length, centers, sample_points, sample_radii, rho_min):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (nx, ny, nz, n_mesh_points, 1, 3) tensor
    mesh_radii: (nx, ny, nz, n_mesh_points, 1) tensor
    object_positions_expanded: (1, 1, 1, 1, n_object_points, 3) tensor
    object_radii_expanded: (1, 1, 1, 1, n_object_points) tensor

    pseudo_densities: (nx, ny, nz, 1) tensor
    """

    nx, ny, nz, _ = centers.shape

    element_volumes = (4 / 3) * jnp.pi * sample_radii ** 3

    object_positions = sphere_positions
    object_radii = sphere_radii
    object_radii_transposed = object_radii.T
    object_radii_expanded = object_radii_transposed[None, None, None, ...]

    # Assuming sample_points, object_positions, object_radii_expanded, sample_radii, and element_volumes are JAX arrays
    distances = jnp.linalg.norm(sample_points[..., None, :] - object_positions[None, None, None, None, :, :], axis=-1)
    volume_overlaps = overlap_volume_sphere_sphere(object_radii_expanded, sample_radii, distances)
    volume_fractions = volume_overlaps / element_volumes
    pseudo_densities = jnp.sum(volume_fractions, axis=(3, 4), keepdims=True).squeeze(3)

    # Clip the pseudo-densities
    pseudo_densities = jnp.clip(pseudo_densities, a_min=rho_min, a_max=1)

    return pseudo_densities