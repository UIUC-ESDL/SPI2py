import jax.numpy as jnp
from .encapsulation import overlap_volume_sphere_sphere

def calculate_pseudo_densities(sphere_positions, sphere_radii, sample_points, sample_radii):
    """
    Projects the points to the mesh and calculates the pseudo-densities

    mesh_positions: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1, 3) tensor
    mesh_radii: (n_el_x, n_el_y, n_el_z, n_mesh_points, 1) tensor
    object_positions_expanded: (1, 1, 1, 1, n_object_points, 3) tensor
    object_radii_expanded: (1, 1, 1, 1, n_object_points) tensor

    pseudo_densities: (n_el_x, n_el_y, n_el_z, 1) tensor
    """

    element_sample_volumes = (4 / 3) * jnp.pi * sample_radii ** 3
    element_volumes = jnp.sum(element_sample_volumes, axis=3, keepdims=True)


    object_positions = sphere_positions
    object_radii = sphere_radii
    object_radii_transposed = object_radii.T
    object_radii_expanded = object_radii_transposed[None, None, None, ...]

    # Assuming sample_points, object_positions, object_radii_expanded, sample_radii, and element_volumes are JAX arrays
    distances = jnp.linalg.norm(sample_points[..., None, :] - object_positions[None, None, None, None, :, :], axis=-1)
    volume_sample_overlaps = overlap_volume_sphere_sphere(object_radii_expanded, sample_radii, distances)
    volume_element_overlaps = jnp.sum(volume_sample_overlaps, axis=4, keepdims=True)
    volume_fractions = volume_element_overlaps / element_volumes
    pseudo_densities = jnp.sum(volume_fractions, axis=(3, 4), keepdims=True).squeeze(3)

    # Clip the pseudo-densities
    pseudo_densities = jnp.clip(pseudo_densities, a_min=0, a_max=1)

    return pseudo_densities
