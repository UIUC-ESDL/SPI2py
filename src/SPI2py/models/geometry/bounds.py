import jax.numpy as jnp
from ..utilities.aggregation import kreisselmeier_steinhauser_max, kreisselmeier_steinhauser_min


def bounding_box_bounds_points(positions):

    # Find overall min and max coordinates
    x_min, y_min, z_min = jnp.min(positions, axis=0).reshape(3, 1)
    x_max, y_max, z_max = jnp.max(positions, axis=0).reshape(3, 1)

    # positions = positions.reshape(-1, 3)
    # x_min, y_min, z_min = kreisselmeier_steinhauser_min(positions)
    # x_max, y_max, z_max = kreisselmeier_steinhauser_max(positions)

    # Combine into a single tensor representing the bounding box
    bounds = jnp.concatenate((x_min, x_max, y_min, y_max, z_min, z_max))

    return bounds


def bounding_box_bounds(positions, radii):
    """
    Calculate the bounding box that contains all spheres.

    Parameters:
    spheres (Tensor): A tensor of shape (n, 4) where each row is [x, y, z, radius]

    Returns:
    Tensor: A tensor of shape (2, 3) representing the two opposite vertices (min and max) of the bounding box.
    """

    # TODO Undo
    # Add a point at the origin
    positions = jnp.vstack([jnp.zeros((1, 3)), positions])
    radii = jnp.vstack([0.1* jnp.ones((1, 1)), radii])

    # Calculate min and max coordinates for each sphere
    min_coords = positions - radii.reshape(-1, 1)
    max_coords = positions + radii.reshape(-1, 1)

    # Find overall min and max coordinates
    x_min, y_min, z_min = jnp.min(min_coords, axis=0).reshape(3, 1)
    x_max, y_max, z_max = jnp.max(max_coords, axis=0).reshape(3, 1)

    # Combine into a single tensor representing the bounding box
    bounds = jnp.concatenate((x_min, x_max, y_min, y_max, z_min, z_max))

    return bounds


def bounding_box_volume(bounds):

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    return volume
