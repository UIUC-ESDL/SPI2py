import torch
import jax.numpy as jnp


# def overlap_volume_sphere_sphere(r_1, r_2, d):
#
#     # Calculate volumes for all spheres
#     volume_1 = (4 / 3) * torch.pi * r_1.pow(3)
#     volume_2 = (4 / 3) * torch.pi * r_2.pow(3)
#
#     # Calculate intersection volume for all pairs, assuming overlapping but not fully enclosed
#     numerator = torch.pi * (r_1 + r_2 - d) ** 2 * (
#             d ** 2 + 2 * d * r_2 - 3 * r_2 ** 2 + 2 * d * r_1 + 6 * r_2 * r_1 - 3 * r_1 ** 2)
#     denominator = 12 * d
#     intersection_volume = torch.where(denominator != 0, numerator / denominator, torch.zeros_like(d))
#
#     # Condition for when one sphere is fully within another (not touching boundary)
#     fully_inside = d + torch.min(r_1, r_2) <= torch.max(r_1, r_2)
#
#     # When one sphere is fully inside another, use the volume of the smaller sphere
#     overlap_volume = torch.where(fully_inside, torch.minimum(volume_1, volume_2), intersection_volume)
#
#     # Condition for no overlap (d >= r_1 + r_2)
#     no_overlap = d >= (r_1 + r_2)
#     overlap_volume = torch.where(no_overlap, torch.zeros_like(d), overlap_volume)
#
#     return overlap_volume

def overlap_volume_sphere_sphere(r_1, r_2, d):
    # Calculate volumes for all spheres
    volume_1 = (4 / 3) * jnp.pi * r_1 ** 3
    volume_2 = (4 / 3) * jnp.pi * r_2 ** 3

    # Calculate intersection volume for all pairs, assuming overlapping but not fully enclosed
    numerator = jnp.pi * (r_1 + r_2 - d) ** 2 * (
            d ** 2 + 2 * d * r_2 - 3 * r_2 ** 2 + 2 * d * r_1 + 6 * r_2 * r_1 - 3 * r_1 ** 2)
    denominator = 12 * d
    intersection_volume = jnp.where(denominator != 0, numerator / denominator, jnp.zeros_like(d))

    # Condition for when one sphere is fully within another (not touching boundary)
    fully_inside = d + jnp.minimum(r_1, r_2) <= jnp.maximum(r_1, r_2)

    # When one sphere is fully inside another, use the volume of the smaller sphere
    overlap_volume = jnp.where(fully_inside, jnp.minimum(volume_1, volume_2), intersection_volume)

    # Condition for no overlap (d >= r_1 + r_2)
    no_overlap = d >= (r_1 + r_2)
    overlap_volume = jnp.where(no_overlap, jnp.zeros_like(d), overlap_volume)

    return overlap_volume