import torch


def bounding_box(spheres):
    """
    Calculate the bounding box that contains all spheres.

    Parameters:
    spheres (Tensor): A tensor of shape (n, 4) where each row is [x, y, z, radius]

    Returns:
    Tensor: A tensor of shape (2, 3) representing the two opposite vertices (min and max) of the bounding box.
    """

    # Calculate min and max coordinates for each sphere
    min_coords = spheres[:, :3] - spheres[:, 3].view(-1, 1)
    max_coords = spheres[:, :3] + spheres[:, 3].view(-1, 1)

    # Find overall min and max coordinates
    overall_min = torch.min(min_coords, dim=0)[0]
    overall_max = torch.max(max_coords, dim=0)[0]

    # Combine into a single tensor representing the bounding box
    bounding_box_vertices = torch.stack([overall_min, overall_max])

    return bounding_box_vertices


def bounding_box_volume(spheres):

    (x_min, y_min, z_min), (x_max, y_max, z_max) = bounding_box(spheres)

    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    return volume

