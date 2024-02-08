import torch
from ..utilities.aggregation import kreisselmeier_steinhauser_max, kreisselmeier_steinhauser_min

def softmax(x, tau=1.0):
    return torch.sum(x * torch.exp(x / tau)) / torch.sum(torch.exp(x / tau))

def softmin(x, tau=1.0):
    return -softmax(-x, tau)

def bounding_box_bounds(positions, radii):
    """
    Calculate the bounding box that contains all spheres.

    Parameters:
    spheres (Tensor): A tensor of shape (n, 4) where each row is [x, y, z, radius]

    Returns:
    Tensor: A tensor of shape (2, 3) representing the two opposite vertices (min and max) of the bounding box.
    """

    # Calculate min and max coordinates for each sphere
    min_coords = positions - radii.view(-1, 1)
    max_coords = positions + radii.view(-1, 1)

    # Find overall min and max coordinates
    x_min, y_min, z_min = torch.min(min_coords, dim=0)[0].reshape(3, 1)
    x_max, y_max, z_max = torch.max(max_coords, dim=0)[0].reshape(3, 1)

    # Combine into a single tensor representing the bounding box
    bounds = torch.concatenate((x_min, x_max, y_min, y_max, z_min, z_max))

    return bounds

def smooth_bounding_box_bounds(positions, radii, rho=100, tau=0.01):

    # Calculate min and max coordinates for each sphere
    min_coords = positions - radii.view(-1, 1)
    max_coords = positions + radii.view(-1, 1)

    # # Find overall min and max coordinates
    # x_min = kreisselmeier_steinhauser_min(min_coords[:, 0], rho=rho)
    # y_min = kreisselmeier_steinhauser_min(min_coords[:, 1], rho=rho)
    # z_min = kreisselmeier_steinhauser_min(min_coords[:, 2], rho=rho)
    #
    # x_max = kreisselmeier_steinhauser_max(max_coords[:, 0], rho=rho)
    # y_max = kreisselmeier_steinhauser_max(max_coords[:, 1], rho=rho)
    # z_max = kreisselmeier_steinhauser_max(max_coords[:, 2], rho=rho)

    # Find overall min and max coordinates
    x_min = softmin(min_coords[:, 0], tau=tau)
    y_min = softmin(min_coords[:, 1], tau=tau)
    z_min = softmin(min_coords[:, 2], tau=tau)

    x_max = softmax(max_coords[:, 0], tau=tau)
    y_max = softmax(max_coords[:, 1], tau=tau)
    z_max = softmax(max_coords[:, 2], tau=tau)

    # Combine into a single tensor representing the bounding box
    bounds = torch.tensor([x_min, x_max, y_min, y_max, z_min, z_max])

    return bounds

def bounding_box_volume(bounds):

    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    volume = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)

    return volume


