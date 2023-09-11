import torch


def bounding_box(positions_array):

    # TODO include radii

    min_x = torch.min(positions_array[:, 0])
    max_x = torch.max(positions_array[:, 0])
    min_y = torch.min(positions_array[:, 1])
    max_y = torch.max(positions_array[:, 1])
    min_z = torch.min(positions_array[:, 2])
    max_z = torch.max(positions_array[:, 2])

    volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    return volume
