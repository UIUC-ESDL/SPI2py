import torch



def bounding_box(x, model):

    # TODO include radii
    # TODO verify calc

    positions_dict = model.calculate_positions(design_vector=x)

    positions_array = torch.vstack([positions_dict[key]['positions'] for key in positions_dict.keys()])

    min_x = torch.min(positions_array[:, 0])
    max_x = torch.max(positions_array[:, 0])
    min_y = torch.min(positions_array[:, 1])
    max_y = torch.max(positions_array[:, 1])
    min_z = torch.min(positions_array[:, 2])
    max_z = torch.max(positions_array[:, 2])

    volume = (max_x - min_x) * (max_y - min_y) * (max_z - min_z)

    return volume
