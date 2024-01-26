import torch


def translate_linear_spline(positions, start_point, control_points, end_point, num_spheres_per_segment):

    # Combine all control points into a single tensor
    node_positions = torch.vstack((start_point, control_points, end_point))

    # Obtain the start and stop points for each line segment
    start_positions = node_positions.view(-1, 6)[:, [0, 1, 2]]
    stop_positions = node_positions.view(-1, 6)[:, [3, 4, 5]]

    # Calculate the translations for each line segment
    translations = torch.empty((0, 3), dtype=torch.float64)
    for start_position, stop_position in zip(start_positions, stop_positions):
        segment_positions = compute_line_segment_intermediate_positions(start_position, stop_position, num_spheres_per_segment)
        translations = torch.vstack((translations, segment_positions))

    # Apply the translations
    translated_positions = positions + translations

    return translated_positions

def compute_line_segment_intermediate_positions(start_position, stop_position, num_spheres_per_segment):

    x_start = start_position[0]
    y_start = start_position[1]
    z_start = start_position[2]

    x_stop = stop_position[0]
    y_stop = stop_position[1]
    z_stop = stop_position[2]

    x_step = (x_stop - x_start) / (num_spheres_per_segment - 1)
    y_step = (y_stop - y_start) / (num_spheres_per_segment - 1)
    z_step = (z_stop - z_start) / (num_spheres_per_segment - 1)

    x_arange = torch.arange(x_start, x_stop + x_step, x_step)
    y_arange = torch.arange(y_start, y_stop + y_step, y_step)
    z_arange = torch.arange(z_start, z_stop + z_step, z_step)

    positions = torch.vstack((x_arange, y_arange, z_arange)).T

    return positions








