"""

"""

import torch
from torch import sin, cos


def assemble_transformation_matrix(translation, rotation):

    if translation.shape != (3, 1):
        translation = translation.reshape(3, 1)

    if rotation.shape != (3, 1):
        rotation = rotation.reshape(3, 1)

    # Initialize the transformation matrix
    t = torch.eye(4, dtype=torch.float64)

    # Insert the translation vector
    t[:3, [3]] = translation

    # Unpack the rotation angles (Euler)
    a = rotation[0]  # alpha
    b = rotation[1]  # beta
    g = rotation[2]  # gamma

    # Calculate rotation matrix (R = R_z(gamma) @ R_y(beta) @ R_x(alpha))
    r = torch.cat(
        (cos(b) * cos(g), sin(a) * sin(b) * cos(g) - cos(a) * sin(g), cos(a) * sin(b) * cos(g) + sin(a) * sin(g),
         cos(b) * sin(g), sin(a) * sin(b) * sin(g) + cos(a) * cos(g), cos(a) * sin(b) * sin(g) - sin(a) * cos(g),
         -sin(b), sin(a) * cos(b), cos(a) * cos(b))).view(3, 3)

    # Insert the rotation matrix
    t[:3, :3] = r

    return t


def apply_transformation_matrix(reference_point, positions, transformation_matrix):
    """
    Assume transposed...?
    """

    # Center the object about its reference position
    positions_shifted = positions - reference_point

    # Pad the positions with ones
    ones = torch.ones((1, positions_shifted.shape[1]))
    positions_shifted_padded = torch.vstack((positions_shifted, ones))

    # Apply the transformation
    transformed_positions_shifted_padded = transformation_matrix @ positions_shifted_padded

    # Remove the padding
    transformed_positions_shifted = transformed_positions_shifted_padded[:3, :]

    # Shift the object back to its original position
    transformed_positions = transformed_positions_shifted + reference_point

    return transformed_positions



