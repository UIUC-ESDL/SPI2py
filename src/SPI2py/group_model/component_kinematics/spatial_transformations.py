"""

"""

import torch


def apply_homogenous_transformation(reference_point, positions, transformation_matrix):
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


