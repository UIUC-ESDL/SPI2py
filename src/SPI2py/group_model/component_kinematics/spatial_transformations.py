"""

"""

import torch
from torch import sin, cos


def rigid_body_transformation(reference_position, positions, translation, rotation):
    """
    Apply translation and rotation to an object represented by a set of points.

    The transformation matrix is implemented as defined in the following reference:
    LaValle, Steven M. Planning algorithms. Cambridge University press, 2006.

    :param reference_position: The reference position of the object
    :type reference_position: np.array(3,1)

    :param positions: The positions of the object
    :type positions: np.array(3,n)

    :param translation: The translation vector
    :type translation: np.array(3,1)

    :param rotation: The rotation vector
    :type rotation: np.array(3,1)

    :return: transformed_positions: The transformed positions of the object
    :rtype: np.array(3,n)
    """

    # Initialize constants
    zero = torch.zeros(1, dtype=torch.float64)
    one = torch.ones(1, dtype=torch.float64)

    # Initialize the transformation matrix
    t = torch.eye(4, dtype=torch.float64)

    # Insert the translation vector
    t[:3, [3]] = translation

    # Unpack the rotation angles (Euler)
    a = rotation[0]  # alpha
    b = rotation[1]  # beta
    g = rotation[2]  # gamma

    # Calculate rotation matrix (R = R_z(gamma) @ R_y(beta) @ R_x(alpha))
    r = torch.cat((cos(b) * cos(g), sin(a) * sin(b) * cos(g) - cos(a) * sin(g), cos(a) * sin(b) * cos(g) + sin(a) * sin(g),
                   cos(b) * sin(g), sin(a) * sin(b) * sin(g) + cos(a) * cos(g), cos(a) * sin(b) * sin(g) - sin(a) * cos(g),
                   -sin(b),         sin(a) * cos(b),                            cos(a) * cos(b))).view(3, 3)

    # Insert the rotation matrix
    t[:3, :3] = r

    # Center the object about its reference position
    positions_shifted = positions - reference_position

    # Pad the positions with ones
    ones = torch.ones((1, positions_shifted.shape[1]))
    positions_shifted_padded = torch.vstack((positions_shifted, ones))

    # Apply the transformation
    transformed_positions_shifted_padded = t @ positions_shifted_padded

    # Remove the padding
    transformed_positions_shifted = transformed_positions_shifted_padded[:3, :]

    # Shift the object back to its original position
    transformed_positions = transformed_positions_shifted + reference_position

    return transformed_positions
