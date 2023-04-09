"""
TODO Replace object calls with functional programming
"""

import numpy as np
from typing import Union
from .transformations import translate, rotate

def calculate_static_positions(self, positions_dict):
    positions_dict[str(self)] = (self.positions, self.radii)

    return positions_dict


def calculate_independent_positions(self,
                                    design_vector: np.ndarray,
                                    positions_dict: Union[None, dict] = None) -> dict:
    """

    """

    new_positions = np.copy(self.positions)

    # TODO Fix workaround. For now assume 1x3 = translation nand 1x6 = translation and rotation

    # if self.three_d_translation is True:
    #     new_reference_position = design_vector[0:3]
    #     new_positions = translate(new_positions, self.reference_position, new_reference_position)
    #
    # if self.three_d_rotation is True:
    #     rotation = design_vector[3:None]
    #     new_positions = rotate_about_point(new_positions, rotation)

    if len(design_vector) >= 3:
        new_reference_position = design_vector[0:3]
        new_positions = translate(new_positions, self.reference_position, new_reference_position)

    if len(design_vector) == 6:
        rotation = design_vector[3:None]
        new_positions = rotate(new_positions, rotation)

    # TODO Check for constrained movement cases

    positions_dict[str(self)] = (new_positions, self.radii)

    return positions_dict


def calculate_dependent_positions(self,
                               design_vector:  np.ndarray,
                               positions_dict: Union[None, dict] = None) -> dict:
    """
    Types of Constrained Motion

    Fully Dependent Constraints
    1. "offset translation and rotation"
    2. ...(?)

    #
    2. "constant translation offset variable rotation"
    2. "variable translation constant rotation offset
    2. "colinear" (not implemented)
    3. "colinear with offset" (not implemented)
    """

    def offset_translation_and_rotation_(self, positions_dict):
        # TODO Remove design vector argument
        # Get the reference point
        reference_point = positions_dict[self.component_name][0][0]

        # Calculate the port position
        port_position = reference_point + self.reference_point_offset

        # Add the port position to the positions dictionary
        positions_dict[str(self)] = (port_position, self.radius)

        return positions_dict

    if self.movement_class == 'offset translation and rotation':
        positions_dict = offset_translation_and_rotation_(self, positions_dict)
    else:
        raise NotImplementedError('This type of constrained motion is not implemented.')

    return positions_dict
