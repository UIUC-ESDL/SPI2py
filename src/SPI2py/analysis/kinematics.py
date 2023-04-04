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
