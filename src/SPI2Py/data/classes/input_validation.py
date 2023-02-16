"""

"""

import warnings
from typing import Union
import numpy as np
import matplotlib.colors as mcolors



class InputValidation:
    def __init__(self,
                 name: str,
                 positions: np.ndarray,
                 rotation: np.ndarray,
                 radii: np.ndarray,
                 color: Union[str, list[str]],
                 movement: str,
                 reference: Union[str, None],
                 degrees_of_freedom: Union[list[int], None]):

        self.name = name
        self.positions = self._validate_positions(positions)
        self.rotation = self._validate_rotation(rotation)
        self.radii = self._validate_radii(radii)
        self.color = self._validate_colors(color)
        self.movement = self._validate_movement(movement)
        self.reference = self._validate_reference(reference)
        self.degrees_of_freedom = self._validate_degrees_of_freedom(degrees_of_freedom)



    def _validate_position(self, position) -> np.ndarray:
        return position

    def _validate_rotation(self, rotation) -> np.ndarray:
        return rotation

    def _validate_positions(self, positions) -> np.ndarray:

        if positions is None:
            raise ValueError('Positions have not been set for %s.' % self.name)

        if not isinstance(positions, list) and not isinstance(positions, np.ndarray):
            raise ValueError('Positions must be a list or numpy array for %s not %s.' % (self.name, type(positions)))

        if isinstance(positions, list):
            warnings.warn('Positions should be a numpy array for %s.' % self.name)
            positions = np.array(positions)

        if len(positions.shape) == 1:
            warnings.warn('Positions are not 2D for %s.' % self.name)
            positions = positions.reshape(-1, 3)

        if positions.shape[1] != 3:
            raise ValueError('Positions must be 3D for %s.' % self.name)

        return positions

    def _validate_radii(self, radii) -> np.ndarray:

            if radii is None:
                raise ValueError('Radii have not been set for %s.' % self.name)

            if isinstance(radii, float):
                warnings.warn('Radii should be a numpy array for %s.' % self.name)
                radii = np.array([radii])

            if isinstance(radii, list):
                warnings.warn('Radii should be a numpy array for %s.' % self.name)
                radii = np.array(radii)

            if len(radii.shape) > 1:
                warnings.warn('Radii should be 1D for %s.' % self.name)
                radii = radii.reshape(-1)

            if radii.shape[0] != self.positions.shape[0]:
                raise ValueError('There must be 1 radius for each position row for %s.' % self.name)

            return radii

    def _validate_color(self, color):

        if isinstance(color, str):
            pass
        else:
            raise ValueError('Colors must be a string for %s.' % self.name)

        self.valid_colors = {**mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS,
                             **mcolors.XKCD_COLORS}

        if color in self.valid_colors:
            pass

        else:

            raise ValueError('Color not recognized for %s. For a list of valid colors inspect the attribute '
                             'self.valid_colors.keys().' % self.name)

    def _validate_colors(self, colors):

        if colors is None:
            raise ValueError('Color has not been set for %s.' % self.name)

        if isinstance(colors, list):

            if len(colors) == 1:
                self._validate_color(colors)

            if len(colors) > 1:
                for color in colors:
                    self._validate_color(color)
        elif isinstance(colors, str):
            self._validate_color(colors)
        else:
            raise ValueError('Colors must be a list or string for %s.' % self.name)

        return colors

    def _validate_movement(self, movement):

        if not isinstance(movement, str):
            raise TypeError('Movement must be a string for %s.' % self.name)

        return movement

    def _validate_reference(self, reference):
        return reference

    def _validate_degrees_of_freedom(self, degrees_of_freedom):

        if not isinstance(degrees_of_freedom, tuple) and degrees_of_freedom is not None:
            raise TypeError('Degrees of freedom must be a tuple for %s.' % self.name)

        if degrees_of_freedom is not None:
            for dof in degrees_of_freedom:
                if dof not in ('x', 'y', 'z', 'rx', 'ry', 'rz'):
                    raise ValueError('Invalid DOF specified for %s.' % self.name)

        return degrees_of_freedom


