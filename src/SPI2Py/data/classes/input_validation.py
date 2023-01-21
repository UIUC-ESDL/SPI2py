"""

"""

import warnings
import numpy as np
import matplotlib.colors as mcolors



class InputValidation:
    def __init__(self, name, positions, radii, color):

        self.name = name
        self.positions = self._validate_positions(positions)
        self.radii = self._validate_radii(radii)
        self.color = self._validate_colors(color)

    def _validate_position(self, position) -> np.ndarray:
        pass

    def _validate_positions(self, positions) -> np.ndarray:

        if positions is None:
            raise ValueError('Positions have not been set for %d.' % self.name)

        if not isinstance(positions, list) and not isinstance(positions, np.ndarray):
            raise ValueError('Positions must be a list or numpy array for %d.' % self.name)

        if isinstance(positions, list):
            warnings.warn('Positions should be a numpy array for %d.' % self.name)
            positions = np.array(positions)

        if len(positions.shape) == 1:
            warnings.warn('Positions are not 2D for %d.' % self.name)
            positions = positions.reshape(-1, 3)

        if positions.shape[1] != 3:
            raise ValueError('Positions must be 3D for %d.' % self.name)

        return positions

    def _validate_radii(self, radii) -> np.ndarray:

            if radii is None:
                raise ValueError('Radii have not been set for %d.' % self.name)

            if isinstance(radii, list):
                warnings.warn('Radii should be a numpy array for %d.' % self.name)
                radii = np.array(radii)

            if len(radii.shape) > 1:
                warnings.warn('Radii should be 1D for %d.' % self.name)
                radii = radii.reshape(-1)

            if radii.shape[0] != self.positions.shape[0]:
                raise ValueError('There must be 1 radius for each position row for %d.' % self.name)

            return radii

    def _validate_color(self, color):

        if isinstance(color, str):
            pass
        else:
            raise ValueError('Colors must be a string for %d.' % self.name)

        if color in mcolors.BASE_COLORS:
            pass
        elif color in mcolors.TABLEAU_COLORS:
            pass
        elif color in mcolors.CSS4_COLORS:
            pass
        elif color in mcolors.XKCD_COLORS:
            pass
        else:
            raise ValueError('Color not recognized for %d.' % self.name)

    def _validate_colors(self, colors):

        if colors is None:
            raise ValueError('Color has not been set for %d.' % self.name)

        if isinstance(colors, list):

            if len(colors) == 1:
                self._validate_color(colors)

            if len(colors) > 1:
                for color in colors:
                    self._validate_color(color)
        elif isinstance(colors, str):
            self._validate_color(colors)
        else:
            raise ValueError('Colors must be a list or string for %d.' % self.name)

        return colors
