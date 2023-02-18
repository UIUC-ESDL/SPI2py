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
                 radii: np.ndarray,
                 color: Union[str, list[str]],
                 movement_class: str,
                 constraints: Union[None, dict],
                 degrees_of_freedom: Union[list[int], None]):

        self.name               = self._validate_name(name)
        self.positions          = self._validate_positions(positions)
        self.radii              = self._validate_radii(radii)
        self.color              = self._validate_colors(color)
        self.movement_class     = self._validate_movement_class(movement_class)
        # self.rotation           = self._validate_rotation(rotation)
        self.constraints        = self._validate_constraints(constraints)
        self.degrees_of_freedom = self._validate_degrees_of_freedom(degrees_of_freedom)


    def _validate_name(self, name):

        if not isinstance(name, str):
            raise TypeError('Name must be a string not %s.' % type(name))

        return name

    def _validate_position(self, position) -> np.ndarray:
        # TODO Implement this function
        return position

    # def _validate_rotation(self, rotation) -> np.ndarray:
    #     # TODO Implement this function
    #     return rotation

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

    def _validate_movement_class(self, movement_class):

        if not isinstance(movement_class, str):
            raise TypeError('Movement must be a string for %s.' % self.name)

        valid_movement_classes = ['static','independent', 'partially_dependent', 'fully_dependent']

        return movement_class

    def _validate_constraints(self, constraints):
        # TODO Ensure that is no reference objects are not specified that movement class isn't dependent
        # TODO Add logic to ensure dynamic fully dependent objects don't reference other dynamic fully dependent objects
        # TODO Update for dictionary and None...
        # if reference_objects is None:
        #
        #     # Quick workaround - include independent since "dependent" is in it...
        #     if 'independent' in self.movement_class:
        #         pass
        #     elif 'dependent' in self.movement_class:
        #         raise ValueError('Reference objects must be specified for dependent movement for %s.' % self.name)
        #     else:
        #         pass
        #
        # elif isinstance(reference_objects, str):
        #     # TODO Add a system-integration test to ensure that only valid reference objects are specified
        #     pass
        #
        # elif isinstance(reference_objects, list):
        #     for reference_object in reference_objects:
        #         if not isinstance(reference_object, str):
        #             raise TypeError('Reference objects must be a string for %s.' % self.name)
        #
        #         # TODO Add a system-integration test to ensure that only valid reference objects are specified
        # else:
        #     raise TypeError('Reference objects must be NoneType, a string or a list for %s.' % self.name)

        return constraints

    def _validate_degrees_of_freedom(self, degrees_of_freedom):

        if not isinstance(degrees_of_freedom, tuple) and degrees_of_freedom is not None:
            raise TypeError('Degrees of freedom must be a tuple for %s.' % self.name)

        if degrees_of_freedom is not None:
            for dof in degrees_of_freedom:
                if dof not in ('x', 'y', 'z', 'rx', 'ry', 'rz'):
                    raise ValueError('Invalid DOF specified for %s.' % self.name)

        return degrees_of_freedom
