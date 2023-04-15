import numpy as np
from typing import Union
import logging
from .transformations import rigid_transformation
logger = logging.getLogger(__name__)


class ValidateRigidBody:
    """
    This class is used to validate the inputs for the RigidBody class.
    """
    def _validate_positions(self, positions: np.ndarray) -> np.ndarray:

        if positions is None:
            raise ValueError('Positions have not been set for %s.' % self.__repr__())

        if not isinstance(positions, list) and not isinstance(positions, np.ndarray):
            raise ValueError(
                'Positions must be a list or numpy array for %s not %s.' % (self.__repr__(), type(positions)))

        if isinstance(positions, list):
            logger.warning('Positions should be a numpy array for %s.' % self.__repr__())
            positions = np.array(positions)

        if len(positions.shape) == 1:
            logger.warning('Positions are not 2D for %s.' % self.__repr__())
            positions = positions.reshape(-1, 3)

        if positions.shape[1] != 3:
            raise ValueError('Positions must be 3D for %s.' % self.__repr__())

        return positions

    def _validate_radii(self, radii: np.ndarray) -> np.ndarray:

        if radii is None:
            raise ValueError('Radii have not been set for %s.' % self.__repr__())

        if isinstance(radii, float):
            logger.warning('Radii should be a numpy array for %s.' % self.__repr__())
            radii = np.array([radii])

        if isinstance(radii, list):
            logger.warning('Radii should be a numpy array for %s.' % self.__repr__())
            radii = np.array(radii)

        if len(radii.shape) > 1:
            logger.warning('Radii should be 1D for %s.' % self.__repr__())
            radii = radii.reshape(-1)

        if radii.shape[0] != self.positions.shape[0]:
            raise ValueError('There must be 1 radius for each position row for %s.' % self.__repr__())

        return radii

    def _validate_movement_class(self, movement_class):

        if not isinstance(movement_class, str):
            raise TypeError('Movement must be a string for %s.' % self.__repr__())

        valid_movement_classes = ['static', 'independent', 'partially_dependent', 'fully_dependent']
        if movement_class not in valid_movement_classes:
            raise ValueError('Movement class not recognized for %s. Valid movement classes are: %s' %
                             (self.__repr__(), valid_movement_classes))

        return movement_class

    def _validate_reference_axes(self, reference_axes):
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

        return reference_axes

    def _validate_degrees_of_freedom(self, degrees_of_freedom):

        if not isinstance(degrees_of_freedom, tuple) and degrees_of_freedom is not None:
            raise TypeError('Degrees of freedom must be a tuple for %s.' % self.name)

        if degrees_of_freedom is not None:
            for dof in degrees_of_freedom:
                if dof not in ('x', 'y', 'z', 'rx', 'ry', 'rz'):
                    raise ValueError('Invalid DOF specified for %s.' % self.name)

        return degrees_of_freedom

class RigidBody(ValidateRigidBody):

    def __init__(self,
                 positions: np.ndarray,
                 radii: np.ndarray,
                 movement_class: str,
                 reference_axes: str = 'origin',
                 degrees_of_freedom: Union[tuple[str], None] = ('x', 'y', 'z', 'rx', 'ry', 'rz')):

        # TODO Remove
        self.positions = self._validate_positions(positions)
        self.radii = self._validate_radii(radii)

        # TODO Remove rotation...
        self.rotation = np.zeros(3)
        self.reference_axes = self._validate_reference_axes(reference_axes)
        self.movement_class = self._validate_movement_class(movement_class)
        self.degrees_of_freedom = self._validate_degrees_of_freedom(degrees_of_freedom)

    @property
    def reference_position(self):
        """
        Returns the reference position of the object.

        TODO Replace with a signed distance function that is consistent regardless of the number of spheres used.
        """
        return self.positions[0]

    @property
    def design_vector_dict(self) -> dict:

        """
        Returns a dictionary of the design vector components.

        An object's design vector is encoded as a reference position and rotation. Since objects are rigid bodies,
        all other geometric properties are a function of the reference position and rotation.
        """

        design_vector_dict = {}

        if 'x' in self.degrees_of_freedom:
            design_vector_dict['x'] = self.reference_position[0]
        if 'y' in self.degrees_of_freedom:
            design_vector_dict['y'] = self.reference_position[1]
        if 'z' in self.degrees_of_freedom:
            design_vector_dict['z'] = self.reference_position[2]
        if 'rx' in self.degrees_of_freedom:
            design_vector_dict['rx'] = self.rotation[0]
        if 'ry' in self.degrees_of_freedom:
            design_vector_dict['ry'] = self.rotation[1]
        if 'rz' in self.degrees_of_freedom:
            design_vector_dict['rz'] = self.rotation[2]

        return design_vector_dict

    @property
    def design_vector(self):
        design_vector = np.array(list(self.design_vector_dict.values()))
        return design_vector

    def decompose_design_vector(self, design_vector: np.ndarray) -> dict:
        """
        Takes a 1D design vector and decomposes it into a dictionary of design variables.
        """

        if len(design_vector) != len(self.degrees_of_freedom):
            raise ValueError('The specified design vector must be the same length as the degrees of freedom.')

        design_vector_dict = {}

        for i, dof in enumerate(self.degrees_of_freedom):
            design_vector_dict[dof] = design_vector[i]

        return design_vector_dict

    def calculate_positions(self, design_vector, objects_dict=None, force_update=False):
        """
        Calculates the positions of the object's spheres.

        Considerations
        6 DOF: x, y, z, rx, ry, rz
        n references axes

        1. Pure translation
        2. Pure rotation
        3. Translation and rotation
        4. Translation/rotation about a reference axis
            -Adopts the reference axis angles and origin position
        5. Translation/rotation about two reference axes (colinear motion)
            -Ignores axes angles, just creates a line between the two axes
        6. Translation/rotation about three reference axes (coplanar motion)
            -Ignores axes angles, just triangulates 3D coordinates into a plane
        """

        dof = self.degrees_of_freedom

        if force_update is True:
            dof = ('x', 'y', 'z', 'rx', 'ry', 'rz')
            x  = design_vector[0]
            y  = design_vector[1]
            z  = design_vector[2]
            rx = design_vector[3]
            ry = design_vector[4]
            rz = design_vector[5]

        else:

            # If the object has no degrees of freedom, then return its current position
            if dof is None:
                return {self.__repr__(): {'type': 'spheres', 'positions': self.positions, 'radii': self.radii}}

            # Extract the design variables from the design vector
            design_vector_dict = self.decompose_design_vector(design_vector)

            if 'x' in dof:
                x = design_vector_dict['x']
            else:
                x = 0

            if 'y' in dof:
                y = design_vector_dict['y']
            else:
                y = 0

            if 'z' in dof:
                z = design_vector_dict['z']
            else:
                z = 0

            if 'rx' in dof:
                rx = design_vector_dict['rx']
            else:
                rx = 0

            if 'ry' in dof:
                ry = design_vector_dict['ry']
            else:
                ry = 0

            if 'rz' in dof:
                rz = design_vector_dict['rz']
            else:
                rz = 0

        # TODO Add reference axes argument
        # Calculate the new positions
        new_positions = rigid_transformation(self.reference_position, self.positions, x, y, z, rx, ry, rz)

        object_dict = {self.__repr__(): {'type': 'spheres', 'positions': new_positions, 'radii': self.radii}}

        if self.ports is not None:

            for i, port in enumerate(self.ports):
                port_name = self.__repr__() + '_' + port['name']
                port_positions = new_positions[self.port_index(port['name'])]
                port_positions = port_positions.reshape(1, 3)
                port_radius = np.array([port['radius']])
                object_dict[port_name] = {'type': 'spheres', 'positions': port_positions, 'radii': port_radius}

        return object_dict


    def set_positions(self,
                      objects_dict: dict = None,
                      design_vector: list = None):
        """
        Update positions of object spheres given a design vector

        :param objects_dict:
        :param design_vector:
        :return:
        """

        if design_vector is not None:
            objects_dict = self.calculate_positions(design_vector, force_update=True)

        self.positions = objects_dict[self.__repr__()]['positions']
        self.radii     = objects_dict[self.__repr__()]['radii']



class FlexibleBody:
    pass