import jax.numpy as np
from typing import Union
import logging
from .spatial_transformations import rigid_transformation
logger = logging.getLogger(__name__)


class RigidBody:

    def __init__(self,
                 positions: np.ndarray,
                 radii: np.ndarray,
                 movement_class: str,
                 reference_axes: str = 'origin',
                 degrees_of_freedom: Union[tuple[str], None] = ('x', 'y', 'z', 'rx', 'ry', 'rz')):

        # TODO Remove
        self.positions = positions
        self.radii = radii

        # TODO Remove rotation...
        self.rotation = np.zeros(3)
        self.reference_axes = reference_axes
        self.movement_class = movement_class
        self.degrees_of_freedom = degrees_of_freedom

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
            if dof == ():
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