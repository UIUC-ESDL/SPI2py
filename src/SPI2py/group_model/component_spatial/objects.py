"""
TODO Can I remove movement classes if I just use degrees of freedom and reference axes?

"""


from SPI2py.group_model.component_geometry.finite_sphere_method import read_xyzr_file, generate_rectangular_prisms
from SPI2py.group_model.component_spatial.spatial_transformations import affine_transformation

from matplotlib import colors as mcolors


import jax.numpy as np
from typing import Union, Sequence

import pyvista as pv


class Component:
    """
    TODO Update set position to just set the origin.. positions should be a SDF(?)
    """
    def __init__(self,
                 name: str,
                 color: str = None,
                 degrees_of_freedom: Sequence[str] = ('x', 'y', 'z', 'rx', 'ry', 'rz'),
                 filepath=None,
                 ports: list[dict] = []):

        self.name = name


        self.positions, self.radii = read_xyzr_file(filepath)
        self.rotation = np.array([0, 0, 0])

        self.ports = ports
        self.port_indices = {}

        # TODO Set default design vectors...

        self.degrees_of_freedom = degrees_of_freedom


        self.color = color

        # TODO Make tuple len zero not none
        self.dof = len(self.degrees_of_freedom)

        if self.ports is not None:
            for port in self.ports:
                self.positions = np.vstack((self.positions, port['origin']))
                self.radii = np.concatenate((self.radii, np.array([port['radius']])))
                self.port_indices[port['name']] = len(self.positions - 1)


    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def reference_position(self):
        """
        Returns the reference position of the object.
        """
        return np.mean(self.positions, axis=0)



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

    def assemble_transformation_vectors(self, design_vector_dict):

        translation = np.zeros((3,1))
        rotation = np.zeros((3,1))
        scale = np.ones((3,1))

        if 'x' in self.degrees_of_freedom:
            translation = translation.at[0].set(design_vector_dict['x'])
        if 'y' in self.degrees_of_freedom:
            translation = translation.at[1].set(design_vector_dict['y'])
        if 'z' in self.degrees_of_freedom:
            translation = translation.at[2].set(design_vector_dict['z'])

        if 'rx' in self.degrees_of_freedom:
            rotation = rotation.at[0].set(design_vector_dict['rx'])
        if 'ry' in self.degrees_of_freedom:
            rotation = rotation.at[1].set(design_vector_dict['ry'])
        if 'rz' in self.degrees_of_freedom:
            rotation = rotation.at[2].set(design_vector_dict['rz'])

        if 'sx' in self.degrees_of_freedom:
            scale[0] = design_vector_dict['sx']
        if 'sy' in self.degrees_of_freedom:
            scale[1] = design_vector_dict['sy']
        if 'sz' in self.degrees_of_freedom:
            scale[2] = design_vector_dict['sz']

        return translation, rotation, scale




    # @property
    # def design_vector_dict(self) -> dict:
    #
    #     """
    #     Returns a dictionary of the design vector components.
    #
    #     An object's design vector is encoded as a reference position and rotation. Since objects are rigid bodies,
    #     all other geometric properties are a function of the reference position and rotation.
    #     """
    #
    #     design_vector_dict = {}
    #
    #     if 'x' in self.degrees_of_freedom:
    #         design_vector_dict['x'] = self.reference_position[0]
    #     if 'y' in self.degrees_of_freedom:
    #         design_vector_dict['y'] = self.reference_position[1]
    #     if 'z' in self.degrees_of_freedom:
    #         design_vector_dict['z'] = self.reference_position[2]
    #     if 'rx' in self.degrees_of_freedom:
    #         design_vector_dict['rx'] = self.rotation[0]
    #     if 'ry' in self.degrees_of_freedom:
    #         design_vector_dict['ry'] = self.rotation[1]
    #     if 'rz' in self.degrees_of_freedom:
    #         design_vector_dict['rz'] = self.rotation[2]
    #
    #
    #     return design_vector_dict

    # @property
    # def design_vector(self):
    #     design_vector = np.array(list(self.design_vector_dict.values()))
    #     return design_vector


    def calculate_positions(self, design_vector=None, objects_dict=None, transformation_vectors=None):
        """
        Calculates the positions of the object's spheres.
        """

        design_vector_dict = self.decompose_design_vector(design_vector)

        translation, rotation, scaling = self.assemble_transformation_vectors(design_vector_dict)


        # TODO Add reference axes argument
        # Calculate the new positions
        # translation = np.array([[x,y,z]]).T
        # rotation = np.array([[rx,ry,rz]]).T
        # scaling = np.ones((3,1))
        new_positions = affine_transformation(self.reference_position.reshape(-1,1), self.positions.T, translation, rotation, scaling).T

        object_dict = {self.__repr__(): {'type': 'spheres', 'positions': new_positions, 'radii': self.radii}}

        return object_dict


    def set_positions(self,
                      objects_dict: dict = None,
                      design_vector: list = None,
                      translation: list = None,
                      rotation: list = None,
                      scale: list = None):
        """
        Update positions of object spheres given a design vector

        :param objects_dict:
        :param design_vector:
        :return:
        """

        if design_vector is not None:
            objects_dict = self.calculate_positions(design_vector, force_update=True)

        if translation is not None and rotation is not None and scale is not None:
            transformation_vectors = [translation, rotation, scale]
            objects_dict = self.calculate_positions(None, transformation_vectors=transformation_vectors)


        self.positions = objects_dict[self.__repr__()]['positions']
        self.radii     = objects_dict[self.__repr__()]['radii']

    def generate_plot_objects(self):
        objects = []
        colors = []
        for position, radius in zip(self.positions, self.radii):
            objects.append(pv.Sphere(radius=radius, center=position))
            colors.append(self.color)

        return objects, colors


class Interconnect:
    """
    Interconnects are made of one or more non-zero-length segments and connect two components.

    TODO Add a class of components for interconnect dividers (e.g., pipe tee for a three-way split)

    When an interconnect is initialized it does not contain spatial information.

    In the SPI2 class the user specifies which layout generation method to use, and that method tells
    the Interconnect InterconnectNodes what their positions are.

    For now, I will assume that interconnect nodes will start along a straight line between components A
    and B. In the near future they may be included in the layout generation method. The to-do is tracked
    in spatial_configuration.py.

    # placeholder
        component_1 = [i for i in self.components if repr(i) == self.component_1][0]
        component_2 = [i for i in self.components if repr(i) == self.component_2][0]
    """

    def __init__(self,
                 name,
                 component_1_name,
                 component_1_port_name,
                 component_2_name,
                 component_2_port_name,
                 radius,
                 color,
                 number_of_waypoints,
                 degrees_of_freedom):

        self.name = name

        self.component_1_name = component_1_name
        self.component_2_name = component_2_name

        self.component_1_port_name = component_1_port_name
        self.component_2_port_name = component_2_port_name

        self.radius = radius
        self.color = color

        self.number_of_waypoints = number_of_waypoints

        self.movement_class = 'partially dependent'
        self.degrees_of_freedom = degrees_of_freedom

        self.waypoint_positions = np.zeros((self.number_of_waypoints, 3))

        self.dof = 3 * self.number_of_waypoints

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name



    @property
    def design_vector(self):
        return self.waypoint_positions.flatten()

    def calculate_positions(self, design_vector, objects_dict):

        # TODO Add num spheres as argument
        spheres_per_segment = 25

        # TODO Make this work with design vectors of not length 3
        # Reshape the design vector to extract xi, yi, and zi positions
        design_vector = np.array(design_vector)
        design_vector = design_vector.reshape((self.number_of_waypoints, 3))

        object_dict = {}

        # pos_1 = objects_dict[self.object_1]['positions'][0]
        # pos_2 = objects_dict[self.object_2]['positions'][0]

        pos_1_index = objects_dict[self.component_1_name]['port_indices'][self.component_1_port_name]
        pos_2_index = objects_dict[self.component_2_name]['port_indices'][self.component_2_port_name]

        pos_1 = objects_dict[self.component_1_name]['positions'][pos_1_index]
        pos_2 = objects_dict[self.component_2_name]['positions'][pos_2_index]

        node_positions = np.vstack((pos_1, design_vector, pos_2))

        start_arr = node_positions[0:-1]
        stop_arr = node_positions[1:None]

        points = np.linspace(start_arr, stop_arr, spheres_per_segment).reshape(-1, 3)
        radii = np.repeat(self.radius, len(points))

        object_dict[str(self)] = {'type': 'interconnect', 'positions': points, 'radii': radii}

        return object_dict

    def set_positions(self, objects_dict: dict):

        self.positions = objects_dict[str(self)]['positions']

        self.radii = objects_dict[str(self)]['radii']


    def plot(self):
        objects = []
        colors = []


        # Plot spheres at each node
        for position, radius in zip(self.positions, self.radii):
            objects.append(pv.Sphere(radius=radius, center=position))
            colors.append(self.color)

        return objects, colors
