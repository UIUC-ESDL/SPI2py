"""
TODO Can I remove movement classes if I just use degrees of freedom and reference axes?

"""
import logging

from SPI2py.group_model.component_geometry.finite_sphere_method import read_mdbd_file, generate_rectangular_prisms
from SPI2py.group_model.component_spatial.spatial_transformations import affine_transformation

from matplotlib import colors as mcolors

logger = logging.getLogger(__name__)

import jax.numpy as np
from typing import Union

import pyvista as pv


class Component:
    """
    TODO Update set position to just set the origin.. positions should be a SDF(?)
    """
    def __init__(self,
                 name: str,
                 color: str=None,
                 degrees_of_freedom: tuple[str] = ('x', 'y', 'z', 'rx', 'ry', 'rz'),
                 shapes=None,
                 ports: Union[None, list[dict]] = None,
                 cad_file: str = None,
                 mdbd_filepath: str = None):

        self.name = self._validate_name(name)



        self.type = 'component'
        self.rotation = np.array([0, 0, 0])
        self.ports = self._validate_ports(ports)
        self.port_names = []
        self.port_indices = []

        self.shapes = shapes
        self.cad_file = cad_file
        self.mdbd_filepath = mdbd_filepath

        if mdbd_filepath is not None:
            positions, radii = read_mdbd_file(mdbd_filepath)
        else:

            origins = []
            dimensions = []
            for shape in shapes:
                origins.append(shape['origin'])
                dimensions.append(shape['dimensions'])

            positions, radii = generate_rectangular_prisms(origins, dimensions)


        self.positions = positions
        self.radii = radii

        # TODO Remove rotation...
        self.rotation = np.zeros(3)
        self.degrees_of_freedom = degrees_of_freedom

        self._valid_colors = {**mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS,
                              **mcolors.XKCD_COLORS}

        self.color = self._validate_colors(color)

        # TODO Make tuple len zero not none
        self.dof = len(self.degrees_of_freedom)

        if self.ports is not None:
            for port in self.ports:
                self.port_names.append(port['name'])
                self.positions = np.vstack((self.positions, port['origin']))
                self.radii = np.concatenate((self.radii, np.array([port['radius']])))
                self.port_indices.append(len(self.positions) - 1)






    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str):
            raise TypeError('Name must be a string not %s.' % type(name))
        return name

    def port_index(self, port_name: str) -> int:
        if port_name not in self.port_names:
            raise ValueError('Port name %s is not in %s.' % (port_name, self.name))
        return self.port_indices[self.port_names.index(port_name)]

    def port_position(self, port_name: str) -> np.ndarray:
        if port_name not in self.port_names:
            raise ValueError('Port name %s is not in %s.' % (port_name, self.name))
        return self.positions[self.port_indices[self.port_names.index(port_name)]]

    def _validate_color(self, color):

        if not isinstance(color, str):
            raise ValueError('Color must be a string.')

        if color not in self._valid_colors:
            raise ValueError('Color not recognized. For a list of valid colors inspect the attribute '
                             'self._valid_colors.keys().')

    def _validate_colors(self, colors):

        if colors is None:
            raise ValueError(f'Color has not been set for {self.__repr__()}')

        if isinstance(colors, list):

            if len(colors) == 1:
                self._validate_color(colors)

            if len(colors) > 1:
                for color in colors:
                    self._validate_color(color)

        elif isinstance(colors, str):
            self._validate_color(colors)
        else:
            raise ValueError('Colors must be a list or string.')

        return colors

    def _validate_ports(self, ports):

        if ports is None:
            return ports

        if not isinstance(ports, list):
            raise TypeError('Ports must be a list.')

        # for port in ports:
        #     if not isinstance(port, Port):
        #         raise TypeError('Ports must be a list of Port objects.')

        return ports

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def plot(self):
        objects = []
        colors = []
        for i in range(len(self.positions)):
            objects.append(pv.Sphere(radius=self.radii[i], center=self.positions[i]))
            colors.append(self.color)

        if self.cad_file is not None:
            mesh = pv.read(self.cad_file)
            mesh = mesh.scale(1/50)
            objects.append(mesh)
            colors.append(self.color)

        return objects, colors



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
        translation = np.array([[x,y,z]]).T
        rotation = np.array([[rx,ry,rz]]).T
        scaling = np.ones((3,1))
        new_positions = affine_transformation(self.reference_position.reshape(-1,1), self.positions.T, translation, rotation, scaling).T

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

        self.object_1 = self.component_1_name + '_' + self.component_1_port_name
        self.object_2 = self.component_2_name + '_' + self.component_2_port_name

        self.radius = radius
        self.color = color

        self.number_of_waypoints = number_of_waypoints

        # Create InterconnectNode objects
        self.nodes = self.create_nodes()
        self.interconnect_nodes = self.nodes[1:-1]  # trims off components 1 and 2

        # Create InterconnectSegment objects
        self.node_pairs = self.create_node_pairs()

        self.movement_class = 'partially dependent'
        self.degrees_of_freedom = degrees_of_freedom

        self.waypoint_positions = np.zeros((self.number_of_waypoints, 3))

        self.dof = 3 * self.number_of_waypoints

    @property
    def capsules(self):

        a, b = self.positions

        directions = []
        heights = []
        centers = []

        for i in range(self.number_of_waypoints+1):

            direction = b[i] - a[i]

            directions.append(direction)

            height = np.linalg.norm(direction)
            heights.append(height)

            center = a[i] + direction / 2
            centers.append(center)

        return directions, heights, centers

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def design_vector(self):
        return self.waypoint_positions.flatten()

    def create_nodes(self):

        # Create the nodes list and add component 1
        nodes = [self.object_1]

        # Add the interconnect nodes
        for i in range(self.number_of_waypoints):
            # Each node should have unique identifier
            node_name = self.name + '_node_' + str(i)
            nodes.append(node_name)

        # Add component 2
        nodes.append(self.object_2)

        return nodes

    def create_node_pairs(self):

        node_pairs = [(self.nodes[i], self.nodes[i + 1]) for i in range(len(self.nodes) - 1)]

        return node_pairs


    def calculate_positions(self, design_vector, objects_dict):

        # TODO Add num spheres as argument
        spheres_per_segment = 25

        # TODO Make this work with design vectors of not length 3
        # Reshape the design vector to extract xi, yi, and zi positions
        design_vector = np.array(design_vector)
        design_vector = design_vector.reshape((self.number_of_waypoints, 3))

        object_dict = {}

        pos_1 = objects_dict[self.object_1]['positions'][0]
        pos_2 = objects_dict[self.object_2]['positions'][0]

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

    # @property
    # def edges(self):
    #     return [segment.edge for segment in self.segments]

    def plot(self):
        objects = []
        colors = []

        a, b = self.positions

        # Plot spheres at each node
        for i in range(len(self.positions)):
            objects.append(pv.Sphere(radius=self.radii[i], center=a[i]))
            colors.append(self.color)

        directions, heights, centers = self.capsules

        # Plot capsules between nodes
        for direction, height, center in zip(directions, heights, centers):
            objects.append(pv.Cylinder(radius=self.radius, direction=direction, height=height, center=center))
            colors.append(self.color)

        return objects, colors
