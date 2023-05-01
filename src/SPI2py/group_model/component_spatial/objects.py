"""
TODO Can I remove movement classes if I just use degrees of freedom and reference axes?

"""
from dataclasses import dataclass
import logging

from .object_kinematics import RigidBody

from matplotlib import colors as mcolors

logger = logging.getLogger(__name__)

import numpy as np
from typing import Union

import pyvista as pv

@dataclass
class ComponentState:

    origin: np.ndarray


class Component(RigidBody):
    """
    TODO Update set position to just set the origin.. positions should be a SDF(?)
    """
    def __init__(self,
                 name: str,
                 positions: np.ndarray,
                 radii: np.ndarray,
                 color: str,
                 movement_class: str,
                 reference_axes: str = 'origin',
                 degrees_of_freedom: Union[tuple[str], None] = ('x', 'y', 'z', 'rx', 'ry', 'rz'),
                 ports: Union[None, list[dict]] = None):

        self.name = self._validate_name(name)
        self.type = 'component'
        self.rotation = np.array([0, 0, 0])
        self.ports = self._validate_ports(ports)
        self.port_names = []
        self.port_indices = []

        RigidBody.__init__(self, positions, radii, movement_class, reference_axes, degrees_of_freedom)

        self._valid_colors = {**mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS,
                              **mcolors.XKCD_COLORS}

        self.color = self._validate_colors(color)

        # TODO Make tuple len zero not none
        self.dof = len(self.degrees_of_freedom)

        if self.ports is not None:
            for port in self.ports:
                self.port_names.append(port['name'])
                self.positions = np.vstack((self.positions, port['origin']))
                self.radii = np.concatenate((self.radii, [port['radius']]))
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
        spheres = []
        colors = []
        for i in range(len(self.positions)):
            spheres.append(pv.Sphere(radius=self.radii[i], center=self.positions[i]))
            colors.append(self.color)

        return spheres, colors




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

        self.origins = None
        # self.number_of_bends = number_of_bends

        self.number_of_waypoints = number_of_waypoints
        self.number_of_segments = self.number_of_waypoints + 1

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
        directions = []
        heights = []
        centers = []

        for i in range(self.number_of_waypoints+1):

            direction = self.positions[i + 1] - self.positions[i]

            directions.append(direction)

            height = np.linalg.norm(direction)
            heights.append(height)

            center = self.positions[i] + direction / 2
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

    # def set_origins(self, design_vector, objects_dict):



    def calculate_positions(self, design_vector, objects_dict):
        # TODO Make this work with design vectors of not length 3
        # Reshape the design vector to extract xi, yi, and zi positions
        design_vector = np.array(design_vector)
        design_vector = design_vector.reshape((self.number_of_waypoints, 3))

        object_dict = {}

        pos_1 = objects_dict[self.object_1]['positions'][0]
        pos_2 = objects_dict[self.object_2]['positions'][0]

        positions = np.vstack((pos_1, design_vector, pos_2))
        radii = np.array([self.radius, self.radius])

        object_dict[str(self)] = {'type': 'interconnect', 'positions': positions, 'radii': radii}

        return object_dict

    def set_positions(self, objects_dict: dict):

        self.positions = objects_dict[str(self)]['positions']

        self.radii = np.repeat(self.radius, self.positions.shape[0])


    # @property
    # def edges(self):
    #     return [segment.edge for segment in self.segments]

    def plot(self):
        objects = []
        colors = []

        # Plot spheres at each node
        for i in range(len(self.positions)):
            objects.append(pv.Sphere(radius=self.radii[i], center=self.positions[i]))
            colors.append(self.color)

        directions, heights, centers = self.capsules

        # Plot capsules between nodes
        for direction, height, center in zip(directions, heights, centers):
            objects.append(pv.Cylinder(radius=self.radius, direction=direction, height=height, center=center))
            colors.append(self.color)

        return objects, colors
