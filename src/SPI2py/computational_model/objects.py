"""
TODO Can I remove movement classes if I just use degrees of freedom and reference axes?

"""

import logging

from scipy.spatial.distance import euclidean

from .kinematics.transformations import translate, rotate, rigid_transformation
from .kinematics.object_kinematics import RigidBody

from matplotlib import colors as mcolors

logger = logging.getLogger(__name__)

import numpy as np
from typing import Union


class Component(RigidBody):

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
        self.type= 'component'
        self.rotation = np.array([0,0,0])
        self.ports = self._validate_ports(ports)
        self.port_names = []
        self.port_indices = []

        RigidBody.__init__(self, positions, radii, movement_class, reference_axes, degrees_of_freedom)

        self._valid_colors = {**mcolors.BASE_COLORS, **mcolors.TABLEAU_COLORS, **mcolors.CSS4_COLORS,
                              **mcolors.XKCD_COLORS}

        self.color = self._validate_colors(color)

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







class InterconnectWaypoint(Component):
    def __init__(self,
                 node,
                 radius,
                 color,
                 degrees_of_freedom: Union[tuple[str], None] = ('x', 'y', 'z'),
                 constraints: Union[None, tuple[str]] = None):
        self.name = node
        self.type = 'component'
        self.node = node
        self.radius = radius
        self.color = color

        # TODO Temp
        self.ports = None

        # delete this redundant (used for plotting)
        self.radii = np.array([radius])
        # TODO Sort out None value vs dummy values
        self.positions = np.array([[0., 0., 0.]])  # Initialize a dummy value
        self.movement_class = 'independent'
        self.degrees_of_freedom = degrees_of_freedom
        self.reference_objects = constraints


"""
TODO
Interconnect should be a single object, not subclasses
-represent as bar
-calculate interference as bars
-plot as bars

"""


class InterconnectEdge(Component):
    def __init__(self,
                 name,
                 object_1,
                 object_2,
                 radius,
                 color,
                 degrees_of_freedom: Union[tuple[str], None] = None,
                 constraints: Union[None, tuple[str]] = None):

        self.name = name
        self.type = 'edge'
        self.object_1 = object_1
        self.object_2 = object_2

        self.radius = radius
        self.color = color

        self.degrees_of_freedom = degrees_of_freedom
        self.reference_objects = constraints

        self.movement_class = 'fully dependent'

    def calculate_positions(self, _, objects_dict):

        object_dict = {}

        pos_1 = objects_dict[self.object_1]['positions'][0]
        pos_2 = objects_dict[self.object_2]['positions'][0]

        positions = np.vstack((pos_1, pos_2))
        radii = np.array([self.radius, self.radius])

        object_dict[str(self)] = {'type': 'interconnect', 'positions': positions, 'radii': radii}

        return object_dict

    def set_positions(self, objects_dict: dict):

        object_dict =  self.calculate_positions([], objects_dict)
        self.positions =object_dict[str(self)]['positions']

        self.radii = np.repeat(self.radius, self.positions.shape[0])


class Interconnect(InterconnectWaypoint, InterconnectEdge):
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
                 number_of_bends):

        self.name = name

        self.component_1_name = component_1_name
        self.component_2_name = component_2_name

        self.component_1_port_name = component_1_port_name
        self.component_2_port_name = component_2_port_name

        self.object_1 = self.component_1_name + '_' + self.component_1_port_name
        self.object_2 = self.component_2_name + '_' + self.component_2_port_name

        self.radius = radius
        self.color = color


        # self.number_of_bends = number_of_bends

        self.number_of_bends = number_of_bends
        self.number_of_edges = self.number_of_bends + 1

        # Create InterconnectNode objects
        self.nodes, self.node_names = self.create_nodes()
        self.interconnect_nodes = self.nodes[1:-1]  # trims off components 1 and 2

        # Create InterconnectSegment objects
        self.node_pairs = self.create_node_pairs()
        self.segments = self.create_segments()

    def create_nodes(self):
        """
        Consideration: if I include the component nodes then... ?

        TODO replace node names w/ objects!

        :return:
        """
        # TODO Make sure nodes are 2D and not 1D!

        # Create the nodes list and add component 1
        nodes = [self.object_1]
        node_names = [self.object_1]

        # Add the interconnect nodes
        for i in range(self.number_of_bends):
            # Each node should have unique identifier
            node_name = self.name + '_node_' + str(i)

            interconnect_node = InterconnectWaypoint(node_name, self.radius, self.color)
            nodes.append(interconnect_node)
            node_names.append(node_name)

        # Add component 2
        nodes.append(self.object_2)
        node_names.append(self.object_2)

        return nodes, node_names

    def create_node_pairs(self):

        node_pairs = [(self.node_names[i], self.node_names[i + 1]) for i in range(len(self.node_names) - 1)]

        return node_pairs

    def create_segments(self):

        segments = []

        i = 0
        for object_1, object_2 in self.node_pairs:
            name = self.component_1_name + '-' + self.component_2_name + '_edge_' + str(i)
            segments.append(InterconnectEdge(name, object_1, object_2, self.radius, self.color))
            i += 1

        return segments

    # @property
    # def edges(self):
    #     return [segment.edge for segment in self.segments]