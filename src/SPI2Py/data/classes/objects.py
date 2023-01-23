"""

    @property
    def positions(self):

        if self.positions is None:
            raise ValueError('Positions have not been set.')

        if len(self.positions.shape) == 1:
            raise ValueError('Positions are not 2D.')

        if self.positions.shape[1] != 3:
            raise ValueError('Positions must be 3D.')

        return self.positions

    @property
    def radii(self):

        if self.radii is None:
            raise ValueError('Radii have not been set.')

        if len(self.radii.shape[0]) != len(self.positions.shape[0]):
            raise ValueError('Radii must be the same length as positions.')

        return self.radii

"""
from typing import Union
import numpy as np
import warnings

from scipy.spatial.distance import euclidean

from .input_validation import InputValidation
from ...analysis.transformations import translate, rotate_about_point


# TODO Add port object

# TODO Add a parent class that sets str, repr, and formats arrays 1D vs 2D, etc.

class Object(InputValidation):
    # TODO Implement a single class to handle how objects move and update positions... let child classes mutate them
    def __init__(self,
                 name: str,
                 positions: np.ndarray,
                 rotation: np.ndarray,
                 radii: np.ndarray,
                 color: Union[str, list[str]],
                 degrees_of_freedom: tuple[str] = ('x', 'y', 'z', 'rx', 'ry', 'rz'),
                 reference_object: Union[None, tuple[str]] = None):

        super(Object, self).__init__(name, positions, rotation, radii, color)

        self.degrees_of_freedom = degrees_of_freedom
        self.reference_object = reference_object

        self.rotation = np.zeros(3)

        # self.reference_position = None

        self.three_d_translation = all([dof in self.degrees_of_freedom for dof in ['x', 'y', 'z']])
        self.three_d_rotation = all([dof in self.degrees_of_freedom for dof in ['rx', 'ry', 'rz']])

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def reference_position(self):
        return self.positions[0]

    # @property
    # def design_vector(self):
    #     """
    #     TODO Provide a method to reduce the design vector (e.g., not translation along z axis)
    #     :return:
    #     """
    #
    #     if self.three_d_translation is True and self.three_d_rotation is not True:
    #         design_vector = self.reference_position
    #
    #     elif self.three_d_translation is True and self.three_d_rotation is True:
    #         design_vector = np.concatenate((self.reference_position, self.rotation))
    #
    #     else:
    #         warnings.warn('This object is fixed')
    #         design_vector = None
    #
    #     return design_vector

    def calculate_independent_positions(self,
                                        design_vector: np.ndarray,
                                        positions_dict: Union[None, dict] = None) -> dict:

        """

        """

        new_positions = np.copy(self.positions)

        if self.three_d_translation is True:
            new_reference_position = design_vector[0:3]
            new_positions = translate(new_positions, self.reference_position, new_reference_position)

        if self.three_d_rotation is True:
            rotation = design_vector[3:None]
            new_positions = rotate_about_point(new_positions, rotation)

        # TODO Check for constrained movement cases

        positions_dict[str(self)] = (new_positions, self.radii)

        return positions_dict

    def calculate_dependent_positions(self,
                                      design_vector: np.ndarray,
                                      positions_dict: Union[None, dict] = None) -> dict:
        """

        """
        # If fixed
        if self.reference_object is None:
            raise ValueError('This object is fixed..? or just static?')

        # If straight line

        return positions_dict
    #
    # def calculate_positions(self,
    #                         design_vector: np.ndarray,
    #                         positions_dict: Union[None, dict] = None) -> dict:
    #
    #     """
    #
    #     """
    #
    #     if positions_dict is None:
    #         positions_dict = {}
    #
    #     if self.reference_object is None:
    #
    #         # Calculate the independent positions
    #         positions_dict = self.calculate_independent_positions(design_vector, positions_dict)
    #
    #     elif self.reference_object is not None:
    #
    #         # Calculate the dependent positions
    #         positions_dict = self.calculate_dependent_positions(design_vector, positions_dict)
    #
    #     return positions_dict


    def set_positions(self, positions_dict: dict) -> dict:
        """
        Update positions of object spheres given a design vector

        :param positions_dict:
        :return:
        """

        self.positions, self.radii = positions_dict[self.name]


class Port(InputValidation):

    def __init__(self, component_name, port_name, color, reference_point_offset, radius):

        self.component_name = component_name
        self.port_name = port_name

        self.name = self.component_name + '_' + self.port_name + '_port'

        self.color = color


        self.reference_point_offset = self._validate_positions(reference_point_offset)

        # Initialize position as the offset from the reference point
        self.positions = self.reference_point_offset

        self.radius = self._validate_radii(radius)


    def __repr__(self):
        return self.component_name + '_' + self.port_name + '_port'

    def __str__(self):
        return self.component_name + '_' + self.port_name + '_port'

    def calculate_positions(self, positions_dict):

        # Get the reference point
        reference_point = positions_dict[self.component_name][0][0]

        # Calculate the port position
        port_position = reference_point + self.reference_point_offset

        # Add the port position to the positions dictionary
        positions_dict[str(self)] = (port_position, self.radius)

        return positions_dict

    def set_positions(self, positions_dict: dict) -> dict:
        """
        Update positions of object spheres given a design vector

        :param positions_dict:
        :return:
        """

        self.positions, self.radii = positions_dict[self.name]


class Component(Object):

    def __init__(self,
                 name: str,
                 positions: np.ndarray,
                 rotation: np.ndarray,
                 radii: np.ndarray,
                 color: Union[str, list[str]],
                 degrees_of_freedom: tuple[str] = ('x', 'y', 'z', 'rx', 'ry', 'rz'),
                 reference_object: Union[None, tuple[str]] = None):

        super(Component, self).__init__(name, positions, rotation, radii, color)

        self.degrees_of_freedom = degrees_of_freedom
        self.reference_object = reference_object

        # Initialize the rotation attribute
        # self.rotation = np.array([0, 0, 0])

    @property
    def design_vector(self):

        design_vector = np.concatenate((self.reference_position, self.rotation))

        return design_vector

    def calculate_positions(self,
                            design_vector,
                            positions_dict={}):

        positions_dict = self.calculate_independent_positions(design_vector, positions_dict)

        return positions_dict


class InterconnectNode(Object):
    def __init__(self,
                 node,
                 radius,
                 color,
                 movement=['3D Translation']):

        self.name = node
        self.node = node
        self.radius = radius
        self.color = color

        # delete this redundant (used for plotting)
        self.radii = np.array([radius])
        # TODO Sort out None value vs dummy values
        self.positions = np.array([[0., 0., 0.]])  # Initialize a dummy value
        self.movement = movement

    @property
    def reference_position(self):
        return self.positions

    @property
    def design_vector(self):
        return self.positions.flatten()

    def calculate_positions(self,
                            design_vector,
                            positions_dict={}):

        # TODO Add functionality to accept positions_dict and work for InterconnectSegments

        new_positions = self.positions


        new_reference_position = design_vector[0:3]
        new_positions = translate(new_positions, self.reference_position, new_reference_position)

        # TODO Should we
        positions_dict[str(self)] = (new_positions, self.radii)

        return positions_dict


class InterconnectEdge(Object):
    def __init__(self,
                 name,
                 object_1,
                 object_2,
                 radius,
                 color):

        self.name = name
        self.object_1 = object_1
        self.object_2 = object_2

        self.radius = radius
        self.color = color

        self.movement = []

        # Create edge tuple for NetworkX graphs
        self.edge = (self.object_1, self.object_2)

        # Placeholder for plot test functionality, random positions
        # self.positions = None
        self.positions = np.empty((0, 3))
        self.radii = None

    def calculate_positions(self,
                            positions_dict: dict)->dict:
        # TODO revise logic for getting the reference point instead of object's first sphere
        # Address varying number of spheres

        # TODO FIX THIS?
        # Design vector not used
        pos_1 = positions_dict[self.object_1][0][0]
        pos_2 = positions_dict[self.object_2][0][0]

        dist = euclidean(pos_1, pos_2)

        # We don't want zero-length interconnects or interconnect segments--they cause problems!
        num_spheres = int(dist / (self.radius * 2))
        if num_spheres == 0:
            num_spheres = 1

        positions = np.linspace(pos_1, pos_2, num_spheres)
        radii = np.repeat(self.radius, num_spheres)

        # TODO Change positions_dict to include kwarg and return addition?
        return {str(self): (positions, radii)}

    def set_positions(self,
                      positions_dict: dict) -> dict:

        # self.positions, self.radii = positions_dict[self.name]

        self.positions = self.calculate_positions(positions_dict)[str(self)][0]  # index zero for tuple

        # TODO Separate this into a different function?
        self.radii = np.repeat(self.radius, self.positions.shape[0])


class Interconnect(InterconnectNode, InterconnectEdge):
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
                 component_1,
                 component_1_port,
                 component_2,
                 component_2_port,
                 radius,
                 color,
                 number_of_bends):

        self.name = name

        self.component_1 = component_1
        self.component_2 = component_2

        self.component_1_port = component_1_port
        self.component_2_port = component_2_port

        self.object_1 = self.component_1 + '_' + self.component_1_port + '_port'
        self.object_2 = self.component_2 + '_' + self.component_2_port + '_port'

        self.radius = radius
        self.color = color

        # self.number_of_bends = number_of_bends

        # Per configuration file
        # TODO connect this setting to the config file
        self.number_of_bends = number_of_bends
        self.number_of_edges = self.number_of_bends + 1

        # Create InterconnectNode objects
        self.nodes, self.node_names = self.create_nodes()
        self.interconnect_nodes = self.nodes[1:-1]  # trims off components 1 and 2

        # Create InterconnectSegment objects
        self.node_pairs = self.create_node_pairs()
        self.segments = self.create_segments()

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def create_nodes(self):
        """
        Consideration: if I include the component nodes then... ?

        :return:
        """
        # TODO Make sure nodes are 2D and not 1D!

        # Create the nodes list and add component 1
        nodes = [self.object_1]
        node_names = [self.object_1]

        # Add the interconnect nodes
        for i in range(self.number_of_bends):
            # Each node should have unique identifier
            node_prefix = self.component_1 + '-' + self.component_1_port + '_' + self.component_2 + '-' + self.component_2_port + '_node_'
            node = node_prefix + str(i)

            interconnect_node = InterconnectNode(node, self.radius, self.color)
            nodes.append(interconnect_node)
            node_names.append(str(interconnect_node))

        # Add component 2
        nodes.append(self.object_2)
        node_names.append(self.object_2)

        return nodes, node_names

    def create_node_pairs(self):

        #

        node_pairs = [(self.node_names[i], self.node_names[i + 1]) for i in range(len(self.node_names) - 1)]

        return node_pairs

    def create_segments(self):

        segments = []

        # TODO Implement
        # TODO Check...
        i = 0
        for object_1, object_2 in self.node_pairs:
            name = self.component_1 + '-' + self.component_2 + '_edge_' + str(i)
            i += 1
            segments.append(InterconnectEdge(name, object_1, object_2, self.radius, self.color))

        return segments

    @property
    def edges(self):
        return [segment.edge for segment in self.segments]



class Structure(Object):
    def __init__(self,
                 name,
                 positions,
                 rotation,
                 radii,
                 color):

        super(Structure, self).__init__(name, positions, rotation, radii, color)

