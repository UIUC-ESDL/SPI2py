"""Module...
Docstring

TODO Fill out all the blank functions and write tests for them...
TODO Look into replacing get/set methods with appropriate decorators...


"""

import numpy as np
from scipy.spatial.distance import euclidean

from src.SPI2Py.analysis.spatial_calculations.transformations import translate, rotate_about_point


class MovableObject:
    # TODO Implement a single class to handle how objects move and update positions... let child classes mutate them
    def __init__(self):
        self.positions = None
        self.reference_position = None
        self.movement = []
        self.movement_depends_on = []

    def calculate_positions(self, design_vector, positions_dict={}):

        # TODO Add functionality to accept positions_dict and work for InterconnectSegments

        new_positions = self.positions

        if '3D Translation' in self.movement:
            new_reference_position = design_vector[0:3]
            new_positions = translate(new_positions, self.reference_position, new_reference_position)

        if '3D Rotation' in self.movement:
            rotation = design_vector[3:None]
            new_positions = rotate_about_point(new_positions, rotation)

        positions_dict[self] = new_positions

        return positions_dict

    def update_positions(self, positions_dict):
        """
        Update positions of object spheres given a design vector

        :param positions_dict:
        :return:
        """
        self.positions = positions_dict[self]


# TODO Add port object
class Component(MovableObject):

    def __init__(self, positions, radii, color, node, name, movement=['3D Translation', '3D Rotation']):

        self.positions = positions
        self.radii = radii
        self.color = color
        self.node = node
        self.name = name
        self.movement = movement

        # Initialize the rotation attribute
        self.rotation = np.array([0, 0, 0])

    @property
    def reference_position(self):
        return self.positions[0]

    @property
    def design_vector(self):
        """
        TODO Provide a method to reduce the design vector (e.g., not translation along z axis)
        :return:
        """
        return np.concatenate((self.reference_position, self.rotation))


class InterconnectNode(MovableObject):
    def __init__(self, node, radius, color, movement=['3D Translation']):
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



class InterconnectSegment(MovableObject):
    def __init__(self, object_1, object_2, diameter, color):
        self.object_1 = object_1
        self.object_2 = object_2

        self.diameter = diameter
        self.radius = diameter / 2
        self.color = color

        # Create edge tuple for NetworkX graphs
        self.edge = (self.object_1.node, self.object_2.node)

        # Placeholder for plot test functionality, random positions
        self.positions = None
        self.radii = None

    def calculate_positions(self, positions_dict):
        # TODO revise logic for getting the reference point instead of object's first sphere
        # Address varying number of spheres

        # Design vector not used
        pos_1 = positions_dict[self.object_1][0]
        pos_2 = positions_dict[self.object_2][0]


        dist = euclidean(pos_1, pos_2)

        # We don't want zero-length interconnects or interconnect segments--they cause problems!
        num_spheres = int(dist / self.diameter)
        if num_spheres == 0:
            num_spheres = 1

        positions = np.linspace(pos_1, pos_2, num_spheres)

        return {self: positions}

    def update_positions(self, positions_dict):
        self.positions = self.calculate_positions(positions_dict)[self]

        # TODO Separate this into a different function?
        self.radii = np.repeat(self.radius, self.positions.shape[0])


class Interconnect(InterconnectNode, InterconnectSegment):
    """
    Interconnects are made of one or more non-zero-length segments and connect two components.

    TODO Add a class of components for interconnect dividers (e.g., pipe tee for a three-way split)

    When an interconnect is initialized it does not contain spatial information.

    In the SPI2 class the user specifies which layout generation method to use, and that method tells
    the Interconnect InterconnectNodes what their positions are.

    For now, I will assume that interconnect nodes will start along a straight line between components A
    and B. In the near future they may be included in the layout generation method. The to-do is tracked
    in organizational.py.
    """

    def __init__(self, component_1, component_2, diameter, color):
        self.component_1 = component_1
        self.component_2 = component_2

        self.diameter = diameter
        self.radius = diameter / 2
        self.color = color

        # Per configuration file
        # TODO connect this setting to the config file
        self.number_of_nodes = 1
        self.number_of_segments = self.number_of_nodes + 1

        # Create InterconnectNode objects
        self.nodes = self.create_nodes()
        self.interconnect_nodes = self.nodes[1:-1] # trims off components 1 and 2

        # Create InterconnectSegment objects
        self.node_pairs = self.create_node_pairs()
        self.segments = self.create_segments()

    def create_nodes(self):
        """
        Consideration: if I include the component nodes then... ?

        :return:
        """
        # TODO Make sure nodes are 2D and not 1D!

        # Create the nodes list and add component 1
        nodes = [self.component_1]

        # Add the interconnect nodes
        for i in range(self.number_of_nodes):
            # Each node should have unique identifier
            node_prefix = str(self.component_1.node) + '-' + str(self.component_2.node) + '_'
            node = node_prefix + str(i)

            nodes.append(InterconnectNode(node,self.diameter/2, self.color))

        # Add component 2
        nodes.append(self.component_2)

        return nodes

    def create_node_pairs(self):


        node_pairs = [(self.nodes[i], self.nodes[i + 1]) for i in range(len(self.nodes) - 1)]

        return node_pairs

    def create_segments(self):

        segments = []

        # TODO Implement
        # TODO Check...
        for object_1, object_2 in self.node_pairs:
            segments.append(InterconnectSegment(object_1, object_2, self.diameter, self.color))

        return segments


    @property
    def edges(self):
        return [segment.edge for segment in self.segments]

    def calculate_positions(self, positions_dict):
        pass

    def update_positions(self, positions_dict):
        pass


class Structure:
    def __init__(self, positions, radii, color, name):
        self.positions = positions
        self.radii = radii
        self.color = color
        self.name = name


class Volume:
    """
    A class that captures the 3D space that we place objects in and optimize
    """
    pass


class Volumes(Volume):
    """
    A class that combines contiguous volumes together.
    """
    pass
