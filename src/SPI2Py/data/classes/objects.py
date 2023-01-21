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

import numpy as np
from scipy.spatial.distance import euclidean

from ...analysis.transformations import translate, rotate_about_point


# TODO Add port object

# TODO Add a parent class that sets str, repr, and formats arrays 1D vs 2D, etc.

# class DynamicObject:
#     # TODO Implement a single class to handle how objects move and update positions... let child classes mutate them
#     def __init__(self):
#         self.positions = None
#         self.radii = None
#         self.reference_position = None
#         self.movement = []
#         self.movement_depends_on = []
#


    # def calculate_positions(self, design_vector, positions_dict={}):
    #
    #     # TODO Add functionality to accept positions_dict and work for InterconnectSegments
    #
    #     new_positions = self.positions
    #
    #     if '3D Translation' in self.movement:
    #         new_reference_position = design_vector[0:3]
    #         new_positions = translate(new_positions, self.reference_position, new_reference_position)
    #
    #     if '3D Rotation' in self.movement:
    #         rotation = design_vector[3:None]
    #         new_positions = rotate_about_point(new_positions, rotation)
    #
    #     # TODO Should we
    #     positions_dict[str(self)] = (new_positions, self.radii)
    #
    #     return positions_dict
    #
    # def update_positions(self, positions_dict):
    #     """
    #     Update positions of object spheres given a design vector
    #
    #     :param positions_dict:
    #     :return:
    #     """
    #     self.positions, self.radii = positions_dict[str(self)]

class Component:

    def __init__(self, name, positions, radii, color, 
                    port_names, port_positions, port_radii, 
                    movement=('3D Translation', '3D Rotation')):

        self.name = name
        self.positions = positions
        self.radii = radii
        self.color = color
        
        self.movement = movement

        self.ports = []

        # Initialize the rotation attribute
        self.rotation = np.array([0, 0, 0])

        # Ports
        self.port_indices = [] # Tracks the index wihtin positions and radii that the port is located
        self.port_names = port_names
        self.port_positions = port_positions
        self.port_radii = port_radii

        if port_positions is not None and port_radii is not None:

            for position, radius in zip(self.port_positions, self.port_radii):
                
                self.positions = np.vstack((self.positions, position)) # TODO Make this cleaner
                self.radii = np.append(self.radii, radius) # TODO Make this cleaner

                port_index = self.positions.shape[0]-1 # Nabs the final row index
                self.port_indices.append(port_index)


    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

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

    def calculate_positions(self, design_vector, positions_dict={}):

        # TODO Add functionality to accept positions_dict and work for InterconnectSegments

        new_positions = self.positions

        if '3D Translation' in self.movement:
            new_reference_position = design_vector[0:3]
            new_positions = translate(new_positions, self.reference_position, new_reference_position)

        if '3D Rotation' in self.movement:
            rotation = design_vector[3:None]
            new_positions = rotate_about_point(new_positions, rotation)

        # TODO Should we
        positions_dict[str(self)] = (new_positions, self.radii)

        return positions_dict

    def update_positions(self, positions_dict):
        """
        Update positions of object spheres given a design vector

        :param positions_dict:
        :return:
        """
        self.positions, self.radii = positions_dict[str(self)]


class InterconnectNode:
    def __init__(self, node, radius, color, movement=['3D Translation']):
        self.name = node
        self.node = node
        self.radius = radius
        self.color = color

        # delete this redundant (used for plotting)
        self.radii = np.array([radius])
        # TODO Sort out None value vs dummy values
        self.positions = np.array([[0., 0., 0.]])  # Initialize a dummy value
        self.movement = movement

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @property
    def reference_position(self):
        return self.positions

    @property
    def design_vector(self):
        return self.positions.flatten()

    def calculate_positions(self, design_vector, positions_dict={}):

        # TODO Add functionality to accept positions_dict and work for InterconnectSegments

        new_positions = self.positions

        if '3D Translation' in self.movement:
            new_reference_position = design_vector[0:3]
            new_positions = translate(new_positions, self.reference_position, new_reference_position)

        if '3D Rotation' in self.movement:
            rotation = design_vector[3:None]
            new_positions = rotate_about_point(new_positions, rotation)

        # TODO Should we
        positions_dict[str(self)] = (new_positions, self.radii)

        return positions_dict

    def update_positions(self, positions_dict):
        """
        Update positions of object spheres given a design vector

        :param positions_dict:
        :return:
        """
        self.positions, self.radii = positions_dict[str(self)]


class InterconnectEdge:
    def __init__(self, name, object_1, object_2, radius, color):
        self.name = name
        self.object_1 = object_1
        self.object_2 = object_2

        self.radius = radius
        self.color = color

        self.movement = []

        # Create edge tuple for NetworkX graphs
        self.edge = (self.object_1, self.object_2)

        # Placeholder for plot test functionality, random positions
        self.positions = None
        self.radii = None

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def calculate_positions(self, positions_dict):
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

    def update_positions(self, positions_dict):
        self.positions = self.calculate_positions(positions_dict)[str(self)][0] # index zero for tuple

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

    def __init__(self, name, component_1, component_1_port, component_2, component_2_port, radius, color):
        self.name = name

        self.component_1 = component_1
        self.component_2 = component_2

        self.component_1_port = component_1_port
        self.component_2_port = component_2_port

        self.radius = radius
        self.color = color

        # Per configuration file
        # TODO connect this setting to the config file
        self.number_of_nodes = 1
        self.number_of_edges = self.number_of_nodes + 1

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
        nodes = [self.component_1]
        node_names = [self.component_1]

        # Add the interconnect nodes
        for i in range(self.number_of_nodes):
            # Each node should have unique identifier
            node_prefix = self.component_1 + '-' + self.component_2 + '_node_'
            node = node_prefix + str(i)

            interconnect_node = InterconnectNode(node, self.radius, self.color)
            nodes.append(interconnect_node)
            node_names.append(str(interconnect_node))


        # Add component 2
        nodes.append(self.component_2)
        node_names.append(self.component_2)

        return nodes, node_names

    def create_node_pairs(self):

        node_pairs = [(self.node_names[i], self.node_names[i + 1]) for i in range(len(self.node_names) - 1)]

        return node_pairs

    def create_segments(self):

        segments = []

        # TODO Implement
        # TODO Check...
        i=0
        for object_1, object_2 in self.node_pairs:

            name = self.component_1 + '-' + self.component_2 + '_edge_' + str(i)
            i+= 1
            segments.append(InterconnectEdge(name, object_1, object_2, self.radius, self.color))

        return segments

    @property
    def edges(self):
        return [segment.edge for segment in self.segments]

    def calculate_positions(self, positions_dict):
        pass

    def update_positions(self, positions_dict):
        pass


class Structure:
    def __init__(self, name, positions, radii, color):
        self.name = name
        self.positions = positions
        self.radii = radii
        self.color = color

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name