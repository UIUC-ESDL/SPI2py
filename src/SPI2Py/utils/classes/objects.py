"""Module...
Docstring

TODO Fill out all the blank functions and write tests for them...
TODO Look into replacing get/set methods with appropriate decorators...


"""

import numpy as np
from scipy.spatial.distance import euclidean

from ..spatial_calculations.transformations import translate, rotate_about_point


class Component:

    def __init__(self, positions, radii, color, node, name, constraints=None):

        self.positions = positions
        self.radii = radii
        self.color = color
        self.node = node
        self.name = name
        self.constraints = constraints

        # Initialize the rotation attribute
        self.rotation = np.array([0, 0, 0])

    @property
    def reference_position(self):
        return self.positions[0]

    def get_positions(self, design_vector):
        """
        Update positions of object spheres given a design vector

        Constraint refers to how if we constrain the object, it will have a different size design vector

        :param design_vector:
        :return:
        """

        positions_dict = {}

        # Assumes (1,6) design vector... will need to expand in future
        if self.constraints is None:

            new_reference_position = design_vector[0:3]
            new_rotation = design_vector[3:None]

            delta_position = new_reference_position - self.reference_position
            delta_rotation = new_rotation - self.rotation

            translated_positions = translate(self.positions, delta_position)

            rotated_translated_positions = rotate_about_point(translated_positions, delta_rotation)

        else:
            print('Placeholder')

        positions_dict[self] = rotated_translated_positions

        return positions_dict

    def update_positions(self, design_vector):
        """
        Update positions of object spheres given a design vector

        Constraint refers to how if we constrain the object, it will have a different size design vector

        :param design_vector:
        :return:
        """

        # Assumes (1,6) design vector... will need to expand in future
        if self.constraints is None:

            new_reference_position = design_vector[0:3]
            new_rotation = design_vector[3:None]

            delta_position = new_reference_position - self.reference_position
            delta_rotation = new_rotation - self.rotation

            translated_positions = translate(self.positions, delta_position)

            rotated_translated_positions = rotate_about_point(translated_positions, delta_rotation)

            # Update values
            self.positions = rotated_translated_positions
            self.rotation = new_rotation

        else:
            print('Placeholder')

    @property
    def design_vector(self):
        return np.concatenate((self.reference_position, self.rotation))


class InterconnectNode:
    def __init__(self):
        self.node = None
        self.position = None

    @property
    def reference_position(self): return self.position

    @property
    def design_vector(self): return self.position

    # def update_position(self, design_vector, constraint=None):
    #
    #     if constraint is None:
    #
    #         new_reference_position = design_vector[0:3]
    #
    #         delta_position = new_reference_position - self.reference_position
    #
    #         translated_positions = translate(self.positions, delta_position)
    #
    #         # Update values
    #         self.position = translated_positions
    #
    #     else:
    #         print('Placeholder')



class InterconnectSegment:
    def __init__(self, object_1, object_2, diameter, color):
        self.object_1 = object_1
        self.object_2 = object_2

        self.diameter = diameter
        self.radius = diameter / 2
        self.color = color

        # Create edge tuple for NetworkX graphs
        self.edge = (self.object_1.node, self.object_2.node)

        # Placeholder for plot test functionality, random positions
        self.positions, self.radii = self.set_positions()

        # Temporary
        self.num_spheres = len(self.radii)

    def set_positions(self):
        pos_1 = self.object_1.reference_position
        pos_2 = self.object_2.reference_position

        dist = euclidean(pos_1, pos_2)

        num_spheres = int(dist / self.diameter)

        # We don't want zero-length interconnects or interconnect segments--they cause problems!
        if num_spheres == 0:
            num_spheres = 1

        positions = np.linspace(pos_1, pos_2, num_spheres)

        # Temporary value

        radii = np.repeat(self.radius, positions.shape[0])

        return positions, radii

    def calculate_positions(self, positions_dict):
        # TODO revise logic for getting the reference point
        # Address varying number of spheres

        pos_1 = positions_dict[self.object_1][0]
        pos_2 = positions_dict[self.object_2][0]

        positions = np.linspace(pos_1, pos_2, self.num_spheres)

        return {self: positions}

    def update_positions(self, positions_dict):
        # TODO revise logic for getting the reference point
        # Address varying number of spheres

        pos_1 = positions_dict[self.object_1][0]
        pos_2 = positions_dict[self.object_2][0]

        positions = np.linspace(pos_1, pos_2, self.num_spheres)

        self.positions = positions


class Interconnect(InterconnectNode, InterconnectSegment):
    """
    Interconnects are made of one or more non-zero-length segments and connect two components.

    TODO Add a class of components for interconnect dividers (e.g., pipe tee for a three-way split)

    When an interconnect is initialized it does not contain spatial information.

    In the SPI2 class the user specifies which layout generation method to use, and that method tells
    the Interconnect InterconnectNodes what their positions are
    """

    def __init__(self, component_1, component_2, diameter, color):
        self.component_1 = component_1
        self.component_2 = component_2

        self.diameter = diameter
        self.radius = diameter / 2
        self.color = color

        # Per configuration file
        self.numbers_of_nodes = 2

    def add_node(self):
        pass

    def add_nodes(self):
        pass

    def add_segment(self):
        pass

    def add_segments(self):
        pass





    @property
    def edges(self):
        pass
        # self.nodes





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


