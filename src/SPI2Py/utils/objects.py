"""Module...
Docstring

TODO Fill out all the blank functions and write tests for them...
TODO Look into replacing get/set methods with appropriate decorators...


"""

import numpy as np
from scipy.spatial.distance import euclidean
from itertools import product, combinations

from src.SPI2Py.utils.visualization.visualization import plot
from src.SPI2Py.utils.spatial_calculations.transformations import translate, rotate_about_point


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
    def __init__(self, node, position=np.array([0., 0., 0.])):
        self.node = node
        self.position = position

    @property
    def reference_position(self):
        return self.positions[0]

    def update_position(self, design_vector, constraint=None):

        if constraint is None:

            new_reference_position = design_vector[0:3]

            delta_position = new_reference_position - self.reference_position

            translated_positions = translate(self.positions, delta_position)

            # Update values
            self.position = translated_positions

        else:
            print('Placeholder')

    @property
    def design_vector(self):
        return self.position


class InterconnectSegment:
    def __init__(self, component_1, component_2, diameter, color):
        self.component_1 = component_1
        self.component_2 = component_2

        self.diameter = diameter
        self.radius = diameter / 2
        self.color = color

        # Create edge tuple for NetworkX graphs
        self.edge = (self.component_1.node, self.component_2.node)

        # Placeholder for plot test functionality, random positions
        self.positions, self.radii = self.set_positions()

        # Temporary
        self.num_spheres = len(self.radii)

    def set_positions(self):
        pos_1 = self.component_1.reference_position
        pos_2 = self.component_2.reference_position

        dist = euclidean(pos_1, pos_2)

        num_spheres = int(dist / self.diameter)

        # Don't want zero-length interconnects
        if num_spheres == 0:
            num_spheres = 1

        positions = np.linspace(pos_1, pos_2, num_spheres)

        # Temporary value

        radii = np.repeat(self.radius, positions.shape[0])

        return positions, radii

    def calculate_positions(self, positions_dict):
        # TODO revise logic for getting the reference point
        # Address varying number of spheres

        pos_1 = positions_dict[self.component_1][0]
        pos_2 = positions_dict[self.component_2][0]

        positions = np.linspace(pos_1, pos_2, self.num_spheres)

        return {self: positions}

    def update_positions(self, positions_dict):
        # TODO revise logic for getting the reference point
        # Address varying number of spheres

        pos_1 = positions_dict[self.component_1][0]
        pos_2 = positions_dict[self.component_2][0]

        positions = np.linspace(pos_1, pos_2, self.num_spheres)

        self.positions = positions


class Interconnect(InterconnectNode, InterconnectSegment):
    """

    """
    pass


class Structure:
    def __init__(self, positions, radii, color, name):
        self.positions = positions
        self.radii = radii
        self.color = color
        self.name = name





class Subsystem:
    """
    Defines the non-spatial aspects of the system.

    """

    def __init__(self, components, interconnect_nodes, interconnects, structures):
        self.components = components
        self.interconnect_nodes = interconnect_nodes
        self.interconnects = interconnects
        self.structures = structures

    @property
    def objects(self):
        return self.components + self.interconnect_nodes + self.interconnects + self.structures

    @property
    def design_vector_objects(self):
        return self.components + self.interconnect_nodes

    @property
    def moving_objects(self):
        return self.components + self.interconnect_nodes + self.interconnects

    @property
    def nodes(self):
        return [design_object.node for design_object in self.design_vector_objects]

    @property
    def edges(self):
        return [interconnect.edge for interconnect in self.interconnects]

    def add_object(self):
        # TODO Implement this function
        pass

    def remove_object(self):
        # TODO Implement this function
        pass

    @property
    def component_component_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        component_component_pairs = list(combinations(self.components, 2))

        return component_component_pairs

    @property
    def component_interconnect_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        component_interconnect_pairs = list(product(self.components, self.interconnects))

        # Remove interconnects attached to components b/c ...
        for component, interconnect in component_interconnect_pairs:

            if component is interconnect.component_1 or component is interconnect.component_2:
                component_interconnect_pairs.remove((component, interconnect))

        return component_interconnect_pairs

    @property
    def interconnect_interconnect_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        # Create a list of all interconnect pairs
        interconnect_interconnect_pairs = list(combinations(self.interconnects, 2))

        # Remove segments that share the same component
        # This is a temporary feature b/c the ends of those two interconnects will share a point
        # and thus always overlap slightly...
        # TODO implement this feature

        # Remove segments that are from the same interconnect
        # If a pipe intersects itself, usually the pipe can just be made shorter...
        # TODO implement this feature

        return interconnect_interconnect_pairs

    @property
    def structure_moving_object_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        structure_moving_object_pairs = list(product(self.structures, self.moving_objects))

        return structure_moving_object_pairs


class System(Subsystem):
    """
    System combines subsystems from various disciplines (e.g., fluid, electrical, structural)
    """
    pass


class SpatialConfiguration(System):
    """

    """

    @property
    def design_vector(self):
        """
        Flattened format is necessary for the optimizer...

        Returns
        -------

        """
        # Convert this to a flatten-like from design_vectors?
        design_vector = np.empty(0)
        for obj in self.design_vector_objects:
            design_vector = np.concatenate((design_vector, obj.design_vector))

        return design_vector

    def slice_design_vector(self, design_vector):
        """
        Since design vectors are irregularly sized, come up with indexing scheme.
        Not the most elegant method.

        Takes argument b.c. new design vector...

        :return:
        """

        # Get the size of the design vector for each design vector object
        design_vector_sizes = []
        for i, obj in enumerate(self.design_vector_objects):
            design_vector_sizes.append(obj.design_vector.size)

        # Index values
        start, stop = 0, 0
        design_vectors = []

        for i, size in enumerate(design_vector_sizes):
            # Increment stop index
            stop += design_vector_sizes[i]

            design_vectors.append(design_vector[start:stop])

            # Increment start index
            start = stop

        return design_vectors

    def get_positions(self, design_vector=None):
        """
        TODO get positions for interconnects, structures, etc
        :param design_vector:
        :param design_vector:
        :return:
        """

        positions_dict = {}

        if design_vector is None:
            for obj in self.objects:
                positions_dict[obj] = obj.positions

        else:
            design_vectors = self.slice_design_vector(design_vector)

            # Get positions of design  objects
            for obj, design_vector_row in zip(self.design_vector_objects, design_vectors):
                positions_dict = {**positions_dict, **obj.get_positions(design_vector_row)}

            for interconnect in self.interconnects:
                positions_dict = {**positions_dict, **interconnect.calculate_positions(positions_dict)}

            # Get positions of interconnect nodes and structures...

        return positions_dict

    def set_positions(self, new_design_vector):

        new_design_vectors = self.slice_design_vector(new_design_vector)

        positions_dict = self.get_positions(new_design_vector)

        for obj, new_design_vector in zip(self.design_vector_objects, new_design_vectors):
            obj.update_positions(new_design_vector)

        # Is there some class-specific logic?
        # for obj in self.components:
        #     pass
        #
        # for obj in self.interconnect_nodes:
        #     pass
        #
        for interconnect in self.interconnects:
            interconnect.update_positions(positions_dict)

    def plot_layout(self, savefig=False, directory=None):

        layout_plot_dict = {}

        for obj in self.objects:
            object_plot_dict = {}

            positions = obj.positions
            radii = obj.radii
            color = obj.color

            object_plot_dict['positions'] = positions
            object_plot_dict['radii'] = radii
            object_plot_dict['color'] = color

            layout_plot_dict[obj] = object_plot_dict

        plot(layout_plot_dict, savefig, directory)

