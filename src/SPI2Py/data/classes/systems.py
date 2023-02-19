"""Module...
Docstring

TODO Fill out all the blank functions and write tests for them...
TODO Look into replacing get/set methods with appropriate decorators...


"""
from itertools import combinations, product

import numpy as np

from SPI2Py.result.visualization.plotting import plot


class System:
    """
    Defines the associative (non-spatial) aspects of the system.

    There are four types of objects:
    1. Static: Objects that cannot move
    2. Independent: Objects that can move independently of each other
    3. Partially Dependent: Objects that are constrained relative to other object(s) but retain some dergee(s) of freedom
        -Coaxial components
    4. Fully Dependent: Objects that are fully constrained to another object
        -The ports of a component

    TODO Change positions dict from pass-thru edit to merge edit
    TODO Add a method to specifically and consistently order all objects based on their dependancy (and check for cirular/overconstrained)
    """

    def __init__(self, components, ports, interconnects, interconnect_nodes, interconnect_segments, structures, config):
        self.components = components
        self.ports = ports
        self.interconnects = interconnects
        self.interconnect_nodes = interconnect_nodes
        self.interconnect_segments = interconnect_segments
        self.structures = structures
        self.config = config


    @property
    def objects(self):
        return self.components + self.ports + self.interconnect_nodes + self.interconnect_segments + self.structures


    @property
    def static_objects(self):

        objects = []

        for obj in self.objects:
            if obj.movement_class == 'static':
                objects.append(obj)

        return objects

    @property
    def independent_objects(self):

        objects = []

        for obj in self.objects:
            if obj.movement_class == 'independent':
                objects.append(obj)

        return objects

    @property
    def partially_dependent_objects(self):

        """
        Partially dependent objects are objects that are constrained to other objects but retain some
        degree of freedom (e.g. coaxial components).

        For the time being, we will simplify the problem by assuming that you cannot chain dependencies.
        For example, we may assert that the position of objects B,C, and D may depend directly on A. However,
        we may not assert that the position of object D depends on the position of C, which depends on the position f B, etc.

        TODO Write unit tests to confirm sorting works properly
        TODO Add check that ensures partially dependent objects are not constrained to other partially or fully dependent objects
        """

        # Create the unordered list
        objects = []

        for obj in self.objects:
            if obj.movement_class == 'partially dependent':
                objects.append(obj)

        return objects

    @property
    def fully_dependent_objects(self):
        # TODO Sort to makesure ports are before edges...

        objects = []

        for obj in self.objects:
            if obj.movement_class == 'fully dependent':
                objects.append(obj)

        return objects

    @property
    def _uncategorized_objects(self):

        """
        This list should be empty. It checks to make sure every object is categorized as one of the four types of objects.
        """

        uncategorized = []

        categorized = self.static_objects + self.independent_objects + self.partially_dependent_objects + self.fully_dependent_objects

        for obj in self.objects:
            if obj not in categorized:
                uncategorized.append(obj)

        return uncategorized

    @property
    def component_component_pairs(self):
        return list(combinations(self.components, 2))


    @property
    def component_interconnect_pairs(self):
        """
        Pairing logic:
        1. Don't check for collision between a component and the interconnect that is attached to it.

        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        pairs = list(product(self.components, self.interconnect_segments))

        # Remove pairs that contain a component and its own interconnect
        for component, interconnect in pairs:
            if component is interconnect.object_1 or component is interconnect.object_2:
                pairs.remove((component, interconnect))

        return pairs

    @property
    def component_structure_pairs(self):
        return list(product(self.components, self.structures))

    @property
    def interconnect_interconnect_pairs(self):
        """
        Pairing logic:
        1. Don't check for collision between two segments of the same interconnect.

        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        # Create a list of all interconnect pairs
        pairs = list(combinations(self.interconnect_segments, 2))


        # Remove segments that are from the same interconnect
        # If a pipe intersects itself, usually the pipe can just be made shorter...
        # TODO implement this feature

        return pairs



    @property
    def interconnect_structure_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        interconnect_structure_pairs = list(product(self.interconnect_segments, self.structures))

        return interconnect_structure_pairs

    @property
    def object_pairs(self):
        object_pairs = []
        object_pairs += [self.component_component_pairs]
        object_pairs += [self.component_interconnect_pairs]
        object_pairs += [self.component_structure_pairs]
        object_pairs += [self.interconnect_interconnect_pairs]
        object_pairs += [self.interconnect_structure_pairs]

        return object_pairs


class SpatialConfiguration:
    """
    When creating a spatial configuration, we must automatically map static objects
    TODO Add an update order...
    """
    def __init__(self, system):
        self.system = system

    @property
    def design_vector(self):
        """
        Flattened format is necessary for the optimizer...

        Returns
        -------

        """

        design_vector = np.empty(0)

        for obj in self.system.independent_objects:
            design_vector = np.concatenate((design_vector, obj.design_vector))

        for obj in self.system.partially_dependent_objects:
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
        for i, obj in enumerate(self.system.independent_objects):
            design_vector_sizes.append(obj.design_vector.size)

        for i, obj in enumerate(self.system.partially_dependent_objects):
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

    def calculate_positions(self, design_vector):
        """
        TODO get positions for interconnects, structures, etc
        :param design_vector:
        :param design_vector:
        :return:
        TODO Remove unnecessary design vector arguments
        """

        positions_dict = {}

        design_vectors = self.slice_design_vector(design_vector)

        # STATIC OBJECTS
        for obj in self.system.static_objects:
            positions_dict = obj.calculate_positions(design_vector,positions_dict)

        # DYNAMIC OBJECTS - Independent then Partially Dependent
        objects = self.system.independent_objects + self.system.partially_dependent_objects
        for obj, design_vector_row in zip(objects, design_vectors):
            positions_dict = obj.calculate_positions(design_vector_row, positions_dict)

        # DYNAMIC OBJECTS - Fully Dependent
        for obj in self.system.fully_dependent_objects:
            positions_dict = obj.calculate_positions(design_vector,positions_dict)

        return positions_dict

    def set_positions(self, positions_dict):
        """set_positions Sets the positions of the objects in the layout.

        Takes a flattened design vector and sets the positions of the objects in the layout.

        :param positions_dict: A dictionary of positions for each object in the layout.
        :type positions_dict: dict
        """

        for obj in self.system.objects:
            obj.set_positions(positions_dict)

    def map_static_objects(self):

        """
        This method maps static objects to spatial configurations.

        Although static objects do not move or have design variables, they still need to be mapped
        to spatial configurations. This method temporarily replaces the default self.calculate_static_positions
        method with the self.calculate_independent_positions method, treating the self.static_position attribute
        as a design vector.
        """

        for static_object in self.system.static_objects:
            pos_dict = static_object.calculate_independent_positions(static_object.static_position, {})
            static_object.set_positions(pos_dict)


    def plot_layout(self, savefig=False, directory=None):

        layout_plot_array = []

        for obj in self.system.objects:

            positions = obj.positions
            radii = obj.radii
            color = obj.color

            layout_plot_array.append([positions, radii, color])

        fig = plot(layout_plot_array, savefig, directory,self.system.config)

        fig.show()


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
