"""Module...
Docstring

TODO Fill out all the blank functions and write tests for them...
TODO Look into replacing get/set methods with appropriate decorators...


"""
from itertools import combinations, product


class Subsystem:
    """
    Defines the non-spatial aspects of the system.

    """

    def __init__(self, components, ports, interconnects, interconnect_nodes, interconnect_segments, structures):
        self.components = components
        self.ports = ports
        self.interconnects = interconnects
        self.interconnect_nodes = interconnect_nodes
        self.interconnect_segments = interconnect_segments
        self.structures = structures

        

    @property
    def objects(self):
        return self.components + self.ports + self.interconnect_nodes + self.interconnect_segments + self.structures

    @property
    def design_vector_objects(self):
        return self.components + self.interconnect_nodes

    @property
    def moving_objects(self):
        return self.components + self.ports + self.interconnect_nodes + self.interconnect_segments

    @property
    def nodes(self):
        return [design_object.node for design_object in self.design_vector_objects]

    @property
    def edges(self):
        return [interconnect.edge for interconnect in self.interconnect_segments]

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

        component_interconnect_pairs = list(product(self.components, self.interconnect_segments))

        # Remove interconnects attached to components b/c ...
        for component, interconnect in component_interconnect_pairs:

            if component is interconnect.object_1 or component is interconnect.object_2:
                component_interconnect_pairs.remove((component, interconnect))

        return component_interconnect_pairs

    @property
    def interconnect_interconnect_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        # Create a list of all interconnect pairs
        interconnect_interconnect_pairs = list(combinations(self.interconnect_segments, 2))

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
