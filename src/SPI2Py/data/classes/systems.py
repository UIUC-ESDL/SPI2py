"""Module...
Docstring

TODO Fill out all the blank functions and write tests for them...
TODO Look into replacing get/set methods with appropriate decorators...


"""
from itertools import combinations, product


class Subsystem:
    """
    Defines the non-spatial aspects of the system.

    There are four types of objects:
    1. Static
        -Structures
        -Sensors that must be fixed in space

    2. Dynamic, Independent: Objects that can move independently of each other
        -...

    3. Dynamic, Partially Dependent: Objects that are constrained relative to other object(s) but retain some dergee(s) of freedom
        -Coaxial components

    4. Dynamic, Fully Dependent: Objects that are fully constrained to another object
        -The ports of a component

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
    def objects_static(self):
        # TODO Implement
        pass

    @property
    def objects_dynamic_independent(self):
        # TODO Implement
        pass

    @property
    def objects_dynamic_partially_dependent(self):
        # TODO Implement
        pass

    @property
    def objects_dynamic_fully_dependent(self):
        # TODO Implement
        pass

    @property
    def _objects_uncategorized(self):

        # Should be empty, checks if above missed anything

        uncategorized = []

        categorized = self.objects_static + self.objects_dynamic_independent + self.objects_dynamic_partially_dependent + self.objects_dynamic_fully_dependent

        for obj in self.objects:
            if obj not in categorized:
                uncategorized.append(obj)

        return uncategorized



    @property
    def design_vector_objects(self):

        # Implement if not fixed in space
        # TODO only take waypoints if they can move ...

        return [component for component in self.components if component.degrees_of_freedom is not None] + self.interconnect_nodes

    @property
    def moving_objects(self):
        # TODO Not fully correct
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
    def component_structure_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        component_structure_pairs = list(product(self.components, self.structures))

        return component_structure_pairs

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
    def interconnect_structure_pairs(self):
        """
        TODO Write unit tests to ensure it creates the correct pairs

        :return:
        """

        interconnect_structure_pairs = list(product(self.interconnect_segments, self.structures))

        return interconnect_structure_pairs


class System(Subsystem):
    """
    System combines subsystems from various disciplines (e.g., fluid, electrical, structural)
    """
    pass
