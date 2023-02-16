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
    def static_objects(self):

        objects = []

        for obj in self.objects:
            if obj.movement == 'static':
                objects.append(obj)

        return objects

    @property
    def dynamic_independent_objects(self):

        objects = []

        for obj in self.objects:
            if obj.movement == 'dynamic_independent':
                objects.append(obj)

        return objects

    @property
    def dynamic_partially_dependent_objects(self):

        """
        Since partially dependent objects can be constrained to other partially dependent objects, we
        need to order them in a way where looping through the list will update properly.

        Since we are checking we invert and check if static/independant

        TODO Write unit tests to confirm sorting works properly
        """

        # Create the unordered list
        objects = []

        for obj in self.objects:
            if obj.movement == 'dynamic_partially_dependent':
                objects.append(obj)

        # Order the list

        # First, for simplicity, sort the objects into lists.
        # Since dynamic partially dependent objects are updated after all other movement classes,
        # if a dynamic partially dependent object is constrained to them, the order doesn't matter
        # However, if a dynamic partially dependent object is constrained to another dynamic partially dependent object,
        # We must make sure the constrained is called after the constraining object.
        dependent_on_another_dependent = []
        not_dependent_on_another_dependent = []


        for obj in objects:

            # If all the reference objects are either static or dynamic independent, then we can add it to the list

            if all([ref_obj in self.static_objects or ref_obj in self.dynamic_independent_objects for ref_obj in obj.reference_objects]):
                not_dependent_on_another_dependent.append(obj)
            else:
                dependent_on_another_dependent.append(obj)

        # Now, we need to order the dependent objects
        # The exact order does not matter, we just must ensure items don't reference future items
        # Sort two lists based on one (list of objects and list of reference objects)

        # Arbitrary number to prevent infinite loops
        iter = 0
        max_iterations = 50

        # Start with all objects in cache and then remove them as they are added to the ordered list
        cache = dependent_on_another_dependent.copy()
        dependent_on_another_dependent_ordered = []

        while len(cache) > 0 and iter < max_iterations:

            # During each loop, first loop through the remaining objects in cache and add nobrainers
            for obj in cache:

                # If no calls then add and delete
                pass

            # Then, loop through the remaining objects in cache and
            for obj in cache:
                # If calls, find the
                pass

            if iter == max_iterations-1:
                # TODO Check for circular references and overconstrained objects in the InputValidation class
                raise Exception('The objects are not properly ordered. There is likely a circular reference.')


        ordered_objects = not_dependent_on_another_dependent + dependent_on_another_dependent_ordered


        return ordered_objects

    @property
    def dynamic_fully_dependent_objects(self):

            objects = []

            for obj in self.objects:
                if obj.movement == 'dynamic_fully_dependent':
                    objects.append(obj)

            return objects

    @property
    def _uncategorized_objects(self):

        """
        This list should be empty. It checks to make sure every object is categorized as one of the four types of objects.
        """

        uncategorized = []

        categorized = self.static_objects + self.dynamic_independent_objects + self.dynamic_partially_dependent_objects + self.dynamic_fully_dependent_objects

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
