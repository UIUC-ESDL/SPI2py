import numpy as np
from itertools import combinations, product
from .objects import Object, Port, Interconnect
from .geometry.geometric_representation import generate_rectangular_prisms
from .kinematics.depricated_kinematics import calculate_independent_positions, calculate_static_positions


class System:
    """
    Defines the associative (non-spatial) aspects of systems.

    The layout module of SPI2py may map a single System to multiple SpatialConfigurations, performing gradient-based
    optimization on each SpatialConfiguration in parallel.

    This associative information includes things such as which objects to check check for collision between.
    Movement classes and associated constraints dictate which objects have design variables and if those design
    variables are constrained.

    There are four types of objects:
    1. Static:              Objects that cannot move (but must still be mapped to the spatial configuration)
    2. Independent:         Objects that can move independently of each other
    3. Partially Dependent: Objects that are constrained relative to other object(s) but retain some degree of freedom
    4. Fully Dependent:     Objects that are fully constrained to another object (e.g., the port of a component)

    Note: For the time being we will assume that you cannot chain dependencies.
    For example, we may assert that the position of objects B,C, and D may depend directly on A.
    However, we may not assert that the position of object D depends on C, which depends on the position of B.

    TODO Change positions dict from pass-thru edit to merge edit
    TODO Add a method to consistently order all objects based on their dependency chains
    TODO Add a method to check for circular dependencies and overconstrained objects
    TODO Write unit tests to confirm that all property-decorated functions correctly apply filters
    """

    def __init__(self,
                 name: str):

        self.name = name
        self.components = []
        self.ports = []
        self.interconnects = []
        self.interconnect_nodes = []
        self.interconnect_segments = []


    def __repr__(self):
        return f'System({self.name})'

    def __str__(self):
        return self.name

    def add_component(self,
                      name: str,
                      color: str,
                      movement_class: str,
                      shapes: list):

        """
        Add a component to the system.

        :param name:
        :param color:
        :param movement_class:
        :param shapes:
        :return:
        """

        origins = []
        dimensions = []
        for shape in shapes:
            origins.append(shape['origin'])
            dimensions.append(shape['dimensions'])

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        component = Object(name, positions, radii, color, movement_class=movement_class)

        # Update the system
        self.components.append(component)

    def add_port(self, component_name, port_name, color, radius, reference_point_offset, movement_class):
        """
        Add a port to the system.

        :param
        """
        port = Port(component_name, port_name, color, radius, reference_point_offset, movement_class=movement_class)
        self.ports.append(port)

    def add_interconnect(self, name, component_1, component_1_port, component_2, component_2_port, radius, color,
                         number_of_bends):
        """
        Add an interconnect to the system.

        """
        interconnect = Interconnect(name, component_1, component_1_port, component_2, component_2_port, radius, color,
                                    number_of_bends)

        self.interconnects.append(interconnect)
        self.interconnect_segments.extend(interconnect.segments)
        self.interconnect_nodes.extend(interconnect.interconnect_nodes)

    @property
    def objects(self):
        return self.components + self.ports + self.interconnect_nodes + self.interconnect_segments

    def _validate_components(self):
        # TODO Implement
        pass

    def _validate_ports(self):
        # TODO Implement
        pass

    def _validate_interconnects(self):
        # TODO Implement
        pass

    def _validate_interconnect_nodes(self):
        # TODO Implement
        pass

    def _validate_interconnect_segments(self):
        # TODO Implement
        pass

    def _validate_structures(self):
        # TODO Implement
        pass

    @property
    def nodes(self):
        return [str(obj) for obj in self.components + self.interconnect_nodes]

    @property
    def edges(self):
        edges = []
        for segment in self.interconnect_segments:
            # Split with "-" to just get component (not port name)
            # TODO Create a more reliable way to get the component name
            edge = (str(segment.object_1).split('-')[0], str(segment.object_2).split('-')[0])
            edges.append(edge)

        return edges

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
        objects = []
        for obj in self.objects:
            if obj.movement_class == 'partially dependent':
                objects.append(obj)
        return objects

    @property
    def fully_dependent_objects(self):
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
        """TODO Vectorize with cartesian product"""
        return list(combinations(self.components, 2))

    @property
    def component_interconnect_pairs(self):
        """
        Pairing logic:
        1. Don't check for collision between a component and the interconnect that is attached to it.

        TODO Write unit tests to ensure it creates the correct pairs
        TODO Vectorize with cartesian product

        :return:
        """

        pairs = list(product(self.components, self.interconnect_segments))

        # Remove pairs that contain a component and its own interconnect
        for component, interconnect in pairs:
            if component is interconnect.object_1 or component is interconnect.object_2:
                pairs.remove((component, interconnect))

        return pairs


    @property
    def interconnect_interconnect_pairs(self):
        """
        Pairing logic:
        1. Don't check for collision between two segments of the same interconnect.

        TODO Write unit tests to ensure it creates the correct pairs
        TODO Vectorize with cartesian product

        :return:
        """

        # Create a list of all interconnect pairs
        pairs = list(combinations(self.interconnect_segments, 2))

        # Remove segments that are from the same interconnect
        # If a pipe intersects itself, usually the pipe can just be made shorter...
        # TODO implement this feature

        return pairs

    @property
    def object_pairs(self):
        object_pairs = []
        object_pairs += [self.component_component_pairs]
        object_pairs += [self.component_interconnect_pairs]
        object_pairs += [self.interconnect_interconnect_pairs]

        return object_pairs

    @property
    def design_vector(self):
        """
        Flattened format is necessary for the optimizer...

        Returns
        -------

        """

        design_vector = np.empty(0)

        for obj in self.independent_objects:
            design_vector = np.concatenate((design_vector, obj.design_vector))

        for obj in self.partially_dependent_objects:
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
        for i, obj in enumerate(self.independent_objects):
            design_vector_sizes.append(obj.design_vector.size)

        for i, obj in enumerate(self.partially_dependent_objects):
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

        :param design_vector:
        :param design_vector:
        :return:

        TODO get positions for interconnects, structures, etc
        TODO Remove unnecessary design vector arguments
        """

        positions_dict = {}

        design_vectors = self.slice_design_vector(design_vector)

        # STATIC OBJECTS
        for obj in self.static_objects:
            # positions_dict = obj.calculate_positions(design_vector, positions_dict)
            positions_dict = calculate_static_positions(obj, positions_dict)

        # DYNAMIC OBJECTS - Independent then Partially Dependent
        objects = self.independent_objects + self.partially_dependent_objects
        for obj, design_vector_row in zip(objects, design_vectors):
            positions_dict = calculate_independent_positions(obj, design_vector_row, positions_dict)

        # DYNAMIC OBJECTS - Fully Dependent
        for obj in self.fully_dependent_objects:
            positions_dict = obj.calculate_positions(design_vector, positions_dict)

        # TODO Add method for partially dependent objects
        # DYNAMIC OBJECTS - Partially Dependent



        return positions_dict


    def set_positions(self, positions_dict):
        """set_positions Sets the positions of the objects in the layout.

        Takes a flattened design vector and sets the positions of the objects in the layout.

        :param positions_dict: A dictionary of positions for each object in the layout.
        :type positions_dict: dict
        """

        for obj in self.objects:
            obj.set_positions(positions_dict)

    def map_static_object(self, object_name, design_vector):
        """
        Maps an object to a spatial configuration.

        :param obj: The object to be mapped.
        :type obj: Object
        :param design_vector: The design vector; must be a (1, 6) array for x, y, z, rx, ry, rz.
        """

        obj = next(obj for obj in self.objects if obj.__repr__() == object_name)

        # Get the positions of the object
        positions = obj.positions
        radii = obj.radii

        positions_dict = calculate_independent_positions(obj, design_vector, {})

        # Set the positions of the object
        obj.set_positions(positions_dict)

        return positions_dict

    # def extract_spatial_graph(self):
    #     """
    #     Extracts a spatial graph from the layout.
    #
    #     TODO Remove unconnected nodes?
    #
    #     :return:
    #     """
    #
    #     nodes = self.nodes
    #     edges = self.edges
    #
    #     node_positions = []
    #
    #     positions_dict = self.calculate_positions(self.design_vector)
    #
    #     # TODO Implement
    #
    #     return nodes, node_positions, edges
