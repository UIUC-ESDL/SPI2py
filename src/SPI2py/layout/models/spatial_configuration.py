import numpy as np

# TODO compartmentalize this dependency
from ...result.visualization.plotting import plot_objects


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

        :param design_vector:
        :param design_vector:
        :return:

        TODO get positions for interconnects, structures, etc
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
            pos_dict = static_object.calculate_independent_positions(static_object.positions[0], {})
            static_object.set_positions(pos_dict)


    def plot(self, savefig=False, directory=None):

        layout_plot_array = []

        for obj in self.system.objects:

            positions = obj.positions
            radii = obj.radii
            color = obj.color

            layout_plot_array.append([positions, radii, color])

        fig = plot_objects(layout_plot_array, savefig, directory, self.system.config)


    def extract_spatial_graph(self):
        """
        Extracts a spatial graph from the layout.

        TODO Remove unconnected nodes?

        :return:
        """

        nodes = self.system.nodes
        edges = self.system.edges

        node_positions = []

        positions_dict = self.calculate_positions(self.design_vector)

        # TODO Implement


        return nodes, node_positions, edges