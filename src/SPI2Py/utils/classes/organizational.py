"""

"""

import numpy as np
from itertools import product, combinations
from ..visualization.visualization import plot


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

            # Get positions of design  classes
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

