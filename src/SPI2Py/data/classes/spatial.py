"""
TODO Make sure that migrating from InterconnectSegment to Interconnect does not mess with collision pairs
"""

import numpy as np

from .systems import System
from ...result.visualization.visualization import plot


class SpatialConfiguration(System):
    """

    TODO Add an update order...
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

    def calculate_positions(self, design_vector=None):
        """
        TODO get positions for interconnects, structures, etc
        :param design_vector:
        :param design_vector:
        :return:
        """

        positions_dict = {}

        if design_vector is None:

            for obj in self.objects:
                positions_dict[str(obj)] = (obj.positions, obj.radii)

        else:

            design_vectors = self.slice_design_vector(design_vector)

            # Get positions of design  classes
            for obj, design_vector_row in zip(self.design_vector_objects, design_vectors):
                positions_dict = {**positions_dict, **obj.calculate_positions(design_vector_row)}

            # For interconnect nodes and segments...
            for interconnect in self.interconnect_segments:
                positions_dict = {**positions_dict, **interconnect.calculate_positions(positions_dict)}

            # Get positions of interconnect nodes and structures...

        return positions_dict

    def set_positions(self, new_design_vector):
        """set_positions Sets the positions of the objects in the layout.

        Takes a flattened design vector and sets the positions of the objects in the layout.

        :param new_design_vector: Desired design vector
        :type new_design_vector: np.ndarray
        """

        new_design_vectors = self.slice_design_vector(new_design_vector)

        positions_dict = self.calculate_positions(new_design_vector)


        for obj, new_design_vector in zip(self.design_vector_objects, new_design_vectors):
            obj.update_positions(positions_dict)


        # Is there some class-specific logic?
        # for obj in self.components:
        #     pass
        #
        # for obj in self.interconnect_nodes:
        #     pass
        #
        for interconnect in self.interconnect_segments:

            interconnect.update_positions(positions_dict)

    def plot_layout(self, savefig=False, directory=None):

        layout_plot_array = []

        for obj in self.objects:

            positions = obj.positions
            radii = obj.radii
            color = obj.color

            layout_plot_array.append([positions, radii, color])

        plot(layout_plot_array, savefig, directory)


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
