"""Module...
Docstring

TODO Fill out all the blank functions and write tests for them...
TODO Look into replacing get/set methods with appropriate decorators...


"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
from itertools import product, combinations
from src.SPI2Py.utils.shape_generator import generate_rectangular_prism, generate_rectangular_prisms
from src.SPI2Py.utils.visualization import plot
from src.SPI2Py.utils.spatial_transformations import translate, rotate


class Component:

    def __init__(self, positions, radii, color, node, name):

        self.positions = positions
        self.radii = radii
        self.color = color
        self.name = name

        self.node = node

        self.rotation = np.array([0, 0, 0])  # Initialize the rotation attribute

    @property
    def reference_position(self):
        return self.positions[0]

    def update_positions(self, design_vector, constraint=None):
        """
        Update positions of object spheres given a design vector

        Constraint refers to how if we constrain the object, it will have a different size design vector

        :param design_vector:
        :param constraint:
        :return:
        """

        # Assumes (1,6) design vector... will need to expand in future
        if constraint is None:

            new_reference_position = design_vector[0:3]
            new_rotation = design_vector[3:None]

            delta_position = new_reference_position - self.reference_position
            delta_rotation = new_rotation - self.rotation

            translated_positions = translate(self.positions, delta_position)

            rotated_translated_positions = rotate(translated_positions, delta_rotation)

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


class Interconnect:
    def __init__(self, component_1, component_2, diameter, color):

        self.component_1 = component_1
        self.component_2 = component_2

        self.diameter = diameter
        self.radius = diameter/2
        self.color = color

        # Create edge tuple for NetworkX graphs
        self.edge = (self.component_1.node, self.component_2.node)

        # Placeholder for plot test functionality, random positions
        self.positions, self.radii = self.update_positions()

    def update_positions(self):
        pos_1 = self.component_1.reference_position
        pos_2 = self.component_2.reference_position

        dist = euclidean(pos_1, pos_2)

        num_spheres = int(dist / self.diameter)

        positions = np.linspace(pos_1, pos_2, num_spheres)

        # Temporary value

        radii = np.repeat(self.radius, positions.shape[0])

        return positions, radii


class Structure:
    def __init__(self, positions, radii, color, name):
        self.positions = positions
        self.radii = radii
        self.color = color
        self.name = name


class Layout:
    def __init__(self, components, interconnect_nodes, interconnects, structures):
        self.components = components
        self.interconnect_nodes = interconnect_nodes
        self.interconnects = interconnects
        self.structures = structures

        # All objects
        self.objects = components + interconnect_nodes + interconnects + structures

        # All objects with a design vector
        self.design_objects = components + interconnect_nodes

        #
        self.nodes = [design_object.node for design_object in self.design_objects]
        self.edges = [interconnect.edge for interconnect in self.interconnects]

        #
        self.design_vector_objects = components + interconnect_nodes

    @property
    def positions(self):

        positions = np.empty((0,3))

        for obj in self.objects:
            positions = np.vstack((positions,obj.positions))

        return positions
    
    @property
    def component_component_pairs(self):
        return list(combinations(self.components, 2))

    @property
    def component_interconnect_pairs(self):

        component_interconnect_pairs = list(product(self.components, self.interconnects))


        # TODO Make it sure it's actually removing the correct pairs...

        # Remove interconnects attached to components b/c ...
        for component, interconnect in component_interconnect_pairs:

            if component is interconnect.component_1 or component is interconnect.component_2:
                component_interconnect_pairs.remove((component, interconnect))

        return component_interconnect_pairs

    @property
    def interconnect_interconnect_pairs(self):

        # TODO Test case: empty
        # TODO Remove segments from the same interconnect
        # TODO Remove segments from the same component

        return list(combinations(self.interconnects, 2))

    @property
    def structure_all_pairs(self):
        pass

    @property
    def all_pairs(self):
        pass

    def generate_random_layout(self):
        """
        Generates random layouts using a force-directed algorithm

        Initially assumes 0 rotation and 1x6 design vector
        TODO Add flexible design vector size and rotation


        :return:
        """

        g = nx.MultiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)

        # Optimal distance between nodes
        k = 1

        scale = 4

        # TODO remove seed for actual problems...
        seed = 1

        # Dimension of layout
        dim = 3

        positions = nx.spring_layout(g, k=k, dim=dim, scale=scale, seed=seed)

        # Generate random angles too?

        # Temporarily pad zeros for rotations
        design_vectors = []
        rotation = np.array([0,0,0])

        for i in positions:
            position = positions[i]
            design_vector = np.concatenate((position, rotation))
            design_vectors.append(design_vector)

        # Flatten design vectors
        # TODO Make more efficient?
        design_vector = np.concatenate(design_vectors)

        return design_vector

    @property
    def design_vector(self):

        # Convert this to a flatten-like from design_vectors?
        design_vector = np.empty(0)
        for obj in self.design_vector_objects:
            design_vector = np.concatenate((design_vector, obj.design_vector))

        return design_vector

    @property
    def design_vectors(self):
        """
        Returns objects and their design vectors

        :return:
        """

        num_design_vectors = len(self.design_vector_objects)
        design_vectors = []
        for i, obj in enumerate(self.design_vector_objects):
            design_vectors.append(obj.design_vector)

        return self.design_vector_objects, design_vectors

    @property
    def design_vector_sizes(self):

        num_design_vectors = len(self.design_vector_objects)
        design_vector_sizes = []
        for i, obj in enumerate(self.design_vector_objects):
            design_vector_sizes.append(obj.design_vector.size)

        return design_vector_sizes

    @property
    def reference_positions(self):

        reference_positions_dict = {}

        for obj in self.objects:
            reference_positions_dict[obj] = obj.reference_position

        return reference_positions_dict

    def slice_design_vector(self, design_vector):
        """
        Since design vectors are irregularly sized, come up with indexing scheme.
        Not the most elegant method.

        Takes argument b.c. new design vector...

        :return:
        """

        design_vectors = []

        # Index values
        start = 0
        stop = 0

        for i, size in enumerate(self.design_vector_sizes):

            # Increment stop index
            stop += self.design_vector_sizes[i]

            design_vectors.append(design_vector[start:stop])

            # Increment start index
            start = stop

        return design_vectors

    def update_positions(self, new_design_vector):

        new_design_vectors = self.slice_design_vector(new_design_vector)

        # TODO Assert len of objects and vectors match

        for obj, new_design_vector in zip(self.design_vector_objects, new_design_vectors):
            obj.update_positions(new_design_vector)

        # Is there some class-specific logic?
        # for obj in self.components:
        #     pass
        #
        # for obj in self.interconnect_nodes:
        #     pass
        #
        # for obj in self.interconnects:
        #     pass

    def get_objective(self):
        pass

    def plot_layout(self):
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

        plot(layout_plot_dict)
