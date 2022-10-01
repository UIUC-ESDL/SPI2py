"""Module...
Docstring

To Do:
-Fill out all the blank functions and write tests for them...
-Look into replacing get/set methods with appropriate decorators...


"""

import numpy as np
import jax.numpy as jnp
import networkx as nx
from itertools import product, combinations
from utils.shape_generator import generate_rectangular_prism, generate_rectangular_prisms
from utils.visualization import plot
from utils.transformations import translate, rotate




class Component:

    def __init__(self, positions, radii, color, node, name):

        self.positions = positions
        self.radii = radii
        self.color = color
        self.name = name

        self.node = node

        self.rotation = np.array([0,0,0]) # Initialize the rotation attribute

    @property
    def reference_position(self):
        return self.positions[0]

    def update_positions(self,design_vector, constraint=None):
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
        return jnp.concatenate((self.reference_position, self.rotation))


class InterconnectNode:
    def __init__(self, node, position=jnp.array([0.,0.,0.])):
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
        self.color = color

        # Create edge tuple for NetworkX graphs
        self.edge = (self.component_1, self.component_2)

        # Placeholder for plot test functionality, random positions
        self.positions = np.array([[0, 0, 0]])
        self.radii = np.array([0.5])


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

        # Get possible object collision pairs
        self.component_component_pairs = 1
        self.component_interconnect_pairs = 1
        self.interconnect_interconnect_pairs = 1
        self.structure_all_pairs = 1

        self.all_pairs = self.component_component_pairs+self.component_interconnect_pairs+self.interconnect_interconnect_pairs+self.structure_all_pairs

    def get_component_component_pairs(self):
        pass

    def get_component_interconnect_pairs(self):
        pass

    def get_interconnect_interconnect_pairs(self):
        pass

    def get_structure_all_pairs(self):
        pass

    def generate_random_layout(self):
        """
        Generates random layouts using a force-directed algorithm

        :return:
        """

        g = nx.MultiGraph()
        g.add_nodes_from(self.nodes)
        g.add_edges_from(self.edges)

        # Optimal distance between nodes
        k = 1

        # Dimension of layout
        dim = 3

        positions = nx.spring_layout(g, k=k, dim=dim)

        # Generate random angles too?

        # Now create a positions dictionary


        return positions

    @property
    def design_vector(self):

        design_vector = jnp.empty(0)
        for obj in self.design_vector_objects:
            design_vector = jnp.concatenate((design_vector,obj.design_vector))

        return design_vector

    @property
    def design_vectors(self):

        num_design_vectors = len(self.design_vector_objects)
        design_vectors = []
        for i, obj in enumerate(self.design_vector_objects):
            design_vectors.append(obj.design_vector)

        return design_vectors

    @property
    def reference_positions(self):

        reference_positions_dict = {}

        for obj in self.objects:
            reference_positions_dict[obj] = obj.reference_position

        return reference_positions_dict


    def get_positions(self):
        pass

    def get_radii(self):
        pass

    # @ reference_positions.setter
    # def update_positions(self,design_vector):
    #
    #     for




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
