"""Module...
Docstring

"""


import numpy as np
import networkx as nx
from itertools import product, combinations
from utils.shape_generator import generate_rectangular_prism, generate_rectangular_prisms
from utils.visualization import plot

class Object:

    # Can I add a component instance counter here? Do I need it?

    def __init__(self, positions, radii, color):
        self.positions = positions
        self.radii = radii
        self.color = color

    def get_positions(self):
        return self.positions

    def get_radii(self):
        return self.radii

    def get_color(self):
        return self.color


class Component(Object):

    def __init__(self, positions, radii, color, node, name):
        super().__init__(positions, radii, color)
        self.node = node
        self.name = name

    def get_node(self):
        return self.node

    def get_design_vector(self):
        pass


class InterconnectNode(Object):
    def __init__(self, node):
        self.node = node

    def get_node(self):
        return self.node


class Interconnect(Object):
    def __init__(self, component_1, component_2, diameter, color):
        self.component_1 = component_1
        self.component_2 = component_2
        self.diameter = diameter
        self.color = color

        # Create edge tuple for NetworkX graphs
        self.edge = (self.component_1, self.component_2)

        # Placeholder for plot test functionality, random positions
        self.positions = np.array([[1, 2, 3]])
        self.radii = np.array([0.5])

    def get_edge(self):
        return self.edge


class Structure(Object):
    pass


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
        self.nodes = [design_object.get_node() for design_object in self.design_objects]
        self.edges = [interconnect.get_edge() for interconnect in self.interconnects]

        #
        self.design_vector_objects = components + interconnect_nodes

        # Get possible collision pairs
        self.component_component_pairs = 1
        self.component_interconnect_pairs = 1
        self.interconnect_interconnect_pairs = 1
        self.structure_all_pairs = 1

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

        return positions

    def get_design_vector(self):
        pass

    def get_positions(self):
        pass

    def get_radii(self):
        pass

    def plot_layout(self):

        layout_plot_dict = {}

        for obj in self.objects:

            object_plot_dict = {}

            positions = obj.get_positions()
            radii = obj.get_radii()
            color = obj.get_color()

            object_plot_dict['positions'] = positions
            object_plot_dict['radii'] = radii
            object_plot_dict['color'] = color

            layout_plot_dict[obj] = object_plot_dict
        print('wait here')
        plot(layout_plot_dict)



        # return layout_plot_dict





