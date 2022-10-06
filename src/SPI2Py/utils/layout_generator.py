"""

"""
import networkx as nx
from utils.objects import Component, Interconnect, InterconnectNode, Structure, Layout
from utils.shape_generator import generate_rectangular_prisms


def create_objects_from_input(inputs):
    """

    :param inputs:
    :return:
    """


    # Create Components
    components = []
    for component in inputs['components']:
        node = component
        name = inputs['components'][component]['name']
        color = inputs['components'][component]['color']
        origins = inputs['components'][component]['origins']
        dimensions = inputs['components'][component]['dimensions']

        positions, radii = generate_rectangular_prisms(origins, dimensions)
        components.append(Component(positions, radii, color, node, name))


    # Create Interconnect Nodes
    interconnect_nodes = []
    # Add more...


    # Create Interconnects
    interconnects = []
    for interconnect in inputs['interconnects']:
        component_1 = inputs['interconnects'][interconnect]['component 1']
        component_2 = inputs['interconnects'][interconnect]['component 2']
        diameter = inputs['interconnects'][interconnect]['diameter']
        color = inputs['interconnects'][interconnect]['color']

        interconnects.append(Interconnect(component_1, component_2, diameter, color))


    # Create Structures
    structures = []
    for structure in inputs['structures']:
        name = inputs['structures'][structure]['name']
        color = inputs['structures'][structure]['color']
        origins = inputs['structures'][structure]['origins']
        dimensions = inputs['structures'][structure]['dimensions']

        positions, radii = generate_rectangular_prisms(origins, dimensions)
        structures.append(Structure(positions, radii, color, name))

    return components, interconnect_nodes, interconnects, structures


def generate_layout(inputs):

    components, interconnect_nodes, interconnects, structures = create_objects_from_input(inputs)

    layout = Layout(components, interconnect_nodes, interconnects, structures)

    return layout


def generate_random_positions():
    # Maybe we want this object oriented?
    pass
