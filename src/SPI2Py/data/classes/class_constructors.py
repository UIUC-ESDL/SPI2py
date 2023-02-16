"""

"""

import numpy as np

from ..spherical_decomposition import generate_rectangular_prisms
from .objects import Component, Port, Interconnect, InterconnectNode, InterconnectEdge, Structure

def extract_inputs(): 
    
    def extract_components():
        pass

    def extract_interconnects():
        pass

    def extract_structures():
        pass
    
    pass


def create_components(inputs):

    components = []

    for component in inputs.values():

        # Extract the component's attributes
        name = component['name']
        color = component['color']
        origins = component['origins']
        dimensions = component['dimensions']
        rotation = component['rotation']

        # Generate the component's positions and radii

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        # Create the component
        component = Component(name, positions, rotation, radii, color)
        # logging.info(' Component: ' + str(component) + ' created.')
        components.append(component)

    return components


def create_ports(inputs):

    ports = []

    for port in inputs.values():

        component_name = port['component name']
        port_name = port['port name']
        color = port['color']
        reference_point_offset = port['reference point offset']
        radius = port['radius']

        _port = Port(component_name, port_name, color, reference_point_offset, radius)

        # logging.info(' Port: ' + str(port) + ' created.')
        ports.append(_port)

    return ports


def create_interconnects(inputs):

    interconnects = []
    nodes = []
    interconnect_nodes = []
    interconnect_segments = []

    for interconnect in inputs.values():

        name = interconnect['name']

        component_1 = interconnect['component 1']
        component_1_port = interconnect['component 1 port']

        component_2 = interconnect['component 2']
        component_2_port = interconnect['component 2 port']

        radius = interconnect['radius']
        color = interconnect['color']

        number_of_bends = interconnect['number of bends']

        interconnect = Interconnect(name, component_1, component_1_port, component_2, component_2_port, radius, color, number_of_bends)
        # logging.info(' Interconnect: ' + str(interconnect) + ' created.')

        # Add Interconnect, InterconnectNodes, and InterconnectSegments to lists
        # Use append for single objects and extend for lists of objects
        interconnects.append(interconnect)
        interconnect_segments.extend(interconnect.segments)
        nodes.extend(interconnect.interconnect_nodes)
    
    # TODO Unstrip port nodes?
    interconnect_nodes = nodes

    return interconnects, interconnect_nodes, interconnect_segments

def create_structures(inputs):
    
    structures = []

    for structure in inputs.values():

        name = structure['name']
        color = structure['color']
        origins = structure['origins']
        dimensions = structure['dimensions']
        rotation = structure['rotation']

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        structure = Structure(name, positions, rotation, radii, color)

        # logging.info(' Structure: ' + str(structure) + ' created.')

        structures.append(structure)

    return structures