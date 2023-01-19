"""

"""

import numpy as np

from ..spherical_decomposition import generate_rectangular_prisms
from .objects import Component, Interconnect, Structure

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

        # Extract the port information
        port_names = component['port names']
        port_origins = np.array([component['port origins']]) # TODO Make array creation cleaner
        port_radii = component['port radii']

        # Generate the component's positions and radii

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        # Create the component
        component = Component(name, positions, radii, color, port_names, port_origins, port_radii)
        # logging.info(' Component: ' + str(component) + ' created.')
        components.append(component)

    return components


def create_interconnects(inputs):

    interconnects = []
    nodes = []
    interconnect_nodes = []
    interconnect_segments = []

    for interconnect in inputs['interconnects']:

        # name = inputs

        component_1 = components[inputs['interconnects'][interconnect]['component 1']]
        component_2 = components[inputs['interconnects'][interconnect]['component 2']]

        radius = inputs['interconnects'][interconnect]['radius']
        color = inputs['interconnects'][interconnect]['color']

        interconnect = Interconnect(component_1, component_2, radius, color)
        # logging.info(' Interconnect: ' + str(interconnect) + ' created.')

        # Add Interconnect, InterconnectNodes, and InterconnectSegments to lists
        # Use append for single objects and extend for lists of objects
        interconnects.append(interconnect)
        interconnect_segments.extend(interconnect.segments)
        nodes.extend(interconnect.nodes)
    
    # TODO Unstrip port nodes?
    interconnect_nodes = nodes[1:-1] # Strip the first and last nodes (port nodes)

def create_structures(inputs):
    
    structures = []

    for structure in inputs['structures']:

        name = inputs['structures'][structure]['name']
        color = inputs['structures'][structure]['color']
        origins = inputs['structures'][structure]['origins']
        dimensions = inputs['structures'][structure]['dimensions']

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        structure = Structure(positions, radii, color, name)

        # logging.info(' Structure: ' + str(structure) + ' created.')

        structures.append(structure)

    return structures