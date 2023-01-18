"""

"""

from ..spherical_decomposition import generate_rectangular_prisms
from .objects import Component, Interconnect, Structure, Port

def create_components(inputs):

    components = []

    for component in inputs['components']:

        # Extract the component's attributes
        node = component
        name = inputs['components'][component]['name']
        color = inputs['components'][component]['color']
        origins = inputs['components'][component]['origins']
        dimensions = inputs['components'][component]['dimensions']

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        # Extract the ports
        # if inputs['components'][component]['ports'] is not None:
        #     ports = inputs['components'][component]['ports']
        #     for port in ports:
        #         port_node = port
        #         port_name = ports[port]['name']
        #         port_color = ports[port]['color']
        #         port_location = ports[port]['location']
        #         port_num_connections = ports[port]['num_connections']

        #         # Add the port to the component
        #         positions.append(port_location)
        #         radii.append(0.01)


        # Create the component
        component = Component(positions, radii, color, node, name)
        # logging.info(' Component: ' + str(component) + ' created.')
        components.append(component)

    return components


def create_interconnects(inputs):

    interconnects = []
    nodes = []
    interconnect_nodes = []
    interconnect_segments = []

    for interconnect in inputs['interconnects']:

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