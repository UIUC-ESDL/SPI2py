"""

"""

from SPI2py.data.shape_generation.spherical_decomposition import generate_rectangular_prisms
from .objects import Component, Port, Interconnect, Structure


def create_components(inputs):

    components = []

    for component in inputs.values():

        # Extract the component's attributes
        name = component['name']
        color = component['color']
        shapes = component['shapes']

        movement_class = component['movement class']


        # Generate the component's positions and radii
        # Temporarily strip away box and rotation since it's all boxes and no rotation
        origins = []
        dimensions = []
        for shape in shapes.values():
            origins.append(shape['origin'])
            dimensions.append(shape['dimensions'])

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        if movement_class != 'static':
            component = Component(name, positions, radii, color, movement_class=movement_class)
        elif movement_class == 'static':
            static_position = component['position']
            component = Component(name, positions, radii, color, movement_class=movement_class, static_position=static_position)
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
        shapes = structure['shapes']
        movement_class = structure['movement class']

        origins = []
        dimensions = []
        for shape in shapes.values():
            origins.append(shape['origin'])
            dimensions.append(shape['dimensions'])

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        if movement_class != 'static':
            structure = Structure(name, positions, radii, color, movement_class=movement_class)
        elif movement_class == 'static':
            static_position = structure['position']
            structure = Structure(name, positions, radii, color, movement_class=movement_class, static_position=static_position)


        # logging.info(' Structure: ' + str(structure) + ' created.')

        structures.append(structure)

    return structures

# def create_system(inputs):
#
#     components = create_components(inputs['components'])
#     ports = create_ports(inputs['ports'])
#     interconnects, interconnect_nodes, interconnect_segments = create_interconnects(inputs['interconnects'])
#     structures = create_structures(inputs['structures'])
#
#     system = System(components, ports, interconnects, interconnect_nodes, interconnect_segments, structures, config)
#
#     return components, ports, interconnects, interconnect_nodes, interconnect_segments, structures