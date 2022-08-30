"""

"""

from utils.objects import Component, Interconnect, InterconnectNode, Structure, Layout
from utils.shape_generator import generate_rectangular_prisms


def generate_layout(inputs):
    components = []
    for component in inputs['components']:
        node = component
        name = inputs['components'][component]['name']
        color = inputs['components'][component]['color']
        origins = inputs['components'][component]['origins']
        dimensions = inputs['components'][component]['dimensions']
        diameters = inputs['components'][component]['diameters']

        positions, radii = generate_rectangular_prisms(origins, dimensions, diameters)
        components.append(Component(node, name, color, positions, radii))

    interconnect_nodes = []
    # Add more...

    interconnects = []
    for interconnect in inputs['interconnects']:
        component_1 = inputs['interconnects'][interconnect]['component 1']
        component_2 = inputs['interconnects'][interconnect]['component 2']
        diameter = inputs['interconnects'][interconnect]['diameter']

        interconnects.append(Interconnect(component_1, component_2, diameter))

    structures = []
    for structure in inputs['structures']:
        origins = inputs['structures'][structure]['origins']
        dimensions = inputs['structures'][structure]['dimensions']
        diameters = inputs['structures'][structure]['diameters']

        positions, radii = generate_rectangular_prisms(origins, dimensions, diameters)
        structures.append(Structure(positions, radii))

    layout = Layout(components, interconnect_nodes, interconnects, structures)

    return layout


