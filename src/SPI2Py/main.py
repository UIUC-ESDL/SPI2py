"""

"""
import yaml

from .utils.spherical_decomposition.prismatic_shapes import generate_rectangular_prisms
from .utils.objects import Component, InterconnectSegment, Structure, SpatialConfiguration
from .utils.spatial_topologies.force_directed_layouts import generate_random_layout


class SPI2:
    """
    The SPI2 Class provides the user with a means to interact with the API...
    """

    def __init__(self):
        self.config = None
        self.inputs = None

        self.components = None
        self.interconnect_nodes = None
        self.interconnects = None
        self.structures = None

        self.layout = None

    def add_input_file(self, input_filepath):

        with open(input_filepath, 'r') as f:
            self.inputs = yaml.safe_load(f)


    def add_configuration_file(self, config_filepath):

        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)


    def create_objects_from_input(self, inputs):
        """

        :param inputs:
        :return:
        """


        # Create Components
        self.components = []
        for component in inputs['components']:
            node = component
            name = inputs['components'][component]['name']
            color = inputs['components'][component]['color']
            origins = inputs['components'][component]['origins']
            dimensions = inputs['components'][component]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)
            self.components.append(Component(positions, radii, color, node, name))



        # Create Interconnects
        self.interconnects = []
        for interconnect in inputs['interconnects']:
            component_1 = self.components[inputs['interconnects'][interconnect]['component 1']]
            component_2 = self.components[inputs['interconnects'][interconnect]['component 2']]
            diameter = inputs['interconnects'][interconnect]['diameter'][0]
            color = inputs['interconnects'][interconnect]['color']

            self.interconnects.append(InterconnectSegment(component_1, component_2, diameter, color))



        # Create Interconnect Nodes
        self.interconnect_nodes = []
        # for interconnect






        # Create Structures
        self.structures = []
        for structure in inputs['structures']:
            name = inputs['structures'][structure]['name']
            color = inputs['structures'][structure]['color']
            origins = inputs['structures'][structure]['origins']
            dimensions = inputs['structures'][structure]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)
            self.structures.append(Structure(positions, radii, color, name))




    def generate_layout(self, layout_generation_method):

        self.create_objects_from_input(self.inputs)

        self.layout = SpatialConfiguration(self.components, self.interconnect_nodes, self.interconnects, self.structures)

        # TODO implement different layout generation methods

        if layout_generation_method == 'force directed':
            initial_layout_design_vector = generate_random_layout(self.layout)
            self.layout.set_positions(initial_layout_design_vector)

        else:
            print('Sorry, no other layout generation methods are implemented yet')
