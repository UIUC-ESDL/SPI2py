"""

"""

import yaml
import json
from datetime import datetime

from src.SPI2Py.optimization.optimizers.gradient_based_method import optimize
from src.SPI2Py.result.visualization.visualization import generate_gif
from src.SPI2Py.data.spherical_decomposition.prismatic_shapes import generate_rectangular_prisms
from src.SPI2Py.data.classes.objects import Component, InterconnectSegment, Interconnect, Structure
from src.SPI2Py.data.classes.organizational import SpatialConfiguration
from src.SPI2Py.layout.spatial_topologies.force_directed_layouts import generate_random_layout


class SPI2:
    """
    The SPI2 Class provides the user with a means to interact with the API...
    """

    def __init__(self):
        self.config = None
        self.inputs = None

        self.components = None
        self.interconnects = None
        self.interconnect_nodes = None
        self.interconnect_segments = None
        self.structures = None

        self.layout = None

        self.result = None
        self.design_vector_log = None

        self.outputs = None

    def add_input_file(self, input_filepath):

        with open(input_filepath, 'r') as f:
            self.inputs = yaml.safe_load(f)


    def add_configuration_file(self, config_filepath):

        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)


    def create_objects_from_input(self):
        """

        :param inputs:
        :return:
        """

        # Create Components
        self.components = []
        for component in self.inputs['components']:
            node = component
            name = self.inputs['components'][component]['name']
            color = self.inputs['components'][component]['color']
            origins = self.inputs['components'][component]['origins']
            dimensions = self.inputs['components'][component]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)
            self.components.append(Component(positions, radii, color, node, name))



        # Create Interconnects

        self.interconnects = []
        self.interconnect_nodes = []
        self.interconnect_segments = []


        for interconnect in self.inputs['interconnects']:

            component_1 = self.components[self.inputs['interconnects'][interconnect]['component 1']]
            component_2 = self.components[self.inputs['interconnects'][interconnect]['component 2']]

            diameter = self.inputs['interconnects'][interconnect]['diameter']
            color = self.inputs['interconnects'][interconnect]['color']

            self.interconnect_segments.append(InterconnectSegment(component_1, component_2, diameter, color))
            # self.interconnects.append(Interconnect(component_1, component_2, diameter, color))

        for interconnect in self.interconnects:
            print('segments:', interconnect)
            print('segments:', interconnect.segments)

            print('ext')
            self.interconnect_segments.extend(interconnect.segments)

            # TODO Remove component nodes???
            # self.interconnect_nodes.extend(interconnect.nodes)











        # Create Structures
        self.structures = []
        for structure in self.inputs['structures']:
            name = self.inputs['structures'][structure]['name']
            color = self.inputs['structures'][structure]['color']
            origins = self.inputs['structures'][structure]['origins']
            dimensions = self.inputs['structures'][structure]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)
            self.structures.append(Structure(positions, radii, color, name))




    def generate_layout(self, layout_generation_method, inputs=None, include_interconnect_nodes=False):
        """

        Parameters
        ----------
        layout_generation_method
        include_interconnect_nodes: If False then initialize interconnects as straight lines between
        the components after generating their layout. If True then include InterconnectNodes in the layout generation
        method (i.e., interconnects won't necessarily connect components in a straight line).

        Returns
        -------

        """

        # TODO Implement ability to include or exclude interconnect nodes from layout generation methods
        if include_interconnect_nodes:
            pass
        else:
            pass

        self.layout = SpatialConfiguration(self.components, self.interconnect_nodes, self.interconnect_segments, self.structures)

        # TODO implement different layout generation methods

        if layout_generation_method == 'manual':
            # TODO Implement functionality to manually define starting points
            self.layout.set_positions(inputs)


        elif layout_generation_method == 'force directed':
            initial_layout_design_vector = generate_random_layout(self.layout)
            self.layout.set_positions(initial_layout_design_vector)

        else:
            print('Sorry, no other layout generation methods are implemented yet')

    def optimize_spatial_configuration(self):
        self.result, self.design_vector_log = optimize(self.layout)

        # Generate GIF
        if self.config['Visualization']['Output GIF'] is True:
            generate_gif(self.layout, self.design_vector_log, 1, self.config['Outputs']['Folderpath'])

    def write_output(self, output_filepath):

        # Create a timestamp
        now = datetime.now()
        now_formatted = now.strftime("%d/%m/%Y %H:%M:%S")

        # TODO Create a prompt to ask user for comments on the results

        # Create the output dictionary
        self.outputs = {'Placeholder': 1,
                   '': 1,
                   'Date and time': now_formatted,
                   '': 1,
                   'Comments': 'Placeholder'}


        with open(output_filepath, 'w') as f:
            json.dump(self.outputs, f)
