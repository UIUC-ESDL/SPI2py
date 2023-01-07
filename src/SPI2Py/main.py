"""

"""

import json
import logging
from datetime import datetime

import yaml

from .optimization.solvers import gradient_based_optimization
from .result.visualization.visualization import generate_gif
from .data.spherical_decomposition import generate_rectangular_prisms
from .data.objects.static_objects import Structure
from .data.objects.dynamic_objects import Component, Interconnect
from .layout.spatial_configuration import SpatialConfiguration
from .layout.generation_methods import generate_random_layout


class SPI2:
    """
    The SPI2 Class provides the user with a means to interact with the API...
    """

    def __init__(self):
        self.directory = None
        self.logger = None
        self.config = None
        self.inputs = None

        self.components = None
        self.interconnects = None
        self.layout = None

        self.result = None
        self.design_vector_log = None

        self.outputs = None

    def add_directory(self, directory):
        self.directory = directory

    def initialize_logger(self):
        logging.basicConfig(filename= self.directory + 'output.log', encoding='utf-8', level=logging.INFO, filemode='w')

    def add_input_file(self, input_filename):
        input_filepath = self.directory + input_filename
        with open(input_filepath, 'r') as f:
            self.inputs = yaml.safe_load(f)

    def add_configuration_file(self, config_filename):
        config_filepath = self.directory + config_filename
        with open(config_filepath, 'r') as f:
            self.config = yaml.safe_load(f)

    def create_components(self):
        components = []
        for component in self.inputs['components']:
            node = component
            name = self.inputs['components'][component]['name']
            color = self.inputs['components'][component]['color']
            origins = self.inputs['components'][component]['origins']
            dimensions = self.inputs['components'][component]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)

            component= Component(positions, radii, color, node, name)
            logging.info(' Component: ' + str(component) + ' created.')

            components.append(component)

        self.components = components

    def create_interconnects(self):
        interconnects = []
        interconnect_nodes = []
        interconnect_segments = []

        for interconnect in self.inputs['interconnects']:

            component_1 = self.components[self.inputs['interconnects'][interconnect]['component 1']]
            component_2 = self.components[self.inputs['interconnects'][interconnect]['component 2']]

            diameter = self.inputs['interconnects'][interconnect]['diameter']
            color = self.inputs['interconnects'][interconnect]['color']

            # Keep this line, if swapped then it works with force-directed layout
            # self.interconnect_segments.append(InterconnectSegment(component_1, component_2, diameter, color))

            interconnect = Interconnect(component_1, component_2, diameter, color)
            logging.info(' Interconnect: ' + str(interconnect) + ' created.')

            interconnects.append(interconnect)

        for interconnect in interconnects:

            interconnect_segments.extend(interconnect.segments)

            # TODO Remove component nodes???
            interconnect_nodes.extend(interconnect.nodes)

        self.interconnects = interconnects
        # TODO Unstrip port nodes?
        self.interconnect_nodes = interconnect_nodes[1:-1] # Strip port nodes?
        self.interconnect_segments = interconnect_segments

    def create_structures(self):
        structures = []
        for structure in self.inputs['structures']:
            name = self.inputs['structures'][structure]['name']
            color = self.inputs['structures'][structure]['color']
            origins = self.inputs['structures'][structure]['origins']
            dimensions = self.inputs['structures'][structure]['dimensions']

            positions, radii = generate_rectangular_prisms(origins, dimensions)

            structure = Structure(positions, radii, color, name)

            logging.info(' Structure: ' + str(structure) + ' created.')

            structures.append(structure)

        self.structures = structures

    def create_objects_from_input(self):
        """

        :param inputs:
        :return:
        """

        # Create Components
        self.create_components()

        # Create Interconnects
        self.create_interconnects()

        # Create Structures
        self.create_structures()

        # Generate SpatialConfiguration
        # slicing nodes for temp fix
        self.layout = SpatialConfiguration(self.components, self.interconnect_nodes, self.interconnect_segments, self.structures)

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


        # TODO implement different layout generation methods

        if layout_generation_method == 'manual':
            self.layout.set_positions(inputs)


        elif layout_generation_method == 'force directed':
            initial_layout_design_vector = generate_random_layout(self.layout)
            self.layout.set_positions(initial_layout_design_vector)

        else:
            print('Sorry, no other layout generation methods are implemented yet')







    def optimize_spatial_configuration(self):
        self.result, self.design_vector_log = gradient_based_optimization(self.layout)

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
