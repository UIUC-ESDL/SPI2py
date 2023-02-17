"""

"""

import json
import logging
from datetime import datetime

import yaml

from .data.spherical_decomposition import generate_rectangular_prisms
from .data.classes.class_constructors import create_components, create_ports, create_interconnects, create_structures
from .data.classes.spatial import SpatialConfiguration

from .layout.generation_methods import generate_random_layout

from .optimization.solvers import gradient_based_optimization

from .result.visualization.visualization import generate_gif



class SPI2:
    """
    The SPI2 Class provides the user with a means to interact with the API...
    """

    def __init__(self, directory, config_file, input_file):

        self.directory = directory
        self.config = self.add_configuration_file(config_file)
        self.inputs = self.add_input_file(input_file)


        self.layout = None

        self.result = None
        self.design_vector_log = None

        self.outputs = None

        self.initialize_logger()

        self.create_objects_from_input()

    def add_directory(self, directory):
        self.directory = directory

    def add_configuration_file(self, config_filename):
        config_filepath = self.directory + config_filename
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def add_input_file(self, input_filename):
        input_filepath = self.directory + input_filename
        with open(input_filepath, 'r') as f:
            inputs = yaml.safe_load(f)
        return inputs

    def initialize_logger(self):
        logger_name = self.directory + self.config['results']['Log Filepath']
        logging.basicConfig(filename=logger_name, encoding='utf-8', level=logging.INFO, filemode='w')


    def create_objects_from_input(self):
        """

        :param inputs:
        :return:
        """

        # Create Components
        components = create_components(self.inputs['components'])

        ports = create_ports(self.inputs['ports'])

        # Create Interconnects
        interconnects, interconnect_nodes, interconnect_segments = create_interconnects(self.inputs['interconnects'])

        # Create Structures
        structures = create_structures(self.inputs['structures'])

        # Generate SpatialConfiguration
        self.layout = SpatialConfiguration(components, ports, interconnects, interconnect_nodes, interconnect_segments, structures)

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
        self.result, self.design_vector_log = gradient_based_optimization(self.layout, self.config['optimization'])

        # Generate GIF
        if self.config['results']['Output GIF'] is True:
            generate_gif(self.layout, self.design_vector_log, 1, self.directory)



    def write_output(self, output_filename):

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


        with open(self.directory + output_filename, 'w') as f:
            json.dump(self.outputs, f)
