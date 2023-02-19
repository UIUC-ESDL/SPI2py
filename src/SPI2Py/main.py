"""

"""

import json
import logging
from datetime import datetime

import yaml

from .data.classes.class_constructors import create_components, create_ports, create_interconnects, create_structures
from .data.classes.objects import Component, Port, Interconnect, InterconnectNode, InterconnectEdge, Structure
from .data.classes.systems import SpatialConfiguration

# Import objective and constraint functions
from .analysis.objectives import aggregate_pairwise_distance
from .analysis.constraints import max_interference

from .layout.generation_methods import generate_random_layout

from .optimization.solvers import run_optimizer

from .result.visualization.animation import generate_gif


class EntryPoint:
    """
    The SPI2 Class provides the user with a means to interact with the API...
    """

    def __init__(self, directory, config_file, input_file):

        # Initialize the parameters
        self.directory              = directory
        self.logger_name            = self.directory + "logger.log"
        self.config                 = self.read_config_file(config_file)
        self.inputs                 = self.read_input_file(input_file)
        self._component_inputs      = self.inputs['components']
        self._port_inputs           = self.inputs['ports']
        self._interconnect_inputs   = self.inputs['interconnects']
        self._structure_inputs      = self.inputs['structures']

        # Initialize the logger
        self.initialize_logger()

        # Create objects from the input file
        self.objects = self.create_objects()

        # Create the system from the objects
        self.layout = self.create_spatial_configuration()

    def read_config_file(self, config_filename):
        config_filepath = self.directory + config_filename
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def read_input_file(self, input_filename):
        input_filepath = self.directory + input_filename
        with open(input_filepath, 'r') as f:
            inputs = yaml.safe_load(f)
        return inputs

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_name, encoding='utf-8', level=logging.INFO, filemode='w')

    def create_objects(self):
        """
        Create the objects from the input files.

        :return:
        """

        objects = []

        components = create_components(self._component_inputs)
        ports = create_ports(self._port_inputs)

        # TODO switch to just interconencts
        interconnects, interconnect_nodes, interconnect_segments = create_interconnects(self._interconnect_inputs)

        structures = create_structures(self._structure_inputs)

        objects = [components, ports, interconnects, interconnect_nodes, interconnect_segments, structures]

        return objects

    def create_spatial_configuration(self):

        # Unpack the objects
        components, ports, interconnects, interconnect_nodes, interconnect_segments, structures = self.objects

        return SpatialConfiguration(components, ports, interconnects, interconnect_nodes, interconnect_segments,
                                           structures, self.config)

    def generate_layout(self, method, inputs=None):
        """
        Map the objects to a 3D layout.

        First, map static objects to the layout since their positions are independent of the layout generation method.

        :param layout_generation_method:
        :param inputs:


        TODO implement different layout generation methods
        """

        self.layout.map_static_objects()

        if method == 'manual':
            positions_dict = self.layout.calculate_positions(inputs)
            self.layout.set_positions(positions_dict)

        else:
            raise NotImplementedError


    def optimize_spatial_configuration(self):

        # TODO Add ability to choose objective function
        objective_function = aggregate_pairwise_distance

        # TODO Add ability to choose constraint functions
        constraint_function = max_interference

        self.result, self.design_vector_log = run_optimizer(self.layout,
                                                            objective_function,
                                                            constraint_function,
                                                            self.config['optimization'])

    def create_gif_animation(self):
        gif_filepath = self.config['results']['GIF Filename']
        generate_gif(self.layout, self.design_vector_log, 1, self.directory, gif_filepath)

    def create_report(self):

        # Unpack dictionary values
        user_name = self.config['User Name']
        problem_description = self.config['Problem Description']
        report_filename = self.config['results']['Report Filename']

        # Create a timestamp
        now = datetime.now()
        now_formatted = now.strftime("%d/%m/%Y %H:%M:%S")

        # TODO Create a prompt to ask user for comments on the results

        # Convert the design vector log of a list of arrays of list to lists
        # json cannot serialize numpy arrays
        design_vector_log = [log.tolist() for log in self.design_vector_log]

        # Create the output dictionary
        self.outputs = {'User Name': user_name,
                        'Date and time': now_formatted,
                        'Problem Description': problem_description,
                        'Comments': 'Placeholder',
                        'Design vector log': design_vector_log}


        with open(self.directory + report_filename, 'w') as f:
            json.dump(self.outputs, f)
