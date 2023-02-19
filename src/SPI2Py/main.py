"""

"""

import json
import logging
from datetime import datetime

import yaml

from .data.classes.class_constructors import create_components, create_ports, create_interconnects, create_structures
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
        self.directory = directory
        self.config    = self.read_config_file(config_file)
        self.inputs    = self.read_input_file(input_file)

        # Initialize the logger
        self.initialize_logger()

        # Create objects from the input file
        self.create_objects_from_input()

        # Create the system from the objects

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
        logger_name = self.directory + self.config['results']['Logger Filename']
        logging.basicConfig(filename=logger_name, encoding='utf-8', level=logging.INFO, filemode='w')

    def create_objects_from_input(self):
        """

        :param inputs:
        :return:
        """

        components = create_components(self.inputs['components'])
        ports = create_ports(self.inputs['ports'])
        interconnects, interconnect_nodes, interconnect_segments = create_interconnects(self.inputs['interconnects'])
        structures = create_structures(self.inputs['structures'])

        # Generate SpatialConfiguration
        self.layout = SpatialConfiguration(components, ports, interconnects, interconnect_nodes, interconnect_segments, structures, self.config)

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
            positions_dict = self.layout.calculate_positions(inputs)
            self.layout.set_positions(positions_dict)


        elif layout_generation_method == 'force directed':
            initial_layout_design_vector = generate_random_layout(self.layout)
            self.layout.set_positions(initial_layout_design_vector)

        else:
            print('Sorry, no other layout generation methods are implemented yet')

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
