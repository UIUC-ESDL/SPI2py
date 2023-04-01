"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""

import json
import logging
import os
from datetime import datetime

import yaml

from .analysis import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
from .analysis import normalized_aggregate_gap_distance
from .analysis import signed_distances, format_constraints
from .data.models.class_constructors import create_components, create_ports, create_interconnects, create_structures
from .data.models.systems import System, SpatialConfiguration
from .optimize.solvers import run_optimizer
from .result.visualization.animation import generate_gif


class Data:
    """
    Data class for interacting with the SPI2py API.
    """

    def __init__(self,
                 directory:  str,
                 input_file: str):

        self.directory  = directory

        # Initialize default configuration
        self._entry_point_directory = os.path.dirname(__file__) + '/'
        self.config                 = self.read_config_file('data/config.yaml')

        self.logger_name            = self.directory + "logger.log"
        self.inputs                 = self.read_input_file(input_file)
        self._component_inputs      = self.inputs['components']
        self._port_inputs           = self.inputs['ports']
        self._interconnect_inputs   = self.inputs['interconnects']
        self._structure_inputs      = self.inputs['structures']

        # Initialize the logger
        self.initialize_logger()

        # Create objects from the input file
        self.system = self.create_system()

    def read_config_file(self, config_filepath):
        config_filepath = self._entry_point_directory + config_filepath
        with open(config_filepath, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def read_input_file(self, input_filename):
        input_filepath = self.directory + input_filename
        with open(input_filepath, 'r') as f:
            inputs = yaml.safe_load(f)
        return inputs

    # def add_object(self, object_type, **kwargs):
    #     """
    #     Add an object to the system.
    #
    #     :param object_type: str
    #     :param object_name: str
    #     :param object_parameters: dict
    #     :return:
    #     """
    #     if object_type == 'component':
    #         self._component_inputs[object_name] = object_parameters
    #     elif object_type == 'port':
    #         self._port_inputs[object_name] = object_parameters
    #     elif object_type == 'interconnect':
    #         self._interconnect_inputs[object_name] = object_parameters
    #     elif object_type == 'structure':
    #         self._structure_inputs[object_name] = object_parameters
    #     else:
    #         raise ValueError('Invalid object type.')
    #
    #     # Update the system
    #     self.system = self.create_system()

    # def create_component(self, component_name, component_parameters):
    #     """
    #     Add a component to the system.
    #
    #     :param component_name: str
    #     :param component_parameters: dict
    #     :return:
    #     """
    #     self._component_inputs[component_name] = component_parameters
    #
    #     # Update the system
    #     self.system = self.create_system()

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_name, encoding='utf-8', level=logging.INFO, filemode='w')

    def print_log(self):
        with open(self.logger_name) as f:
            print(f.read())

    def create_system(self):
        """
        Create the objects from the input files.

        :return:
        """

        components = create_components(self._component_inputs)
        ports = create_ports(self._port_inputs)

        # TODO switch to just interconnects
        interconnects, interconnect_nodes, interconnect_segments = create_interconnects(self._interconnect_inputs)

        structures = create_structures(self._structure_inputs)

        system = System(components, ports, interconnects, interconnect_nodes, interconnect_segments, structures, self.config)

        return system


class Layout:
    """
    Layout class for interacting with the SPI2py API.
    """
    def __init__(self):

        # Systems do not start with a spatial configuration
        self.spatial_configuration = None

    def _create_spatial_configuration(self, system, method, inputs=None):
        """
        Map the objects to a 3D layout.

        First, map static objects to the layout since their positions are independent of the layout generation method.

        :param method:
        :param inputs:

        TODO implement different layout generation methods
        """

        spatial_configuration = SpatialConfiguration(system)

        spatial_configuration.map_static_objects()

        if method == 'manual':
            positions_dict = spatial_configuration.calculate_positions(inputs)
            spatial_configuration.set_positions(positions_dict)

        else:
            raise NotImplementedError

        self.spatial_configuration = spatial_configuration


class Analysis:
    """
    Analysis class for interacting with the SPI2py API.
    """
    pass

    def calculate_metrics(self):
        pass

    def calculate_objective_function(self):
        pass

    def calculate_constraint_functions(self):
        pass


class Optimize:
    """
    Optimize class for interacting with the SPI2py API.
    """
    def __init__(self):
        self._spatial_configuration = None

    def _optimize_spatial_configuration(self,
                                        spatial_configuration,
                                        objective_function,
                                        constraint_function,
                                        constraint_aggregation_function,
                                        config):

        # TODO Switch config to analysis, simplify
        nlcs = format_constraints(spatial_configuration,
                                  constraint_function,
                                  constraint_aggregation_function,
                                  config)

        self.result, self.design_vector_log = run_optimizer(spatial_configuration,
                                                            objective_function,
                                                            nlcs,
                                                            config)


class Result:
    """
    Result class for interacting with the SPI2py API.
    """
    def __init__(self):
        # self.result = None
        # self.design_vector_log = None
        self.outputs = {}

    def create_gif(self):
        gif_filepath = self.config['results']['GIF Filename']
        generate_gif(self.spatial_configuration, self.design_vector_log, 1, self.directory, gif_filepath)

    def create_report(self):

        # Unpack dictionary values
        user_name = self.config['Username']
        problem_description = self.config['Problem Description']
        report_filename = self.config['results']['Report Filename']

        # Create a timestamp
        now = datetime.now()
        now_formatted = now.strftime("%d/%m/%Y %H:%M:%S")

        # Convert the design vector log of a list of arrays of list to lists
        # json cannot serialize numpy arrays
        design_vector_log = [log.tolist() for log in self.design_vector_log]

        # TODO Merge results instead of overwriting self.outputs
        # Create the output dictionary
        self.outputs = {'Username': user_name,
                        'Date and time': now_formatted,
                        'Problem Description': problem_description,
                        'Comments': 'Placeholder',
                        'Design vector log': design_vector_log}


        with open(self.directory + report_filename, 'w') as f:
            json.dump(self.outputs, f)


class EntryPoint(Data, Layout, Analysis, Optimize, Result):
    """
    EntryPoint class for interacting with the SPI2py API.
    """
    def __init__(self,
                 directory,
                 input_file):

        # Initialize the Data class
        Data.__init__(self, directory, input_file)

        # Initialize the Layout class
        Layout.__init__(self)

        # Initialize the Analysis class
        Analysis.__init__(self)

        # Initialize the Optimize class
        Optimize.__init__(self)

        # Initialize the Result class
        Result.__init__(self)

    # Define Methods

    def create_spatial_configuration(self, method, inputs=None):
        self._create_spatial_configuration(self.system, method, inputs)

    def optimize_spatial_configuration(self,
                                       objective_function,
                                       constraint_function,
                                       constraint_aggregation_function):

        # Set the objective function
        if objective_function == 'normalized aggregate gap distance':
            _objective_function = normalized_aggregate_gap_distance
        else:
            raise NotImplementedError

        # Set the constraint function
        if constraint_function == 'signed distances':
            _constraint_function = signed_distances
        else:
            raise NotImplementedError

        # Set the constraint aggregation function if applicable
        if constraint_aggregation_function == 'kreisselmeier steinhauser':
            _constraint_aggregation_function = kreisselmeier_steinhauser

        elif constraint_aggregation_function == 'P-norm':
            _constraint_aggregation_function = p_norm

        elif constraint_aggregation_function == 'induced exponential':
            _constraint_aggregation_function = induced_exponential

        elif constraint_aggregation_function == 'induced power':
            _constraint_aggregation_function = induced_power

        elif constraint_aggregation_function == 'None':
            # TODO Add the ability to not use a constraint aggregation function
            raise NotImplementedError

        else:
            raise NotImplementedError

        self._optimize_spatial_configuration(self.spatial_configuration,
                                             _objective_function,
                                             _constraint_function,
                                             _constraint_aggregation_function,
                                             self.config)









