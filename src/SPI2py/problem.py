"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""
import json
from datetime import datetime

import logging
import os

import yaml

# Import the model
from .model_objects import Component, Port, Interconnect, Structure
from .model_systems import System
from .model_spatial_configurations import SpatialConfiguration

# Data Import
from .data import generate_rectangular_prisms

# Layout Imports

# Analysis Imports
from .analysis.objectives import normalized_aggregate_gap_distance
from .analysis.constraints import signed_distances, format_constraints
from .analysis.constraint_aggregation import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
# from src.SPI2py.kinematics import ...

# Optimize Imports
from .optimize.solvers import run_optimizer

# Result Imports
from src.SPI2py.result.animation import generate_gif


class Data:
    """
    Data class for interacting with the SPI2py API.
    """

    def __init__(self,
                 directory:  str,
                 system_name: str):

        self.directory  = directory
        self.system_name = system_name
        # Initialize default configuration
        self._entry_point_directory = os.path.dirname(__file__) + '/'
        self.config                 = self.read_config_file('config.yaml')

        self.logger_name            = self.directory + "logger.log"

        # Initialize the logger
        self.initialize_logger()

        self.system = self.create_system(system_name)

        # Create objects from the input file


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

    def add_component(self,
                      name: str,
                      color: str,
                      movement_class: str,
                      shapes: list):


        """
        Add a component to the system.

        :param name:
        :param color:
        :param movement_class:
        :param shapes:
        :return:
        """

        origins = []
        dimensions = []
        for shape in shapes:
            origins.append(shape['origin'])
            dimensions.append(shape['dimensions'])

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        component = Component(name, positions, radii, color, movement_class=movement_class)

        # Update the system
        self.system.components.append(component)

    def add_port(self, component_name, port_name, color, radius,reference_point_offset, movement_class):
        """
        Add a port to the system.

        :param
        """
        port = Port(component_name, port_name, color, radius,reference_point_offset, movement_class=movement_class)
        self.system.ports.append(port)

    def add_interconnect(self, name, component_1, component_1_port, component_2, component_2_port, radius, color, number_of_bends):
        """
        Add an interconnect to the system.

        """
        interconnect = Interconnect(name, component_1, component_1_port, component_2, component_2_port, radius, color, number_of_bends)

        self.system.interconnects.append(interconnect)
        self.system.interconnect_segments.extend(interconnect.segments)
        self.system.interconnect_nodes.extend(interconnect.interconnect_nodes)

    def add_structure(self, name, color, movement_class, shapes):
        """
        Add a structure to the system.

        """

        origins = []
        dimensions = []
        for shape in shapes:
            origins.append(shape['origin'])
            dimensions.append(shape['dimensions'])

        positions, radii = generate_rectangular_prisms(origins, dimensions)

        structure = Structure(name, positions, radii, color, movement_class)

        self.system.structures.append(structure)

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_name, encoding='utf-8', level=logging.INFO, filemode='w')

    def print_log(self):
        with open(self.logger_name) as f:
            print(f.read())

    def create_system(self, name):
        """
        Create the objects from the input files.

        :return:
        """
        # TODO Replace all with interconnects
        system = System(name, self.config)

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
            # TODO Fix this
            # spatial_configuration.map_static_objects()

            positions_dict = spatial_configuration.calculate_positions(inputs)
            spatial_configuration.set_positions(positions_dict)

        else:
            raise NotImplementedError

        self.spatial_configuration = spatial_configuration


class Analysis:
    """
    Analysis class for interacting with the SPI2py API.
    """
    def __init__(self):
        self.objective_function = None
        self.constraint_function = None
        self.constraint_aggregation_function = None

    def set_objective_function(self, objective_function):

        # Set the objective function
        if objective_function == 'normalized aggregate gap distance':
            _objective_function = normalized_aggregate_gap_distance
        else:
            raise NotImplementedError

        self.objective_function = _objective_function

    def set_constraint_function(self, constraint_function):

        # Set the constraint function
        if constraint_function == 'signed distances':
            _constraint_function = signed_distances
        else:
            raise NotImplementedError

        self.constraint_function = _constraint_function

    def set_constraint_aggregation_function(self, constraint_aggregation_function):

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

        self.constraint_aggregation_function = _constraint_aggregation_function


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
                                        options):

        # TODO Switch config to analysis, simplify
        nlcs = format_constraints(spatial_configuration,
                                  constraint_function,
                                  constraint_aggregation_function,
                                  self.config)

        self.result, self.design_vector_log = run_optimizer(spatial_configuration,
                                                            objective_function,
                                                            nlcs,
                                                            options)

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

class Problem(Data, Layout, Analysis, Optimize, Result):
    """
    EntryPoint class for interacting with the SPI2py API.
    """
    def __init__(self,
                 directory,
                 system_name):

        # Initialize the Data class
        Data.__init__(self, directory, system_name)

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
                                       constraint_aggregation_function,
                                       options
                                       ):

        self.set_objective_function(objective_function)
        self.set_constraint_function(constraint_function)
        self.set_constraint_aggregation_function(constraint_aggregation_function)

        # if options['objective scaling factor'] is not None:


        self._optimize_spatial_configuration(self.spatial_configuration,
                                             self.objective_function,
                                             self.constraint_function,
                                             self.constraint_aggregation_function,
                                             options)









