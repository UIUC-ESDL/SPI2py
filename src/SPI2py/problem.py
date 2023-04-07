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
from .analysis.scaling import scale_model_based_objective
# from src.SPI2py.kinematics import ...

# Optimize Imports
from .optimize.solvers import run_optimizer

# Result Imports
from .result.visualization import plot_objects


class Problem:
    """
    EntryPoint class for interacting with the SPI2py API.
    """
    def __init__(self,
                 directory,
                 system_name):

        # Initialize the Data class
        self.directory = directory
        self.system_name = system_name
        # Initialize default configuration
        self._entry_point_directory = os.path.dirname(__file__) + '/'
        self.config = self.read_config_file('config.yaml')

        self.logger_name = self.directory + "logger.log"

        self.initialize_logger()

        self.system = self.create_system(system_name)

        # Initialize the Layout class
        self.spatial_configuration = None

        # Initialize the Analysis class
        self.objective_function = None
        self.constraint_function = None
        self.constraint_aggregation_function = None

        # Initialize the Optimize class
        self._spatial_configuration = None

        # Initialize the Result class
        self.outputs = {}

    # DATA METHODS

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

    # LAYOUT METHODS

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

    def create_spatial_configuration(self, method, inputs=None):
        self._create_spatial_configuration(self.system, method, inputs)

    # ANALYSIS METHODS

    def set_objective_function(self,
                               objective_function,
                               objective_scaling=False,
                               design_vector_scale_factor=1,
                               design_vector_scale_type='constant',
                               objective_scale_factor=1,
                               objective_scale_type='constant'):

        # Set the objective function
        if objective_function == 'normalized aggregate gap distance':
            _objective_function = normalized_aggregate_gap_distance
        else:
            raise NotImplementedError

        # if objective_scaling:
        #     _objective_function = scale_model_based_objective(_objective_function)

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

    # OPTIMIZE METHODS

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

    # RESULT METHODS


    def plot(self, x):
        """
        Plot the model at a given state.

        TODO add option to plot all design vectors

        :param x: Design vector
        """

        positions = []
        radii = []
        colors = []

        for obj in self.system.objects:

            positions.append(obj.positions)
            radii.append(obj.radii)
            colors.append(obj.color)

        fig = plot_objects(positions, radii, colors)

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





