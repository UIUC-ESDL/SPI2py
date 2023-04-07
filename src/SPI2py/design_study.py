"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""
import json
from datetime import datetime

import logging
import os

import yaml

# Import the model

from .model import System

# Data Import


# Layout Imports

# Analysis Imports
from .analysis.objectives import normalized_aggregate_gap_distance
from .analysis.constraints import signed_distances, format_constraints
from .analysis.constraint_aggregation import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
from .analysis.scaling import scale_model_based_objective
# from analysis.kinematics import ...

# Optimize Imports
from .optimize.solvers import run_optimizer

# Result Imports
from .result.visualization import plot_objects


class DesignStudy:
    """
    EntryPoint class for interacting with the SPI2py API.
    """
    def __init__(self,
                 directory,
                 study_name):

        # Initialize the Data class
        self.directory = directory

        self.study_name = study_name

        self.system = None

        self._entry_point_directory = os.path.dirname(__file__) + '/'
        self.config = self.read_config_file('config.yaml')

        self.logger_name = self.directory + "logger.log"

        self.initialize_logger()

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

    def __repr__(self):
        return str(self.study_name)

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

    def initialize_logger(self):
        logging.basicConfig(filename=self.logger_name, encoding='utf-8', level=logging.INFO, filemode='w')

    def print_log(self):
        with open(self.logger_name) as f:
            print(f.read())

    def add_system(self, system):
        self.system = system

    # LAYOUT METHODS

    def _create_spatial_configuration(self, system, method, inputs=None):
        """
        Map the objects to a 3D layout.

        First, map static objects to the layout since their positions are independent of the layout generation method.

        :param method:
        :param inputs:

        TODO implement different layout generation methods
        """

        spatial_configuration = system

        # spatial_configuration.map_static_objects()

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
                               model_based_objective_function,
                               design_vector_scale_factor=1,
                               design_vector_scale_type='constant',
                               objective_scale_factor=1,
                               objective_scale_type='constant'):


        # SELECT THE OBJECTIVE FUNCTION

        if model_based_objective_function == 'normalized aggregate gap distance':
            _objective_function = normalized_aggregate_gap_distance
        else:
            raise NotImplementedError


        # SCALE THE OBJECTIVE FUNCTION

        def objective_function(x):
            return scale_model_based_objective(x, _objective_function, self.system,
                                               design_vector_scale_factor=design_vector_scale_factor,
                                               design_vector_scale_type=design_vector_scale_type,
                                               objective_scale_factor=objective_scale_factor,
                                               objective_scale_type=objective_scale_type)

        self.objective_function = objective_function

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


    def optimize_spatial_configuration(self,
                                       objective_function,
                                       constraint_function,
                                       constraint_aggregation_function,
                                       options
                                       ):



        self.set_objective_function(objective_function,
                                    design_vector_scale_factor=1,
                                    design_vector_scale_type='constant',
                                    objective_scale_factor=0.1,
                                    objective_scale_type='constant')

        self.set_constraint_function(constraint_function)
        self.set_constraint_aggregation_function(constraint_aggregation_function)

        nlcs = format_constraints(self.system,
                                  self.constraint_function,
                                  self.constraint_aggregation_function,
                                  self.config)

        self.result, self.design_vector_log = run_optimizer(self.system,
                                                            self.objective_function,
                                                            nlcs,
                                                            options)

    # RESULT METHODS


    def plot(self, x):
        """
        Plot the model at a given state.

        TODO add option to plot all design vectors
        TODO add option to plot design vector--> move to system object

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

    # def create_gif(self):
    #     gif_filepath = self.config['results']['GIF Filename']
    #     generate_gif(self.spatial_configuration, self.design_vector_log, 1, self.directory, gif_filepath)

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





