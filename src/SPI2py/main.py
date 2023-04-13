"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""
import json
from datetime import datetime

import logging
import os
import numpy as np

import yaml
from scipy.optimize import NonlinearConstraint

# TODO Import the model

# Data Import


# Layout Imports

# Analysis Imports
from .computational_model.geometry.distance import normalized_aggregate_gap_distance, signed_distances
from .computational_model.visualization.plotting import plot_3d

from .driver.analysis.constraint_aggregation import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
from .driver.analysis import scale_model_based_objective
# from analysis.kinematics import ...

# Optimize Imports
from .driver.optimize.solvers import run_optimizer

# Result Imports



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

        self.objectives = []
        self.constraints = []
        self.constraint_functions = []

        # Initialize the Optimize class
        self._spatial_configuration = None

        # Initialize the Result class
        self.outputs = {}

        self.initial_design_vectors = {}


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

    def add_initial_design_vector(self, object_name, spatial_config_name, design_vector):

        if spatial_config_name not in list(self.initial_design_vectors.keys()):
            self.initial_design_vectors[spatial_config_name] = {}

        self.initial_design_vectors[spatial_config_name][object_name] = design_vector

    def generate_spatial_configuration(self, name, method):
        """
        Map the objects to a 3D layout.

        First, map static objects to the layout since their positions are independent of the layout generation method.

        :param method:
        :param inputs:

        TODO implement different layout generation methods
        """

        if method == 'manual':

            design_vector_dict = self.initial_design_vectors[name]

            positions_dict = self.system.calculate_positions(design_vector_dict=design_vector_dict)
            self.system.set_positions(positions_dict)

        else:
            raise NotImplementedError

        self.spatial_configuration = self.system


    # ANALYSIS METHODS

    def add_objective(self,
                      objective,
                      model,
                      options):

        """
        Add an objective to the design study.

        TODO Move objective to the model module...?

        :param objective: The objective function to be added.
        :param options: The options for the objective function.
        """

        # UNPACK THE OPTIONS

        design_vector_scaling_type   = options['design vector scaling type']
        design_vector_scaling_factor = options['design vector scaling factor']
        objective_scaling_type       = options['objective scaling type']
        objective_scaling_factor     = options['objective scaling factor']


        # SELECT THE OBJECTIVE FUNCTION HANDLE

        if objective == 'normalized aggregate gap distance':
            _objective_function = normalized_aggregate_gap_distance
        else:
            raise NotImplementedError


        # SCALE THE OBJECTIVE FUNCTION


        def objective_function(x):
            return scale_model_based_objective(x, _objective_function, model,
                                               design_vector_scale_type=design_vector_scaling_type,
                                               design_vector_scale_factor=design_vector_scaling_factor,
                                               objective_scale_type=objective_scaling_type,
                                               objective_scale_factor=objective_scaling_factor)

        self.objectives.append(objective_function)

    def add_constraint(self,
                       constraint,
                       model,
                       options):

        """
        Add a constraint to the design study.
        """

        # UNPACK THE OPTIONS
        type = options['type']
        object_class_1 = options['object class 1']
        object_class_2 = options['object class 2']
        constraint_tolerance = options['constraint tolerance']
        constraint_aggregation = options['constraint aggregation']
        constraint_aggregation_parameter = options['constraint aggregation parameter']

        # SELECT THE OBJECT PAIR

        if object_class_1 == 'component' and object_class_2 == 'component':
            object_pair = model.component_component_pairs
        elif object_class_1 == 'component' and object_class_2 == 'interconnect' or \
                object_class_1 == 'interconnect' and object_class_2 == 'component':
            object_pair = model.component_interconnect_pairs
        elif object_class_1 == 'interconnect' and object_class_2 == 'interconnect':
            object_pair = model.interconnect_interconnect_pairs
        else:
            raise NotImplementedError

        # SELECT THE CONSTRAINT FUNCTION HANDLE
        if constraint == 'signed distances':
            def _constraint_function(x): return signed_distances(x, model, object_pair)
        else:
            raise NotImplementedError

        # SELECT THE CONSTRAINT AGGREGATION FUNCTION HANDLE
        if constraint_aggregation is None:
            pass
        elif constraint_aggregation == 'kreisselmeier steinhauser':
            _constraint_aggregation_function = kreisselmeier_steinhauser
        elif constraint_aggregation == 'P-norm':
            _constraint_aggregation_function = p_norm
        elif constraint_aggregation == 'induced exponential':
            _constraint_aggregation_function = induced_exponential
        elif constraint_aggregation == 'induced power':
            _constraint_aggregation_function = induced_power
        else:
            raise NotImplementedError

        # TODO SCALE THE CONSTRAINT FUNCTION

        if constraint_aggregation is None:
            nlc = NonlinearConstraint(_constraint_function, -np.inf, constraint_tolerance)
            self.constraint_functions.append(_constraint_function)
            self.constraints.append(nlc)
        else:
            def constraint_aggregation_function(x):
                return _constraint_aggregation_function(_constraint_function(x), rho=constraint_aggregation_parameter)
            nlc = NonlinearConstraint(constraint_aggregation_function, -np.inf, constraint_tolerance)
            self.constraint_functions.append(constraint_aggregation_function)
            self.constraints.append(nlc)


    def calculate_metrics(self, x):
        """
        Calculate the objective function and constraint functions.

        """
        objective = self.objective_function(x)

        constraints = [constraint_function(x) for constraint_function in self.constraint_functions]

        return objective, constraints

    def calculate_objective_function(self):
        pass

    def calculate_constraint_functions(self):
        pass

    # OPTIMIZE METHODS


    def optimize_spatial_configuration(self, options: dict):

        # TODO Remove objective indexing...
        self.result, self.design_vector_log = run_optimizer(self.system,
                                                            self.objectives[0],
                                                            self.constraints,
                                                            options)

    # RESULT METHODS


    def plot(self):
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

        fig = plot_3d(positions, radii, colors)

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





