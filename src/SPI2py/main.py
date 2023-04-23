"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""
import json
from datetime import datetime

import logging
import os


import yaml


# TODO Import the model

# Data Import


# Layout Imports

# Analysis Imports
from .computational_model.geometry.distance import normalized_aggregate_gap_distance
from .computational_model.geometry.discrete_collision_detection import signed_distances

from src.SPI2py.computational_model.analysis.constraint_aggregation import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
from src.SPI2py.computational_model.analysis import scale_model_based_objective


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

            objects_dict = self.system.calculate_positions(design_vector_dict=design_vector_dict)
            self.system.set_positions(objects_dict=objects_dict)

        else:
            raise NotImplementedError

        self.spatial_configuration = self.system


    # ANALYSIS METHODS






    def calculate_objective_function(self):
        pass

    def calculate_constraint_functions(self):
        pass

    # OPTIMIZE METHODS


    def optimize_spatial_configuration(self, options: dict):

        # TODO Remove objective indexing...
        self.result, self.design_vector_log = run_optimizer(self.system,
                                                            self.system.objectives[0],
                                                            self.system.constraints,
                                                            options)

    # RESULT METHODS





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





