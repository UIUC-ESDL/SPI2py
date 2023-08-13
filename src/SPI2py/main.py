"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""

import os
import yaml

from SPI2py.group_model.component_spatial.bounding_volumes import bounding_box


# TODO Import the model

# Data Import


# Layout Imports

# Analysis Imports


# Optimize Imports
from .drivers.optimize.solvers import run_optimizer

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


        # Initialize the Layout class
        self.spatial_configuration = None

        # Initialize the Optimize class
        self._spatial_configuration = None

        self.initial_design_vectors = {}


    def __repr__(self):
        return str(self.study_name)


    def add_system(self, system):
        self.system = system

    # LAYOUT METHODS

    def set_initial_position(self, object_name, spatial_config_name, design_vector):

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



        design_vector_dict = self.initial_design_vectors[name]

        objects_dict = self.system.calculate_positions(design_vector_dict=design_vector_dict)
        self.system.set_positions(objects_dict=objects_dict)



        self.spatial_configuration = self.system


    # OPTIMIZE METHODS


    def optimize_spatial_configuration(self, options: dict):

        # TODO Remove objective indexing...
        self.result, self.design_vector_log = run_optimizer(self.system,
                                                            self.system.objective,
                                                            self.system.constraints,
                                                            options)






