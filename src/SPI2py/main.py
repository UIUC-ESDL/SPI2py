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




    def optimize_spatial_configuration(self, options: dict):

        # TODO Remove objective indexing...
        self.result, self.design_vector_log = run_optimizer(self.system,
                                                            self.system.objective,
                                                            self.system.constraints,
                                                            options)






