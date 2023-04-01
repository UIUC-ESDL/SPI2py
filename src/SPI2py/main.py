"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""

import json

from datetime import datetime

from .data import Data
from .analysis import signed_distances, format_constraints

from .data.models.systems import System, SpatialConfiguration
from .analysis import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
from .analysis import normalized_aggregate_gap_distance
from .optimize.solvers import run_optimizer
from .result.visualization.animation import generate_gif




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
                 directory):

        # Initialize the Data class
        Data.__init__(self, directory)

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









