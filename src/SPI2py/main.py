"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""

from .data import Data
from .layout import Layout
from .analysis import Analysis
from .optimize import Optimize
from .result import Result

# TODO Compartmentalize these imports
from .analysis import signed_distances
from .analysis import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
from .analysis import normalized_aggregate_gap_distance


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









