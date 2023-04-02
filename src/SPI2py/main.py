"""Main module for the SPI2py package.

The EntryPoint Class provides the user with a means to interact with the SPI2py API.
"""

from .data import Data
from .layout import Layout
from .analysis import Analysis
from .optimize import Optimize
from .result import Result

# TODO Compartmentalize these imports
# from .analysis import signed_distances
# from .analysis import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power
# from .analysis import normalized_aggregate_gap_distance


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

        self.set_objective_function(objective_function)
        self.set_constraint_function(constraint_function)
        self.set_constraint_aggregation_function(constraint_aggregation_function)

        self._optimize_spatial_configuration(self.spatial_configuration,
                                             self.objective_function,
                                             self.constraint_function,
                                             self.constraint_aggregation_function,
                                             self.config)









