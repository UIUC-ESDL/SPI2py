#TODO Compartmentalize this import
from ..analysis import signed_distances, format_constraints

from .solvers import run_optimizer


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