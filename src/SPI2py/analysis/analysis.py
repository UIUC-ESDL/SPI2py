from .objectives import normalized_aggregate_gap_distance
from .constraints import signed_distances
from .constraint_aggregation import kreisselmeier_steinhauser, p_norm, induced_exponential, induced_power

class Analysis:
    """
    Analysis class for interacting with the SPI2py API.
    """
    def __init__(self):
        self.objective_function = None
        self.constraint_function = None
        self.constraint_aggregation_function = None

    def set_objective_function(self, objective_function):

        # Set the objective function
        if objective_function == 'normalized aggregate gap distance':
            _objective_function = normalized_aggregate_gap_distance
        else:
            raise NotImplementedError

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