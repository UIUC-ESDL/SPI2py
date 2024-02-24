import torch
from openmdao.core.explicitcomponent import ExplicitComponent
from torch.autograd.functional import jacobian
from ..models.utilities.aggregation import kreisselmeier_steinhauser_max, kreisselmeier_steinhauser_min



class Multiplexer(ExplicitComponent):
    """
    An ExplicitComponent that vertically stacks a series of inputs with sizes (n, 3) or (n, 1).

    TODO Explicitly implement compute and compute partials. Block identity matrices don't require autograd.
    """

    def initialize(self):
        # Initialize with a list of sizes for each input
        self.options.declare('input_sizes', types=list, desc='List of sizes (n) for each input array')
        self.options.declare('column_size', types=int, desc='Column size, either 1 or 3')

    def setup(self):
        input_sizes = self.options['input_sizes']
        column_size = self.options['column_size']
        total_rows = sum(input_sizes)

        # Define inputs and output
        for i, size in enumerate(input_sizes):
            self.add_input(f'input_{i}', shape=(size, column_size))

        self.add_output('stacked_output', shape=(total_rows, column_size))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the options
        input_sizes = self.options['input_sizes']

        # Get the input arrays
        input_arrays = ()
        for i in range(len(input_sizes)):
            input_arrays = input_arrays + (inputs[f'input_{i}'],)

        # Convert the input arrays to torch tensors
        input_tensors = ()
        for input_array in input_arrays:
            input_tensors = input_tensors + (torch.tensor(input_array, dtype=torch.float64),)

        # Stack inputs vertically
        stacked_output = self._compute_vstack(*input_tensors)

        # Convert the stacked output to a numpy array
        stacked_output = stacked_output.detach().numpy()

        # Set the output
        outputs['stacked_output'] = stacked_output

    def compute_partials(self, inputs, partials):

        # Get the options
        input_sizes = self.options['input_sizes']

        # Get the input arrays
        input_arrays = ()
        for i in range(len(input_sizes)):
            input_arrays = input_arrays + (inputs[f'input_{i}'],)

        # Convert the input arrays to torch tensors
        input_tensors = ()
        for input_array in input_arrays:
            input_tensors = input_tensors + (torch.tensor(input_array, dtype=torch.float64, requires_grad=True),)

        # Calculate the partial derivatives
        jac_stacked_output = jacobian(self._compute_vstack, input_tensors)

        # Convert the partial derivatives to numpy arrays
        jac_stacked_output_np = []
        for jac in jac_stacked_output:
            jac_stacked_output_np.append(jac.detach().numpy())

        # Set the partial derivatives
        for i in range(len(input_sizes)):
            partials['stacked_output', f'input_{i}'] = jac_stacked_output_np[i]


    @staticmethod
    def _compute_vstack(*args):
        return torch.vstack(args)


class Aggregator(ExplicitComponent):

    def setup(self):

        # Define inputs and output
        self.add_input('input_vector', shape_by_conn=True)
        self.add_output('aggregated_output', shape=(1, 1))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the input arrays
        input_vector = inputs['input_vector']

        # Convert the input arrays to torch tensors
        input_vector = torch.tensor(input_vector, dtype=torch.float64)

        # Stack inputs vertically
        aggregated_output = self._compute_aggregation(input_vector)

        # Convert the stacked output to a numpy array
        aggregated_output = aggregated_output.detach().numpy()

        # Set the output
        outputs['aggregated_output'] = aggregated_output

    def compute_partials(self, inputs, partials):
        # Get the input arrays
        input_vector = inputs['input_vector']

        # Convert the input arrays to torch tensors
        input_vector = torch.tensor(input_vector, dtype=torch.float64)

        # Calculate the partial derivatives
        jac_aggregated_output = jacobian(self._compute_aggregation, input_vector)

        # Convert the partial derivatives to numpy arrays
        jac_aggregated_output_np = jac_aggregated_output.detach().numpy()

        # Set the partial derivatives
        partials['aggregated_output', 'input_vector'] = jac_aggregated_output_np

    @staticmethod
    def _compute_aggregation(input_vector):
        raise NotImplementedError


class MaxAggregator(Aggregator):

    @staticmethod
    def _compute_aggregation(input_vector):
        aggregated_output = kreisselmeier_steinhauser_max(input_vector)
        return aggregated_output


class MinAggregator(Aggregator):

    @staticmethod
    def _compute_aggregation(input_vector):
        aggregated_output = kreisselmeier_steinhauser_min(input_vector)
        return aggregated_output
