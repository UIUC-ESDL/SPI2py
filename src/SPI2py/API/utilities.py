import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent


class Multiplexer(ExplicitComponent):
    """
    An ExplicitComponent that vertically stacks a series of inputs, where each input has shape (n_i, m).
    The output will have shape (sum(n_i), m). Partial derivatives are specified as block-diagonal identities,
    using a sparse (hard-coded) approach.

    Options
    -------
    n_i : list
        A list of row counts for each input.
    m : int
        Number of columns for each input (e.g., 1 for 1D signals, 3 for 3D vectors).
    """

    def initialize(self):
        self.options.declare('n_i', types=list, desc='List of row counts for each input.')
        self.options.declare('m', types=int, desc='Number of columns per input.')

    def setup(self):
        n_i = self.options['n_i']
        m = self.options['m']
        total = sum(n_i)
        for i, n in enumerate(n_i):
            self.add_input(f'input_{i}', shape=(n, m))
        self.add_output('stacked_output', shape=(total, m))

    def compute(self, inputs, outputs):
        # Gather all inputs and vertically stack them.
        arrays = [inputs[f'input_{i}'] for i in range(len(self.options['n_i']))]
        outputs['stacked_output'] = np.vstack(arrays)

    def setup_partials(self):
        n_i = self.options['n_i']
        m = self.options['m']

        # The global output is the vertical stack of all flattened inputs.
        # Total number of scalar entries in the output
        total = sum(n_i) * m
        offset = 0
        for i, n in enumerate(n_i):
            # Flattened size of input_i
            block_size = n * m

            # Create a full zero matrix for the Jacobian block of shape (total, block_size).
            # Only a small block will be nonzero.
            J = np.zeros((total, block_size))

            # For the current input, only the rows corresponding to its block (from offset to offset+block_size)
            # are active, and in that block the derivative is an identity.
            J[offset:offset + block_size, :] = np.eye(block_size)

            # Extract nonzero indices.
            rows, cols = np.nonzero(J)

            # Declare the sparse partial derivative.
            self.declare_partials('stacked_output', f'input_{i}', rows=rows, cols=cols, val=1.0)
            offset += block_size



