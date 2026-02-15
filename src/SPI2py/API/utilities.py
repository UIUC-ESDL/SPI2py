import numpy as np
from openmdao.core.explicitcomponent import ExplicitComponent


class Multiplexer(ExplicitComponent):
    """
    Vertically stack inputs input_i with shape (n_i, m) into stacked_output (sum(n_i), m).
    Jacobian is block identity (sparse) w.r.t. each input.
    """

    def initialize(self):
        self.options.declare('n_i', types=list, desc='List of row counts for each input.')
        self.options.declare('m', types=int, desc='Number of columns per input.')

    def setup(self):
        self.n_i = list(self.options['n_i'])
        self.m = int(self.options['m'])

        total_rows = sum(self.n_i)
        self.add_output('stacked_output', shape=(total_rows, self.m))

        # Precompute output row slices for each input
        self._row_slices = []
        r0 = 0
        for i, n in enumerate(self.n_i):
            self.add_input(f'input_{i}', shape=(n, self.m))
            r1 = r0 + n
            self._row_slices.append(slice(r0, r1))
            r0 = r1

    def compute(self, inputs, outputs):
        y = outputs['stacked_output']
        for i, slc in enumerate(self._row_slices):
            y[slc, :] = inputs[f'input_{i}']

    def setup_partials(self):
        total = sum(self.n_i) * self.m  # flattened output size
        offset = 0
        for i, n in enumerate(self.n_i):
            block_size = n * self.m

            # Identity mapped into the correct output rows (flattened indexing)
            rows = offset + np.arange(block_size, dtype=int)
            cols = np.arange(block_size, dtype=int)

            self.declare_partials(
                'stacked_output', f'input_{i}',
                rows=rows, cols=cols, val=np.ones(block_size)
            )

            offset += block_size

    def compute_partials(self, inputs, partials):
        # Values are already defined as constant ones so there is no need to compute them here.
        pass





