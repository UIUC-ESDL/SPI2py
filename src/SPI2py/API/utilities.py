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
        total = sum(n_i) * m  # total number of scalar entries in the output
        offset = 0
        for i, n in enumerate(n_i):
            block_size = n * m  # flattened size of input_i
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


# class Multiplexer(ExplicitComponent):
#     """
#     An ExplicitComponent that vertically stacks a series of inputs with sizes (n, 3) or (n, 1).
#     """
#
#     def initialize(self):
#         # Initialize with a list of sizes for each input
#         self.options.declare('n_i', types=list, desc='The number of spheres for each component')
#         self.options.declare('m', types=int, desc='Column size, either 1 or 3')
#
#     def setup(self):
#         n_i = self.options['n_i']
#         n = sum(n_i)
#         m = self.options['m']
#
#         # Define inputs and output
#         for i, size in enumerate(n_i):
#             self.add_input(f'input_{i}', shape=(size, m))
#
#         self.add_output('stacked_output', shape=(n, m))
#
#     def setup_partials(self):
#         self.declare_partials('*', '*')
#
#     def compute(self, inputs, outputs):
#
#         # Get the options
#         n_i = self.options['n_i']
#         m = self.options['m']  # FOR DEBUGGING
#
#         # Get the input arrays
#         input_arrays = ()
#         for i in range(len(n_i)):
#             input_arrays = input_arrays + (inputs[f'input_{i}'],)
#
#         # Convert the input arrays to torch tensors
#         input_tensors = ()
#         for input_array in input_arrays:
#             input_tensors = input_tensors + (jnp.array(input_array),)
#
#         # Stack inputs vertically
#         stacked_output = self._multiplex(*input_tensors)
#
#         # Set the output
#         outputs['stacked_output'] = stacked_output
#
#     def compute_partials(self, inputs, partials):
#
#         # Get the options
#         n_i = self.options['n_i']
#
#         # Get the input arrays
#         input_arrays = ()
#         for i in range(len(n_i)):
#             input_arrays = input_arrays + (inputs[f'input_{i}'],)
#
#         # Convert the input arrays to torch tensors
#         input_tensors = ()
#         for input_array in input_arrays:
#             input_tensors = input_tensors + (jnp.array(input_array),)
#
#         # Calculate the partial derivatives wrt all inputs
#         argnums = tuple(range(len(n_i)))
#         jac_stacked_output = jacfwd(self._multiplex, argnums=argnums)(*input_tensors)
#
#         # Convert the partial derivatives to numpy arrays
#         jac_stacked_output_np = []
#         for jac in jac_stacked_output:
#             jac_stacked_output_np.append(jac)
#
#         # Set the partial derivatives
#         for i in range(len(n_i)):
#             partials['stacked_output', f'input_{i}'] = jac_stacked_output_np[i]
#
#
#     @staticmethod
#     def _multiplex(*args):
#         return jnp.vstack(args)



