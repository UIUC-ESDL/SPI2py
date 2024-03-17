import torch
from torch.func import jacrev
from openmdao.api import ExplicitComponent
from ..models.geometry.bounding_box_volume import bounding_box_bounds_points, bounding_box_volume


class BoundingBoxVolume(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_points_per_object', types=list, desc='Number of points per object')

    def setup(self):
        n_i = self.options['n_points_per_object']
        n = sum(n_i)

        self.add_input('points', shape=(n, 3))
        self.add_output('bounding_box_volume', shape=(1,))
        self.add_output('bounding_box_bounds', shape=(6,))

    def setup_partials(self):
        self.declare_partials('bounding_box_volume', 'points')

    def compute(self, inputs, outputs):

        # Get the input variables
        positions = inputs['points']

        # Convert the inputs to torch tensors
        positions = torch.tensor(positions, dtype=torch.float64)

        # Calculate the bounding box volume
        bb_volume, bb_bounds = self._bounding_box_volume(positions)

        # Set the outputs
        outputs['bounding_box_bounds'] = bb_bounds
        outputs['bounding_box_volume'] = bb_volume

    def compute_partials(self, inputs, partials):

        # Get the input variables
        positions = inputs['points']

        # Convert the inputs to torch tensors
        positions = torch.tensor(positions, dtype=torch.float64, requires_grad=True)

        # Define the Jacobian matrices using PyTorch Autograd
        jac_bbv = jacrev(self._bounding_box_volume_no_bounds, argnums=(0))

        # Evaluate the Jacobian matrices
        jac_bbv_val = jac_bbv(positions)

        # Convert the outputs to numpy arrays
        jac_bbv_positions = jac_bbv_val.detach().numpy()

        # Set the outputs
        partials['bounding_box_volume', 'points'] = jac_bbv_positions

    @staticmethod
    def _bounding_box_volume(positions):
        bb_bounds = bounding_box_bounds_points(positions)
        bb_volume = bounding_box_volume(bb_bounds)
        return bb_volume, bb_bounds

    @staticmethod
    def _bounding_box_volume_no_bounds(positions):
        bb_bounds = bounding_box_bounds_points(positions)
        bb_volume = bounding_box_volume(bb_bounds)
        return bb_volume