import torch
from torch.autograd.functional import jacobian
from openmdao.api import ExplicitComponent
from ..models.geometry.bounding_box_volume import bounding_box_bounds, bounding_box_volume


class BoundingBoxVolume(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_spheres_per_object', types=list, desc='Number of spheres per object')

    def setup(self):
        n_i = self.options['n_spheres_per_object']
        n = sum(n_i)

        self.add_input('positions', shape=(n, 3))
        self.add_input('radii', shape=(n, 1))

        self.add_output('bounding_box_volume', shape=(1,))
        self.add_output('bounding_box_bounds', shape=(6,))

    def setup_partials(self):
        self.declare_partials('bounding_box_volume', 'positions')
        self.declare_partials('bounding_box_volume', 'radii')

    def compute(self, inputs, outputs):

        # Get the input variables
        positions = inputs['positions']
        radii = inputs['radii']

        # Convert the inputs to torch tensors
        positions = torch.tensor(positions, dtype=torch.float64)
        radii = torch.tensor(radii, dtype=torch.float64)

        # Calculate the bounding box volume
        bb_volume, bb_bounds = self._bounding_box_volume(positions, radii)

        # Set the outputs
        outputs['bounding_box_bounds'] = bb_bounds
        outputs['bounding_box_volume'] = bb_volume

    def compute_partials(self, inputs, partials):

        # Get the input variables
        sphere_positions = inputs['positions']
        sphere_radii = inputs['radii']

        # Convert the inputs to torch tensors
        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=True)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64, requires_grad=True)

        # Define the Jacobian with PyTorch Autograd
        jac_bbv = jacobian(self._bounding_box_volume_no_bounds, (sphere_positions, sphere_radii))

        # Convert the outputs to numpy arrays
        jac_bbv_positions = jac_bbv[0].detach().numpy()
        jac_bbv_radii = jac_bbv[1].detach().numpy()

        # Set the outputs
        partials['bounding_box_volume', 'positions'] = jac_bbv_positions
        partials['bounding_box_volume', 'radii'] = jac_bbv_radii

    @staticmethod
    def _bounding_box_volume(positions, radii):
        bb_bounds = bounding_box_bounds(positions, radii)
        bb_volume = bounding_box_volume(bb_bounds)
        return bb_volume, bb_bounds

    @staticmethod
    def _bounding_box_volume_no_bounds(positions, radii):
        bb_bounds = bounding_box_bounds(positions, radii)
        bb_volume = bounding_box_volume(bb_bounds)
        return bb_volume