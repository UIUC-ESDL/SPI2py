# import torch
import jax.numpy as np
from jax import grad, jacfwd, jacrev
from openmdao.api import ExplicitComponent, Group
# from torch.autograd.functional import jacobian

from SPI2py.models.kinematics.distance_calculations import signed_distances_spheres_spheres

class CombinatorialCollisionDetection(Group):
    pass

class PairwiseCollisionDetection(ExplicitComponent):

    # def initialize(self):
        # self.options.declare('components', types=list)
        # self.options.declare('interconnects', types=list)

    def setup(self):
        self.add_input('positions_a', shape_by_conn=True)
        self.add_input('radii_a', shape_by_conn=True)

        self.add_input('positions_b', shape_by_conn=True)
        self.add_input('radii_b', shape_by_conn=True)

        # FIXME Hard-coded
        self.add_output('signed_distances', shape=(10, 10))

    def setup_partials(self):
        self.declare_partials('signed_distances', 'positions_a')
        self.declare_partials('signed_distances', 'radii_a')
        self.declare_partials('signed_distances', 'positions_b')
        self.declare_partials('signed_distances', 'radii_b')

    def compute(self, inputs, outputs):

        # Extract the inputs
        positions_a = inputs['positions_a']
        radii_a = inputs['radii_a']
        positions_b = inputs['positions_b']
        radii_b = inputs['radii_b']

        # Convert the inputs to torch tensors
        # positions_a = torch.tensor(positions_a, dtype=torch.float64)
        # radii_a = torch.tensor(radii_a, dtype=torch.float64)
        # positions_b = torch.tensor(positions_b, dtype=torch.float64)
        # radii_b = torch.tensor(radii_b, dtype=torch.float64)
        positions_a = np.array(positions_a)
        radii_a = np.array(radii_a)
        positions_b = np.array(positions_b)
        radii_b = np.array(radii_b)

        signed_distances = self._compute_signed_distances(positions_a, radii_a, positions_b, radii_b)

        outputs['signed_distances'] = signed_distances

    def compute_partials(self, inputs, partials):

        # Extract the inputs
        positions_a = inputs['positions_a']
        radii_a = inputs['radii_a']
        positions_b = inputs['positions_b']
        radii_b = inputs['radii_b']

        # Convert the inputs to torch tensors
        # positions_a = torch.tensor(positions_a, dtype=torch.float64, requires_grad=True)
        # radii_a = torch.tensor(radii_a, dtype=torch.float64, requires_grad=True)
        # positions_b = torch.tensor(positions_b, dtype=torch.float64, requires_grad=True)
        # radii_b = torch.tensor(radii_b, dtype=torch.float64, requires_grad=True)
        positions_a = np.array(positions_a)
        radii_a = np.array(radii_a)
        positions_b = np.array(positions_b)
        radii_b = np.array(radii_b)

        # Calculate the partial derivatives
        # jac_signed_distances = jacobian(self._compute_signed_distances, (positions_a, radii_a, positions_b, radii_b))
        jac_signed_distances = jacfwd(self._compute_signed_distances, (0, 1, 2, 3))(positions_a, radii_a, positions_b, radii_b)


        # # Slice the Jacobian
        # jac_signed_distances_positions_a = jac_signed_distances[0].detach().numpy()
        # jac_signed_distances_radii_a = jac_signed_distances[1].detach().numpy()
        # jac_signed_distances_positions_b = jac_signed_distances[2].detach().numpy()
        # jac_signed_distances_radii_b = jac_signed_distances[3].detach().numpy()

        jac_signed_distances_positions_a = jac_signed_distances[0]
        jac_signed_distances_radii_a = jac_signed_distances[1]
        jac_signed_distances_positions_b = jac_signed_distances[2]
        jac_signed_distances_radii_b = jac_signed_distances[3]

        # Write the outputs
        partials['signed_distances', 'positions_a'] = jac_signed_distances_positions_a
        partials['signed_distances', 'radii_a'] = jac_signed_distances_radii_a
        partials['signed_distances', 'positions_b'] = jac_signed_distances_positions_b
        partials['signed_distances', 'radii_b'] = jac_signed_distances_radii_b


    @staticmethod
    def _compute_signed_distances(positions_a, radii_a, positions_b, radii_b):
        signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b)
        return signed_distances
