import jax.numpy as np

import torch
from torch.func import jacfwd
from openmdao.api import ExplicitComponent, Group
from itertools import combinations, product

from SPI2py.models.kinematics.distance_calculations import signed_distances_spheres_spheres
from SPI2py.models.physics.continuum.geometric_projection import projection_volume

class CombinatorialCollisionDetection(Group):
    pass

class PairwiseCollisionDetection(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_spheres', types=int, desc='Number of spheres for input a')
        self.options.declare('m_spheres', types=int, desc='Number of spheres for input b')


    def setup(self):
        n = self.options['n_spheres']
        m = self.options['m_spheres']

        self.add_input('positions_a', shape=(n, 3))
        self.add_input('radii_a', shape=(n, 1))

        self.add_input('positions_b', shape=(m, 3))
        self.add_input('radii_b', shape=(m, 1))

        self.add_output('signed_distances', shape=(n, m))

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
        positions_a = torch.tensor(positions_a, dtype=torch.float64)
        radii_a = torch.tensor(radii_a, dtype=torch.float64)
        positions_b = torch.tensor(positions_b, dtype=torch.float64)
        radii_b = torch.tensor(radii_b, dtype=torch.float64)

        # Calculate the signed distances
        signed_distances = self._compute_signed_distances(positions_a, radii_a, positions_b, radii_b)

        # Convert the outputs to numpy arrays
        signed_distances = signed_distances.detach().numpy()

        # Write the outputs
        outputs['signed_distances'] = signed_distances

    def compute_partials(self, inputs, partials):

        # Extract the inputs
        positions_a = inputs['positions_a']
        radii_a = inputs['radii_a']
        positions_b = inputs['positions_b']
        radii_b = inputs['radii_b']

        # Convert the inputs to torch tensors
        positions_a = torch.tensor(positions_a, dtype=torch.float64, requires_grad=True)
        radii_a = torch.tensor(radii_a, dtype=torch.float64, requires_grad=True)
        positions_b = torch.tensor(positions_b, dtype=torch.float64, requires_grad=True)
        radii_b = torch.tensor(radii_b, dtype=torch.float64, requires_grad=True)

        # Define the Jacobian matrices using PyTorch Autograd
        jac_signed_distances = jacfwd(self._compute_signed_distances, argnums=(0, 1, 2, 3))

        # Evaluate the Jacobian matrices
        jac_signed_distances_vals = jac_signed_distances(positions_a, radii_a, positions_b, radii_b)

        # Slice the Jacobian
        jac_signed_distances_positions_a = jac_signed_distances_vals[0].detach().numpy()
        jac_signed_distances_radii_a = jac_signed_distances_vals[1].detach().numpy()
        jac_signed_distances_positions_b = jac_signed_distances_vals[2].detach().numpy()
        jac_signed_distances_radii_b = jac_signed_distances_vals[3].detach().numpy()

        # Write the outputs
        partials['signed_distances', 'positions_a'] = jac_signed_distances_positions_a
        partials['signed_distances', 'radii_a'] = jac_signed_distances_radii_a
        partials['signed_distances', 'positions_b'] = jac_signed_distances_positions_b
        partials['signed_distances', 'radii_b'] = jac_signed_distances_radii_b

    @staticmethod
    def _compute_signed_distances(positions_a, radii_a, positions_b, radii_b):
        signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b)
        return signed_distances


class VolumeFractionConstraint(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_projections', types=int, desc='Number of projections')
        self.options.declare('relative_volume_tolerance', types=float, desc='Relative volume tolerance')
        self.options.declare('min_xyz', types=(int, float), desc='Minimum value of the x-, y-, and z-axis')
        self.options.declare('max_xyz', types=(int, float), desc='Maximum value of the x-, y-, and z-axis')
        self.options.declare('n_el_xyz', types=int, desc='Number of elements along the x-, y-, and z-axis')
        self.a = 1  # FIXME

    def setup(self):
        n = self.options['n_projections']


        for i in range(n):
            self.add_input(f'element_pseudo_densities_{i}', shape_by_conn=True)

        self.add_output('volume_fraction_constraint', shape=(1,))

    def setup_partials(self):
        n_projections = self.options['n_projections']
        for i in range(n_projections):
            self.declare_partials('volume_fraction_constraint', f'element_pseudo_densities_{i}')

    def compute(self, inputs, outputs):

        # Get the options
        n = self.options['n_projections']

        # Calculate the volume
        max_xyz = self.options['max_xyz']
        min_xyz = self.options['min_xyz']
        n_el_xyz = self.options['n_el_xyz']
        element_length = (max_xyz - min_xyz) / n_el_xyz

        # Extract the inputs and convert to a numpy array
        element_pseudo_densities = []
        for i in range(n):
            element_pseudo_densities.append(inputs[f'element_pseudo_densities_{i}'])




        volume_fraction_constraint = self._volume_fraction_constraint(element_pseudo_densities, element_length)


        # Write the outputs
        outputs['volume_fraction_constraint'] = volume_fraction_constraint

    # def compute_partials(self, inputs, partials):
    #     pass

    @staticmethod
    def _volume_fraction_constraint(element_pseudo_densities, element_length):

        # Calculate the sum of the individual volumes
        individual_volumes = 0
        for i in range(len(element_pseudo_densities)):
            projected_volume = element_pseudo_densities[i].sum() * element_length ** 3
            individual_volumes += projected_volume

        # Calculate the sum of the combined volumes

        # Add all the element pseudo-densities together
        combined_pseudo_densities = np.zeros_like(element_pseudo_densities[0])
        for i in range(len(element_pseudo_densities)):
            combined_pseudo_densities += element_pseudo_densities[i]

        # Ensure that no element densities exceed 1
        # TODO Evaluate smooth clipping, etc.
        combined_pseudo_densities = combined_pseudo_densities.clip(0, 1)

        # Calculate the volume
        combined_volume = combined_pseudo_densities.sum() * element_length ** 3

        # Calculate the volume fraction constraint (negative-null form) TODO Add tolerance? No, in OpenMDAO
        volume_fraction_constraint = individual_volumes / combined_volume  - 1

        return volume_fraction_constraint