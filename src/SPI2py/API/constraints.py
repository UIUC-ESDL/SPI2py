import jax.numpy as jnp
from jax import jacfwd
from openmdao.api import ExplicitComponent, Group

from ..models.kinematics.distance_calculations import signed_distances_spheres_spheres


# class CombinatorialCollisionDetection(Group):
#     raise NotImplementedError


# class PairwiseCollisionDetection(ExplicitComponent):
#
#     def initialize(self):
#         self.options.declare('n_spheres', types=int, desc='Number of spheres for input a')
#         self.options.declare('m_spheres', types=int, desc='Number of spheres for input b')
#
#     def setup(self):
#         n = self.options['n_spheres']
#         m = self.options['m_spheres']
#
#         self.add_input('positions_a', shape=(n, 3))
#         self.add_input('radii_a', shape=(n, 1))
#
#         self.add_input('positions_b', shape=(m, 3))
#         self.add_input('radii_b', shape=(m, 1))
#
#         self.add_output('signed_distances', shape=(n, m))
#
#     def setup_partials(self):
#         self.declare_partials('signed_distances', 'positions_a')
#         self.declare_partials('signed_distances', 'radii_a')
#         self.declare_partials('signed_distances', 'positions_b')
#         self.declare_partials('signed_distances', 'radii_b')
#
#     def compute(self, inputs, outputs):
#
#         # Extract the inputs
#         positions_a = inputs['positions_a']
#         radii_a = inputs['radii_a']
#         positions_b = inputs['positions_b']
#         radii_b = inputs['radii_b']
#
#         # Convert the inputs to torch tensors
#         positions_a = torch.tensor(positions_a, dtype=torch.float64)
#         radii_a = torch.tensor(radii_a, dtype=torch.float64)
#         positions_b = torch.tensor(positions_b, dtype=torch.float64)
#         radii_b = torch.tensor(radii_b, dtype=torch.float64)
#
#         # Calculate the signed distances
#         signed_distances = self._compute_signed_distances(positions_a, radii_a, positions_b, radii_b)
#
#         # Convert the outputs to numpy arrays
#         signed_distances = signed_distances.detach().numpy()
#
#         # Write the outputs
#         outputs['signed_distances'] = signed_distances
#
#     def compute_partials(self, inputs, partials):
#
#         # Extract the inputs
#         positions_a = inputs['positions_a']
#         radii_a = inputs['radii_a']
#         positions_b = inputs['positions_b']
#         radii_b = inputs['radii_b']
#
#         # Convert the inputs to torch tensors
#         positions_a = torch.tensor(positions_a, dtype=torch.float64, requires_grad=True)
#         radii_a = torch.tensor(radii_a, dtype=torch.float64, requires_grad=True)
#         positions_b = torch.tensor(positions_b, dtype=torch.float64, requires_grad=True)
#         radii_b = torch.tensor(radii_b, dtype=torch.float64, requires_grad=True)
#
#         # Define the Jacobian matrices using PyTorch Autograd
#         jac_signed_distances = jacfwd(self._compute_signed_distances, argnums=(0, 1, 2, 3))
#
#         # Evaluate the Jacobian matrices
#         jac_signed_distances_vals = jac_signed_distances(positions_a, radii_a, positions_b, radii_b)
#
#         # Slice the Jacobian
#         jac_signed_distances_positions_a = jac_signed_distances_vals[0].detach().numpy()
#         jac_signed_distances_radii_a = jac_signed_distances_vals[1].detach().numpy()
#         jac_signed_distances_positions_b = jac_signed_distances_vals[2].detach().numpy()
#         jac_signed_distances_radii_b = jac_signed_distances_vals[3].detach().numpy()
#
#         # Write the outputs
#         partials['signed_distances', 'positions_a'] = jac_signed_distances_positions_a
#         partials['signed_distances', 'radii_a'] = jac_signed_distances_radii_a
#         partials['signed_distances', 'positions_b'] = jac_signed_distances_positions_b
#         partials['signed_distances', 'radii_b'] = jac_signed_distances_radii_b
#
#     @staticmethod
#     def _compute_signed_distances(positions_a, radii_a, positions_b, radii_b):
#         signed_distances = signed_distances_spheres_spheres(positions_a, radii_a, positions_b, radii_b)
#         return signed_distances


class VolumetricCollision(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_projections', types=int, desc='Number of objects with projections')
        self.options.declare('relative_volume_tolerance', types=float, desc='Relative volume tolerance')

    def setup(self):

        self.add_input('element_length', val=0)

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
        element_length = inputs['element_length']
        element_length = jnp.array(element_length)

        # Extract the inputs and convert to a numpy array
        element_pseudo_densities = []
        for i in range(n):
            element_pseudo_densities.append(inputs[f'element_pseudo_densities_{i}'])

        element_pseudo_densities_tensors = ()
        for i in range(n):
            element_pseudo_densities_tensors = element_pseudo_densities_tensors + (jnp.array(element_pseudo_densities[i]),)

        volume_fraction_constraint = self._volume_fraction_overlap(element_length, *element_pseudo_densities_tensors)

        # Write the outputs
        outputs['volume_fraction_constraint'] = volume_fraction_constraint

    def compute_partials(self, inputs, partials):

        # Get the options
        n = self.options['n_projections']

        # Calculate the volume
        element_length = inputs['element_length']
        # element_length = torch.tensor(element_length, dtype=torch.float64, requires_grad=True)
        element_length = jnp.array(element_length)

        # Extract the inputs and convert to a numpy array
        element_pseudo_densities = ()
        for i in range(n):
            element_pseudo_densities = element_pseudo_densities + (inputs[f'element_pseudo_densities_{i}'],)

        element_pseudo_densities_tensors = ()
        for i in range(n):
            element_pseudo_densities_tensors = element_pseudo_densities_tensors + (jnp.array(element_pseudo_densities[i]),)

        # Calculate the partial derivatives wrt all inputs
        argnums = tuple(range(n))
        jac_volume_fraction_constraint = jacfwd(self._volume_fraction_overlap, argnums=argnums)(element_length, *element_pseudo_densities_tensors)

        # Convert the partial derivatives to numpy arrays
        jac_volume_fraction_constraint_np = []
        for jac in jac_volume_fraction_constraint:
            jac_volume_fraction_constraint_np.append(jac)

        # Set the partial derivatives
        for i in range(n):
            partials['volume_fraction_constraint', f'element_pseudo_densities_{i}'] = jac_volume_fraction_constraint_np[i]

    @staticmethod
    def _volume_fraction_overlap(element_length, *pseudo_densities):

        # Aggregate the projected volumes
        volume_individuals = 0
        for i in range(len(pseudo_densities)):
            volume_individuals += pseudo_densities[i].sum() * element_length ** 3

        # Superimpose the projections
        # TODO Add Rho min
        pseudo_densities_superimposed = jnp.zeros_like(pseudo_densities[0])
        for i in range(len(pseudo_densities)):
            pseudo_densities_superimposed += pseudo_densities[i]

        pseudo_densities_superimposed = pseudo_densities_superimposed.clip(0, 1)

        # Calculate volume of the superimposed projection
        volume_superimposed = pseudo_densities_superimposed.sum() * element_length ** 3

        # Calculate the volume fraction constraint (negative-null form)
        volume_fraction = volume_individuals / volume_superimposed  - 1

        return volume_fraction
