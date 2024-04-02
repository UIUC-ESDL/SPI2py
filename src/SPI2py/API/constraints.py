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


class VolumeFractionCollision(ExplicitComponent):
    def initialize(self):
        self.options.declare('n_projections', types=int, desc='Number of objects with projections')

    def setup(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the inputs
        self.add_input('element_length', val=0)
        for i in range(n_projections):
            self.add_input(f'pseudo_densities_{i}', shape_by_conn=True)

        # Set the outputs
        self.add_output('volume_fraction', shape=(1,))

    def setup_partials(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the partial derivatives
        for i in range(n_projections):
            self.declare_partials('volume_fraction', f'pseudo_densities_{i}')

    def compute(self, inputs, outputs):

        # Get the options
        n_projections = self.options['n_projections']

        # Get the inputs
        element_length = jnp.array(inputs['element_length'])
        pseudo_densities = [jnp.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections)]

        # Calculate the volume fraction constraint
        volume_fraction = self.volume_fraction_collision(pseudo_densities, element_length)

        # Write the outputs
        outputs['volume_fraction'] = volume_fraction

    def compute_partials(self, inputs, partials):

        # Get the options
        n_projections = self.options['n_projections']

        # Get the inputs
        element_length = jnp.array(inputs['element_length'])
        pseudo_densities = [jnp.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections)]

        # Calculate the partial derivatives
        jac_volume_fraction = jacfwd(self.volume_fraction_collision)(pseudo_densities, element_length)

        # Set the partial derivatives
        for i, jac_volume_fraction_i in enumerate(jac_volume_fraction):
            partials['volume_fraction', f'pseudo_densities_{i}'] = jac_volume_fraction_i

    @staticmethod
    def volume_fraction_collision(pseudo_densities, element_length):

        # Aggregate the projected volumes
        volume_individuals = 0
        for pseudo_densities_i in pseudo_densities:
            volume_individuals += pseudo_densities_i.sum() * element_length ** 3

        # Superimpose the projections
        # TODO Add Rho min
        pseudo_densities_superimposed = jnp.zeros_like(pseudo_densities[0])
        for pseudo_densities_i in pseudo_densities:
            pseudo_densities_superimposed += pseudo_densities_i

        pseudo_densities_superimposed = pseudo_densities_superimposed.clip(0, 1)

        # Calculate volume of the superimposed projection
        volume_superimposed = pseudo_densities_superimposed.sum() * element_length ** 3

        # Calculate the volume fraction constraint (negative-null form)
        volume_fraction = volume_individuals / volume_superimposed  - 1

        return volume_fraction

    class VolumeFractionEncapsulation(ExplicitComponent):
        def initialize(self):
            self.options.declare('n_projections', types=int, desc='Number of objects with projections')

        def setup(self):

            # Get the options
            n_projections = self.options['n_projections']

            # Set the inputs
            self.add_input('element_length', val=0)
            for i in range(n_projections):
                self.add_input(f'true_volume_{i}', shape_by_conn=True)
                self.add_input(f'projected_volume_{i}', shape_by_conn=True)

            # Set the outputs
            self.add_output('volume_fraction', shape=(1,))

        def setup_partials(self):

            # Get the options
            n_projections = self.options['n_projections']

            # Set the partial derivatives
            for i in range(n_projections):
                self.declare_partials('volume_fraction', f'true_volume_{i}')
                self.declare_partials('volume_fraction', f'projected_volume_{i}')

        def compute(self, inputs, outputs):

            # Get the options
            n_projections = self.options['n_projections']

            # Get the inputs
            element_length = jnp.array(inputs['element_length'])
            true_volumes = (jnp.array(inputs[f'true_volume_{i}']) for i in range(n_projections))
            projected_volumes = (jnp.array(inputs[f'projected_volume_{i}']) for i in range(n_projections))

            # Calculate the volume fraction constraint
            volume_fraction = self.volume_fraction_encapsulation(element_length, true_volumes, projected_volumes)

            # Write the outputs
            outputs['volume_fraction'] = volume_fraction

        # def compute_partials(self, inputs, partials):
        #
        #     # Get the options
        #     n_projections = self.options['n_projections']
        #
        #     # Get the inputs
        #     element_length = jnp.array(inputs['element_length'])
        #     pseudo_densities = (jnp.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections))
        #
        #     # Calculate the partial derivatives wrt all inputs
        #     arg_nums = tuple(range(n_projections))
        #     jac_volume_fraction = jacfwd(self.volume_fraction_collision, argnums=arg_nums)
        #     jac_volume_fraction_val = jac_volume_fraction(element_length, *pseudo_densities)
        #
        #     # Convert the partial derivatives to numpy arrays
        #     jac_volume_fraction_constraint_np = []
        #     for jac in jac_volume_fraction_val:
        #         jac_volume_fraction_constraint_np.append(jac)
        #
        #     # Set the partial derivatives
        #     for i in range(n_projections):
        #         partials['volume_fraction', f'pseudo_densities_{i}'] = jac_volume_fraction_constraint_np[i]

        @staticmethod
        def volume_fraction_encapsulation(element_length, *pseudo_densities):

            pass

            # # Aggregate the projected volumes
            # volume_individuals = 0
            # for i in range(len(pseudo_densities)):
            #     volume_individuals += pseudo_densities[i].sum() * element_length ** 3
            #
            # # Superimpose the projections
            # # TODO Add Rho min
            # pseudo_densities_superimposed = jnp.zeros_like(pseudo_densities[0])
            # for i in range(len(pseudo_densities)):
            #     pseudo_densities_superimposed += pseudo_densities[i]
            #
            # pseudo_densities_superimposed = pseudo_densities_superimposed.clip(0, 1)
            #
            # # Calculate volume of the superimposed projection
            # volume_superimposed = pseudo_densities_superimposed.sum() * element_length ** 3
            #
            # # Calculate the volume fraction constraint (negative-null form)
            # volume_fraction = volume_individuals / volume_superimposed - 1
            #
            # return volume_fraction
