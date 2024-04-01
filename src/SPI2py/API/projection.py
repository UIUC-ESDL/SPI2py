import jax.numpy as jnp
from jax import jacfwd
from openmdao.api import ExplicitComponent, Group
from ..models.projection.projection import calculate_pseudo_densities

class Projections(Group):
    def initialize(self):
        self.options.declare('n_comp_projections', types=int, desc='Number of component projections')
        self.options.declare('n_int_projections', types=int, desc='Number of interconnect projections')
        self.options.declare('rho_min', types=float, desc='Minimum value of the density', default=1e-5)

    def setup(self):

        # Get the options
        n_comp_projections = self.options['n_comp_projections']
        n_int_projections = self.options['n_int_projections']
        rho_min = self.options['rho_min']

        # Projection counter
        i=0

        # TODO Can I automatically connect this???
        # Add the projection components
        for j in range(n_comp_projections):
            self.add_subsystem('projection_' + str(i), Projection(rho_min=rho_min))
            # self.connect(f'system.components.comp_{j}.transformed_sphere_positions', 'projection_' + str(i) + '.points')
            i += 1

        # Add the interconnect projection components
        for j in range(n_int_projections):
            self.add_subsystem('projection_' + str(i), Projection(rho_min=rho_min))
            # self.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', 'projection_int_' + str(i) + '.points')
            i += 1


class Projection(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid

    TODO Deal with objects outside of mesh!
    """

    def initialize(self):
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')

    def setup(self):

        # Mesh Inputs
        self.add_input('element_length', val=0)
        self.add_input('centers', shape_by_conn=True)
        self.add_input('sample_points', shape_by_conn=True)
        self.add_input('sample_radii', shape_by_conn=True)

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)

        # Outputs
        self.add_output('pseudo_densities', compute_shape=lambda shapes: (shapes['centers'][0],shapes['centers'][1],shapes['centers'][2]))
        self.add_output('volume', val=0.0)

    def setup_partials(self):
        self.declare_partials('pseudo_densities', 'sphere_positions')

    def compute(self, inputs, outputs):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        element_length = inputs['element_length']
        centers = inputs['centers']
        sample_points = inputs['sample_points']
        sample_radii = inputs['sample_radii']

        # Get the Object inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']

        # Convert the input to Jax arrays
        element_length = jnp.array(element_length)
        centers = jnp.array(centers)
        sample_points = jnp.array(sample_points)
        sample_radii = jnp.array(sample_radii)


        sphere_positions = jnp.array(sphere_positions)
        sphere_radii = jnp.array(sphere_radii)

        # Project
        pseudo_densities = self._project(sphere_positions, sphere_radii,
                                                 element_length,
                                                 centers, sample_points, sample_radii, rho_min)

        # Calculate the volume
        volume = jnp.sum(pseudo_densities) * element_length ** 3

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities
        outputs['volume'] = volume

    def compute_partials(self, inputs, partials):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        element_length = inputs['element_length']
        centers = inputs['centers']
        sample_points = inputs['sample_points']
        sample_radii = inputs['sample_radii']

        # Get the Object inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']

        # Convert the input to Jax arrays
        element_length = jnp.array(element_length)
        centers = jnp.array(centers)
        sample_points = jnp.array(sample_points)
        sample_radii = jnp.array(sample_radii)

        sphere_positions = jnp.array(sphere_positions)
        sphere_radii = jnp.array(sphere_radii)

        # Calculate the Jacobian of the kernel
        grad_sphere_positions = jacfwd(self._project, argnums=0)
        grad_sphere_positions_val = grad_sphere_positions(sphere_positions, sphere_radii, element_length,
                                                 centers,sample_points, sample_radii, rho_min)


        partials['pseudo_densities', 'sphere_positions'] = grad_sphere_positions_val

    @staticmethod
    def _project(sphere_positions, sphere_radii, element_length, centers, sample_points, sample_radii, rho_min):

        pseudo_densities = calculate_pseudo_densities(sphere_positions, sphere_radii, element_length, centers, sample_points, sample_radii, rho_min)

        return pseudo_densities