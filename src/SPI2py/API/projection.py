import jax.numpy as jnp
from jax import jacfwd
from openmdao.api import ExplicitComponent, Group

from ..models.projection.projection import project_component
from ..models.projection.projection import project_interconnect
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from ..models.utilities.aggregation import kreisselmeier_steinhauser_max


class Projections(Group):
    pass


class ProjectComponent(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid
    """

    def initialize(self):
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')
        self.options.declare('kernel_steps_per_unit_length', types=int, desc='Number of kernel steps per unit length', default=1)

    def setup(self):

        # Mesh Inputs
        self.add_input('element_size', val=0)
        self.add_input('mesh_centers', shape_by_conn=True)

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)
        # self.add_input('volume', val=0.0)

        # Outputs
        self.add_output('pseudo_densities', compute_shape=lambda shapes: (shapes['mesh_centers'][0], shapes['mesh_centers'][1], shapes['mesh_centers'][2]))
        # self.add_output('volume_estimation_error', val=0.0, desc='How accurately the projection represents the object')

        # volume_kernel = jnp.sum(4/3 * jnp.pi * kernel_radii ** 3)
        # volume_approximation_error = abs((volume_kernel - volume_element) / volume_element)

    # def setup_partials(self):
    #     self.declare_partials('pseudo_densities', 'sphere_positions')



    def compute(self, inputs, outputs):

        # Get the Mesh inputs
        kernel_steps_per_unit_length = self.options['kernel_steps_per_unit_length']
        kernel_points, kernel_radii = create_uniform_kernel(kernel_steps_per_unit_length, mode='circumscription')
        kernel_points = kernel_points.reshape(-1, 3)
        kernel_radii = kernel_radii.reshape(-1, 1)


        # Get the Mesh inputs
        element_size   = jnp.array(inputs['element_size'])
        mesh_centers  = jnp.array(inputs['mesh_centers'])
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii     = jnp.array(inputs['sphere_radii'])
        # volume           = jnp.array(inputs['volume'])

        # Compute the pseudo-densities
        pseudo_densities = self._project(mesh_centers, element_size, sphere_positions, sphere_radii, kernel_points, kernel_radii)

        # Compute the volume estimation error
        # projected_volume = jnp.sum(pseudo_densities * element_size ** 3)
        # volume_estimation_error = jnp.abs(volume - projected_volume) / volume

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities
        # outputs['volume_estimation_error'] = volume_estimation_error

    # def compute_partials(self, inputs, partials):
    #
    #     # Get the inputs
    #     element_bounds = jnp.array(inputs['element_bounds'])
    #     sample_points = jnp.array(inputs['element_sphere_positions'])
    #     sample_radii = jnp.array(inputs['element_sphere_radii'])
    #     sphere_positions = jnp.array(inputs['sphere_positions'])
    #     sphere_radii     = jnp.array(inputs['sphere_radii'])
    #
    #     # Calculate the Jacobian of the pseudo-densities
    #     jac_pseudo_densities = jacfwd(self._project)(sphere_positions, sphere_radii, sample_points, sample_radii, element_bounds)
    #
    #     # Set the partials
    #     partials['pseudo_densities', 'sphere_positions'] = jac_pseudo_densities


    @staticmethod
    def _project(mesh_centers, mesh_size,
                 obj_points, obj_radii,
                 kernel_points, kernel_radii):

        # TODO Fix mesh size to scalar
        pseudo_densities, kernel_points, kernel_radii = project_component(mesh_centers, float(mesh_size[0]),
                                                                          obj_points, obj_radii,
                                                                          kernel_points, kernel_radii)
        return pseudo_densities


class ProjectInterconnect(ExplicitComponent):

    def initialize(self):
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')

    def setup(self):

        # Mesh Inputs
        self.add_input('element_size', val=0)
        self.add_input('mesh_centers', shape_by_conn=True)

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)
        self.add_input('volume', val=0.0)

        # Outputs
        self.add_output('pseudo_densities',
                        compute_shape=lambda shapes: (shapes['centers'][0], shapes['centers'][1], shapes['centers'][2]))

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the Mesh inputs
        element_size   = jnp.array(inputs['element_size'])
        element_bounds   = jnp.array(inputs['element_bounds'])
        sample_points    = jnp.array(inputs['element_sphere_positions'])
        sample_radii     = jnp.array(inputs['element_sphere_radii'])
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii     = jnp.array(inputs['sphere_radii'])
        volume           = jnp.array(inputs['volume'])

        # Compute the pseudo-densities
        pseudo_densities = self._project(sample_points, sample_radii, sphere_positions, sphere_radii)

        # Compute the volume estimation error
        # projected_volume = jnp.sum(pseudo_densities * element_size ** 3)
        # volume_estimation_error = jnp.abs(volume - projected_volume) / volume

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities
        # outputs['volume_estimation_error'] = volume_estimation_error

    # def compute_partials(self, inputs, partials):
    #
    #     # Get the inputs
    #     element_bounds = jnp.array(inputs['element_bounds'])
    #     sample_points    = jnp.array(inputs['sample_points'])
    #     sample_radii     = jnp.array(inputs['sample_radii'])
    #     sphere_positions = jnp.array(inputs['sphere_positions'])
    #     sphere_radii     = jnp.array(inputs['sphere_radii'])
    #     aabb = jnp.array(inputs['AABB'])
    #
    #     # Calculate the Jacobian of the pseudo-densities
    #     jac_pseudo_densities = jacfwd(self._project)(sphere_positions, sphere_radii, sample_points, sample_radii, aabb, element_bounds)
    #
    #     # Set the partials
    #     partials['pseudo_densities', 'sphere_positions'] = jac_pseudo_densities


    @staticmethod
    def _project(sample_points, sample_radii, sphere_positions, sphere_radii):

        import numpy as np
        def create_cylinders(points, radius):
            x1 = np.array(points[:-1])  # Start positions (-1, 3)
            x2 = np.array(points[1:])  # Stop positions (-1, 3)
            r = np.full((x1.shape[0], 1), radius)
            return x1, x2, r


        # FIXME different radii for different int segments
        X1, X2, R = create_cylinders(sphere_positions, sphere_radii[0])

        pseudo_densities = project_interconnect(sample_points, sample_radii, X1, X2, R)
        return pseudo_densities



class ProjectionAggregator(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_projections', types=int, desc='Number of projections')
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)

    def setup(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the inputs
        self.add_input('element_length', val=0)

        for i in range(n_projections):
            self.add_input(f'pseudo_densities_{i}', shape_by_conn=True)


        # Set the outputs
        self.add_output('pseudo_densities', copy_shape='pseudo_densities_0')
        self.add_output('max_pseudo_density', val=0.0, desc='How much of each object overlaps/is out of bounds')

    def setup_partials(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the partials
        for i in range(n_projections):
            self.declare_partials('pseudo_densities', f'pseudo_densities_{i}')
            self.declare_partials('max_pseudo_density', f'pseudo_densities_{i}')


    def compute(self, inputs, outputs):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        element_length = jnp.array(inputs['element_length'])
        pseudo_densities = [jnp.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections)]

        # Calculate the values
        aggregate_pseudo_densities, max_pseudo_density = self._aggregate_pseudo_densities(pseudo_densities, element_length, rho_min)


        # Write the outputs
        outputs['pseudo_densities'] = aggregate_pseudo_densities
        outputs['max_pseudo_density'] = max_pseudo_density

    def compute_partials(self, inputs, partials):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        element_length = jnp.array(inputs['element_length'])
        pseudo_densities = [jnp.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections)]

        # Calculate the partial derivatives
        jac_pseudo_densities, jac_max_pseudo_density = jacfwd(self._aggregate_pseudo_densities)(pseudo_densities, element_length, rho_min)

        # Set the partial derivatives
        jacs = zip(jac_pseudo_densities, jac_max_pseudo_density)
        for i, (jac_pseudo_densities_i, jac_max_pseudo_density_i) in enumerate(jacs):
            partials['pseudo_densities', f'pseudo_densities_{i}'] = jac_pseudo_densities_i
            partials['max_pseudo_density', f'pseudo_densities_{i}'] = jac_max_pseudo_density_i

    @staticmethod
    def _aggregate_pseudo_densities(pseudo_densities, element_length, rho_min):

        # Aggregate the pseudo-densities
        aggregate_pseudo_densities = jnp.zeros_like(pseudo_densities[0])
        for pseudo_density in pseudo_densities:
            aggregate_pseudo_densities += pseudo_density

        # Ensure that no pseudo-density is below the minimum value
        aggregate_pseudo_densities = jnp.maximum(aggregate_pseudo_densities, rho_min)

        # TODO rethink max if we need to overlap interconnects and components

        # Calculate the maximum pseudo-density
        max_pseudo_density = kreisselmeier_steinhauser_max(aggregate_pseudo_densities)

        return aggregate_pseudo_densities, max_pseudo_density

#####

