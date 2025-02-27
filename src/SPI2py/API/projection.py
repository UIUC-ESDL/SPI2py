import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, vjp
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

        # General parameters
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')
        self.options.declare('kernel_steps_per_unit_length', types=int, desc='Number of kernel steps per unit length', default=1)

        # Mesh parameters
        self.options.declare('element_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('kernel_points', types=jnp.ndarray, desc='Points representing the mesh kernel')
        self.options.declare('kernel_radii', types=jnp.ndarray, desc='Radii of kernel points')

    def setup(self):

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('pseudo_densities', compute_shape=lambda shapes: (nx, ny, nz))

        # Archived code
        # self.add_input('volume', val=0.0)
        # self.add_output('volume_estimation_error', val=0.0, desc='How accurately the projection represents the object')
        # volume_kernel = jnp.sum(4/3 * jnp.pi * kernel_radii ** 3)
        # volume_approximation_error = abs((volume_kernel - volume_element) / volume_element)

    def setup_partials(self):
        self.declare_partials('pseudo_densities', 'sphere_positions', method='exact')
        self.declare_partials('pseudo_densities', 'sphere_radii', method='exact')


    def compute(self, inputs, outputs):

        # Get the Mesh inputs
        element_size = jnp.atleast_1d(self.options['element_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_points = jnp.array(self.options['kernel_points'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the Mesh inputs
        # TODO Fix atleast 1d?
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii     = jnp.array(inputs['sphere_radii'])

        # Compute the pseudo-densities
        pseudo_densities = self._compute_primal(mesh_centers, element_size, sphere_positions, sphere_radii, kernel_points, kernel_radii)

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):

        # Get the Mesh inputs
        element_size = jnp.atleast_1d(self.options['element_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_points = jnp.array(self.options['kernel_points'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the Mesh inputs
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii = jnp.array(inputs['sphere_radii'])

        # Define primals in the order expected by _compute_primal.
        primals = (mesh_centers, element_size, sphere_positions, sphere_radii, kernel_points, kernel_radii)

        if mode == 'fwd':
            # For forward mode, supply the tangent (perturbation) for each input.
            # Assume that kernel_points and kernel_radii are constant,
            # so we supply zeros for them.
            tangents = (jnp.zeros_like(mesh_centers),
                        jnp.zeros_like(element_size),
                        d_inputs['sphere_positions'],
                        d_inputs['sphere_radii'],
                        jnp.zeros_like(kernel_points),
                        jnp.zeros_like(kernel_radii))
            # jax.jvp returns (primal_out, tangent_out)
            _, tangent_out = jvp(self._compute_primal, primals, tangents)
            # Set the output tangent (directional derivative) for pseudo_densities.
            d_outputs['pseudo_densities'] = tangent_out

        elif mode == 'rev':
            # In reverse mode, use vjp to get a pullback function.
            primal_out, pullback = vjp(self._compute_primal, *primals)
            # d_outputs['pseudo_densities'] holds the cotangent (sensitivity) for the pseudo_densities.
            cotangent = d_outputs['pseudo_densities']
            # pullback returns a tuple of gradients in the order of primals.
            grads = pullback(cotangent)

            # d_inputs['element_size'] = grads[1]
            # d_inputs['mesh_centers'] = grads[0]
            d_inputs['sphere_positions'] = grads[2]
            d_inputs['sphere_radii'] = grads[3]
            # Ignore the gradients for kernel_points and kernel_radii if they are constant.


    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        obj_points, obj_radii,
                        kernel_points, kernel_radii):


        pseudo_densities, kernel_points, kernel_radii = project_component(mesh_centers, mesh_size,
                                                                          obj_points, obj_radii,
                                                                          kernel_points, kernel_radii)
        return pseudo_densities


class ProjectInterconnect(ExplicitComponent):

    def initialize(self):

        # General parameters
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')
        self.options.declare('kernel_steps_per_unit_length', types=int, desc='Number of kernel steps per unit length', default=1)

        # Mesh parameters
        self.options.declare('element_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('kernel_points', types=jnp.ndarray, desc='Points representing the mesh kernel')
        self.options.declare('kernel_radii', types=jnp.ndarray, desc='Radii of kernel points')

    def setup(self):

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('pseudo_densities', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('pseudo_densities', 'sphere_positions', method='exact')
        self.declare_partials('pseudo_densities', 'sphere_radii', method='exact')

    def compute(self, inputs, outputs):

        # Get the Mesh inputs
        element_size = jnp.atleast_1d(self.options['element_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_points = jnp.array(self.options['kernel_points'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the Mesh inputs
        # TODO Fix atleast 1d?
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii     = jnp.array(inputs['sphere_radii'])

        # Compute the pseudo-densities
        pseudo_densities = self._compute_primal(mesh_centers, element_size, sphere_positions, sphere_radii, kernel_points, kernel_radii)

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities

    @staticmethod
    def _compute_primal(mesh_centers, element_size, cyl_points, cyl_radii, kernel_points, kernel_radii):
        pseudo_densities, _, _ = project_interconnect(mesh_centers, element_size, cyl_points, cyl_radii, kernel_points, kernel_radii)
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

