import jax.numpy as jnp
from jax import jacfwd, jacrev, jvp, vjp
from openmdao.api import ExplicitComponent, Group
from ..models.projection.projection import project_component
from ..models.projection.projection import project_interconnect
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

        # Mesh parameters
        self.options.declare('element_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('kernel_points', types=jnp.ndarray, desc='Points representing the mesh kernel')
        self.options.declare('kernel_radii', types=jnp.ndarray, desc='Radii of kernel points')

    def setup(self):

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)
        self.add_input('heat_load', val=0.0)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_heat_loads', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('densities', 'sphere_positions', method='exact')
        self.declare_partials('densities', 'sphere_radii', method='exact')
        self.declare_partials('penalized_densities', 'sphere_positions', method='exact')
        self.declare_partials('penalized_densities', 'sphere_radii', method='exact')
        self.declare_partials('penalized_heat_loads', 'heat_load', method='exact')
        self.declare_partials('penalized_heat_loads', 'sphere_positions', method='exact')
        self.declare_partials('penalized_heat_loads', 'sphere_radii', method='exact')


    def compute(self, inputs, outputs):

        # Get the mesh parameters
        element_size = jnp.atleast_1d(self.options['element_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_points = jnp.array(self.options['kernel_points'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii     = jnp.array(inputs['sphere_radii'])

        # Compute the pseudo-densities
        densities, densities_penalized, heat_load_mod = self._compute_primal(mesh_centers, element_size, sphere_positions, sphere_radii, kernel_points, kernel_radii)

        # Write the outputs
        outputs['densities'] = densities

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

        # Forward mode, compute the Jacobian-vector product
        if mode == 'fwd':

            # Define zeros for constant terms
            d_inputs_mesh_centers = jnp.zeros_like(mesh_centers)
            d_inputs_element_size = jnp.zeros_like(element_size)
            d_inputs_kernel_points = jnp.zeros_like(kernel_points)
            d_inputs_kernel_radii = jnp.zeros_like(kernel_radii)

            tangents = (d_inputs_mesh_centers,
                        d_inputs_element_size,
                        d_inputs['sphere_positions'],
                        d_inputs['sphere_radii'],
                        d_inputs_kernel_points,
                        d_inputs_kernel_radii)

            # JVP returns (primal_out, tangent_out), discard primal_out
            _, tangent_out = jvp(self._compute_primal, primals, tangents)

            # Set the output tangent (directional derivative) for densities.
            d_outputs['densities'] = tangent_out

        # Reverse mode, compute the vector-Jacobian product
        elif mode == 'rev':

            primal_out, pullback = vjp(self._compute_primal, *primals)

            # d_outputs['densities'] holds the cotangent (sensitivity) for the densities.
            cotangent = d_outputs['densities']


            # pullback returns a tuple of gradients in the order of primals.
            grads = pullback(cotangent)

            # Only extract the gradients for the input (non-constant) terms.
            d_inputs['sphere_positions'] = grads[2]
            d_inputs['sphere_radii'] = grads[3]


    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        obj_points, obj_radii,
                        kernel_points, kernel_radii,
                        heat_load=0.0):

        # Calculate the pseudo-densities
        densities, penalized_densities = project_component(mesh_centers, mesh_size,
                                                           obj_points, obj_radii,
                                                           kernel_points, kernel_radii)

        # Heat load
        penalized_heat_loads = heat_load * penalized_densities

        return densities, penalized_densities, penalized_heat_loads


class ProjectInterconnect(ExplicitComponent):

    def initialize(self):

        # General parameters
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')

        # Mesh parameters
        self.options.declare('element_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('kernel_points', types=jnp.ndarray, desc='Points representing the mesh kernel')
        self.options.declare('kernel_radii', types=jnp.ndarray, desc='Radii of kernel points')

    def setup(self):

        # Object Inputs
        self.add_input('cyl_positions', shape_by_conn=True)
        self.add_input('cyl_radius', shape_by_conn=True)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('densities', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('densities', 'cyl_positions', method='exact')
        self.declare_partials('densities', 'cyl_radius', method='exact')

    def compute(self, inputs, outputs):

        # Get the Mesh parameters
        element_size = jnp.atleast_1d(self.options['element_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_points = jnp.array(self.options['kernel_points'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        cyl_positions = jnp.array(inputs['cyl_positions'])
        cyl_radius     = jnp.array(inputs['cyl_radius'])

        # Compute the pseudo-densities
        densities = self._compute_primal(mesh_centers, element_size,
                                                cyl_positions, cyl_radius,
                                                kernel_points, kernel_radii)

        # Write the outputs
        outputs['densities'] = densities

    def compute_partials(self, inputs, partials):

        # Get the Mesh parameters
        element_size = jnp.atleast_1d(self.options['element_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_points = jnp.array(self.options['kernel_points'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        cyl_positions = jnp.array(inputs['cyl_positions'])
        cyl_radius = jnp.array(inputs['cyl_radius'])

        # Calculate the partial derivatives
        jac_densities = jacrev(self._compute_primal)(mesh_centers, element_size,
                                                            cyl_positions, cyl_radius,
                                                            kernel_points, kernel_radii)

        # Set the partial derivatives
        partials['densities', 'cyl_positions'] = jac_densities[2]
        partials['densities', 'cyl_radius'] = jac_densities[3]

    @staticmethod
    def _compute_primal(mesh_centers, element_size,
                        cyl_points, cyl_radii,
                        kernel_points, kernel_radii):

        densities, _, _ = project_interconnect(mesh_centers, element_size,
                                                      cyl_points, cyl_radii,
                                                      kernel_points, kernel_radii)

        return densities



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
            self.add_input(f'densities_{i}', shape_by_conn=True)


        # Set the outputs
        self.add_output('aggregated_densities', copy_shape='densities_0')
        self.add_output('max_density', val=0.0)

    def setup_partials(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the partials
        for i in range(n_projections):
            self.declare_partials('aggregated_densities', f'densities_{i}')
            self.declare_partials('max_density', f'densities_{i}')


    def compute(self, inputs, outputs):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        densities = [jnp.array(inputs[f'densities_{i}']) for i in range(n_projections)]

        # Calculate the values
        aggregated_densities, max_density = self._compute_primal(densities, rho_min)


        # Write the outputs
        outputs['aggregated_densities'] = aggregated_densities
        outputs['max_density'] = max_density

    def compute_partials(self, inputs, partials):

        # TODO Implement as jacvec product

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        densities = [jnp.array(inputs[f'densities_{i}']) for i in range(n_projections)]

        # Calculate the partial derivatives
        jac_densities, jac_max_density = jacfwd(self._compute_primal)(densities, rho_min)

        # Set the partial derivatives
        jacs = zip(jac_densities, jac_max_density)
        for i, (jac_densities_i, jac_max_density_i) in enumerate(jacs):
            partials['aggregated_densities', f'densities_{i}'] = jac_densities_i
            partials['max_density', f'densities_{i}'] = jac_max_density_i

    @staticmethod
    def _compute_primal(densities, rho_min):

        # Aggregate the pseudo-densities
        aggregate_densities = jnp.zeros_like(densities[0])
        for density in densities:
            aggregate_densities += density

        # Ensure that no pseudo-density is below the minimum value
        aggregated_densities = jnp.maximum(aggregate_densities, rho_min)

        # Calculate the maximum pseudo-density
        max_density = kreisselmeier_steinhauser_max(aggregated_densities)

        return aggregated_densities, max_density

