import numpy as np
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
        self.options.declare('mesh_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('kernel_centers', types=jnp.ndarray, desc='Points representing the mesh kernel')
        self.options.declare('kernel_radii', types=jnp.ndarray, desc='Radii of kernel points')

    def setup(self):

        # Object Inputs
        self.add_input('centers', shape_by_conn=True)
        self.add_input('radii', shape_by_conn=True)
        self.add_input('heat_load', val=0.0)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_heat_loads', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('densities', 'centers', method='exact')
        self.declare_partials('densities', 'radii', method='exact')
        self.declare_partials('penalized_densities', 'centers', method='exact')
        self.declare_partials('penalized_densities', 'radii', method='exact')
        self.declare_partials('penalized_heat_loads', 'centers', method='exact')
        self.declare_partials('penalized_heat_loads', 'radii', method='exact')
        self.declare_partials('penalized_heat_loads', 'heat_load', method='exact')

    def compute(self, inputs, outputs):

        # Get the mesh parameters
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        centers = jnp.array(inputs['centers'])
        radii     = jnp.array(inputs['radii'])
        heat_load = jnp.array(inputs['heat_load'])

        # Compute the pseudo-densities
        densities, penalized_densities, penalized_heat_loads = self._compute_primal(mesh_centers, mesh_size,
                                                                                    centers, radii,
                                                                                    kernel_centers, kernel_radii,
                                                                                    heat_load)

        # Set the outputs
        outputs['densities'] = np.asarray(densities)
        outputs['penalized_densities'] = np.asarray(penalized_densities)
        outputs['penalized_heat_loads'] = np.asarray(penalized_heat_loads)

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        # Get constants.
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii  = jnp.array(self.options['kernel_radii'])
        p = self.options['penalization_exponent']

        # Get design inputs.
        centers = jnp.array(inputs['centers'])
        radii   = jnp.array(inputs['radii'])
        heat_load = jnp.array(inputs['heat_load'])  # shape (1,)

        # Define primals: note that we include heat_load as the 7th argument.
        primals = (mesh_centers, mesh_size, centers, radii, kernel_centers, kernel_radii, heat_load)

        if mode == 'fwd':
            # For forward mode, define tangents.
            t_mesh_centers = jnp.zeros_like(mesh_centers)
            t_mesh_size = jnp.zeros_like(mesh_size)
            t_kernel_centers = jnp.zeros_like(kernel_centers)
            t_kernel_radii = jnp.zeros_like(kernel_radii)
            t_centers = d_inputs.get('centers', jnp.zeros_like(centers))
            t_radii = d_inputs.get('radii', jnp.zeros_like(radii))
            t_heat_load = d_inputs.get('heat_load', jnp.zeros_like(heat_load))
            tangents = (t_mesh_centers,
                        t_mesh_size,
                        t_centers,
                        t_radii,
                        t_kernel_centers,
                        t_kernel_radii,
                        t_heat_load)
            # Compute JVP.
            _, tangent_out = jvp(self._compute_primal, primals, tangents)
            # tangent_out is a tuple with three arrays.
            d_outputs['densities'] = np.asarray(tangent_out[0])
            d_outputs['penalized_densities'] = np.asarray(tangent_out[1])
            d_outputs['penalized_heat_loads'] = np.asarray(tangent_out[2])

        elif mode == 'rev':
            # Reverse mode: compute VJP.
            primal_out, pullback = vjp(self._compute_primal, *primals)
            # d_outputs are the cotangents for the outputs.
            cotangent = (d_outputs['densities'],
                         d_outputs['penalized_densities'],
                         d_outputs['penalized_heat_loads'])
            grads = pullback(cotangent)
            # grads is a tuple with derivatives for each input in the order of primals.
            # We only differentiate with respect to the design inputs (centers, radii, heat_load)
            d_inputs['centers'] = np.asarray(grads[2])
            d_inputs['radii'] = np.asarray(grads[3])
            d_inputs['heat_load'] = np.asarray(grads[6])


    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        obj_centers, obj_radii,
                        kernel_centers, kernel_radii,
                        heat_load):

        # Calculate the pseudo-densities
        densities, penalized_densities = project_component(mesh_centers, mesh_size,
                                                           obj_centers, obj_radii,
                                                           kernel_centers, kernel_radii)

        # Heat load
        penalized_heat_loads = heat_load * penalized_densities

        return densities, penalized_densities, penalized_heat_loads


class ProjectInterconnect(ExplicitComponent):

    def initialize(self):

        # General parameters
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')

        # Mesh parameters
        self.options.declare('mesh_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('kernel_centers', types=jnp.ndarray, desc='Points representing the mesh kernel')
        self.options.declare('kernel_radii', types=jnp.ndarray, desc='Radii of kernel points')

    def setup(self):

        # Object Inputs
        self.add_input('control_points', shape_by_conn=True)
        self.add_input('radius', shape_by_conn=True)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_heat_loads', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('densities', 'control_points', method='exact')
        self.declare_partials('densities', 'radius', method='exact')

    def compute(self, inputs, outputs):

        # Get the Mesh parameters
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        control_points = jnp.array(inputs['control_points'])
        radius     = jnp.array(inputs['radius'])

        # Compute the pseudo-densities
        densities = self._compute_primal(mesh_centers, mesh_size,
                                                control_points, radius,
                                                kernel_centers, kernel_radii)

        # Write the outputs
        outputs['densities'] = densities

    def compute_partials(self, inputs, partials):

        # Get the Mesh parameters
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        control_points = jnp.array(inputs['control_points'])
        radius = jnp.array(inputs['radius'])

        # Calculate the partial derivatives
        jac_densities = jacrev(self._compute_primal)(mesh_centers, mesh_size,
                                                            control_points, radius,
                                                            kernel_centers, kernel_radii)

        # Set the partial derivatives
        partials['densities', 'control_points'] = jac_densities[2]
        partials['densities', 'radius'] = jac_densities[3]

    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        cyl_points, cyl_radii,
                        kernel_centers, kernel_radii):

        densities, _, _ = project_interconnect(mesh_centers, mesh_size,
                                                      cyl_points, cyl_radii,
                                                      kernel_centers, kernel_radii)

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

