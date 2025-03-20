import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jacfwd, jvp, vjp
from openmdao.api import ExplicitComponent, Group

from ..models.geometry.cylinders import create_cylinders
from ..models.projection.projection import project_component, project_interconnect, project_capsules
from ..models.utilities.aggregation import kreisselmeier_steinhauser_max


class Projections(Group):
    pass


class ProjectLinearSplineComponent(ExplicitComponent):

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
        self.add_input('start_points', shape_by_conn=True)
        self.add_input('end_points', shape_by_conn=True)
        self.add_input('radii', shape_by_conn=True)
        self.add_input('heat_load', val=0.0)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('penalized_densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_heat_loads', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('penalized_densities', ['start_points', 'end_points', 'radii'], method='exact')
        self.declare_partials('penalized_heat_loads', ['start_points', 'end_points', 'radii','heat_load'], method='exact')

    def compute(self, inputs, outputs):

        # Get the Mesh parameters
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        start_points = jnp.array(inputs['start_points'])
        end_points = jnp.array(inputs['end_points'])
        radii     = jnp.array(inputs['radii'])
        heat_load = jnp.array(inputs['heat_load'])

        # Compute the pseudo-densities
        penalized_densities, penalized_heat_loads = self._compute_primal(mesh_centers, mesh_size,
                                                                         kernel_centers, kernel_radii,
                                                                         start_points, end_points, radii,
                                                                         heat_load)

        # Write the outputs
        outputs['penalized_densities'] = penalized_densities
        outputs['penalized_heat_loads'] = penalized_heat_loads

    def compute_partials(self, inputs, partials):
        # Get the Mesh parameters
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        start_points = jnp.array(inputs['start_points'])
        end_points = jnp.array(inputs['end_points'])
        radii = jnp.array(inputs['radii'])
        heat_load = jnp.array(inputs['heat_load'])


        jac_pd, jac_phl = jacfwd(self._compute_primal, argnums=(4, 5, 6, 7))(mesh_centers, mesh_size,
                                                                            kernel_centers, kernel_radii,
                                                                            start_points, end_points, radii,
                                                                            heat_load)


        # Write the outputs
        partials['penalized_densities', 'start_points'] = jac_pd[0]
        partials['penalized_densities', 'end_points'] = jac_pd[1]
        partials['penalized_densities', 'radii'] = jac_pd[2]
        partials['penalized_heat_loads', 'start_points'] = jac_phl[0]
        partials['penalized_heat_loads', 'end_points'] = jac_phl[1]
        partials['penalized_heat_loads', 'radii'] = jac_phl[2]
        partials['penalized_heat_loads', 'heat_load'] = jac_phl[3]


    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        kernel_centers, kernel_radii,
                        start_points, end_points, radii,
                        heat_load):

        _, penalized_densities = project_capsules(mesh_centers, mesh_size,
                                               kernel_centers, kernel_radii,
                                               start_points, end_points, radii)

        # Heat load
        penalized_heat_loads = heat_load * penalized_densities

        return penalized_densities, penalized_heat_loads



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
        """
        Compute the Jacobian-vector product (JVP) for forward mode and the vector-Jacobian product (VJP) for reverse mode.
        We assume that 'centers' is a constant input so that its tangent is set to zeros.
        """
        # Get mesh and kernel parameters.
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])
        # Get input variables.
        centers = jnp.array(inputs['centers'])
        radii = jnp.array(inputs['radii'])
        heat_load = jnp.array(inputs['heat_load'])
        # Pack the primals in the order expected by _compute_primal.
        primals = (mesh_centers, mesh_size, centers, radii, kernel_centers, kernel_radii, heat_load)

        if mode == 'fwd':
            # For forward mode, define the tangent seeds.
            t_mesh_centers = jnp.zeros_like(mesh_centers)
            t_mesh_size = jnp.zeros_like(mesh_size)
            t_kernel_centers = jnp.zeros_like(kernel_centers)
            t_kernel_radii = jnp.zeros_like(kernel_radii)


            t_centers = d_inputs['centers'] if 'centers' in d_inputs else jnp.zeros_like(centers)
            # For 'radii' and 'heat_load', use the provided tangent if available; otherwise, use zeros.
            t_radii = d_inputs['radii'] if 'radii' in d_inputs else jnp.zeros_like(radii)
            t_heat_load = d_inputs['heat_load'] if 'heat_load' in d_inputs else jnp.zeros_like(heat_load)
            tangents = (t_mesh_centers, t_mesh_size, t_centers, t_radii, t_kernel_centers, t_kernel_radii, t_heat_load)
            # Compute the Jacobian-vector product (JVP) using JAX.
            _, tangent_out = jvp(self._compute_primal, primals, tangents)
            # Unpack and set the outputs.
            d_outputs['densities'] = np.asarray(tangent_out[0])
            d_outputs['penalized_densities'] = np.asarray(tangent_out[1])
            d_outputs['penalized_heat_loads'] = np.asarray(tangent_out[2])
        elif mode == 'rev':
            # Reverse mode: compute the vector-Jacobian product (VJP).
            primal_out, pullback = vjp(self._compute_primal, *primals)
            # The cotangent for outputs is provided in d_outputs.
            cotangent = (d_outputs['densities'],
                         d_outputs['penalized_densities'],
                         d_outputs['penalized_heat_loads'])
            # Compute the pullback (adjoint).
            grads = pullback(cotangent)
            # grads is a tuple corresponding to the derivatives with respect to each input in primals.
            # We only set derivatives for the design inputs.
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
        self.add_input('heat_load', val=0.0)

        # Outputs
        nx, ny, nz = self.options['mesh_centers'].shape[:3]
        self.add_output('densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_heat_loads', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('densities', 'control_points', method='exact')
        self.declare_partials('densities', 'radius', method='exact')
        self.declare_partials('penalized_densities', 'control_points', method='exact')
        self.declare_partials('penalized_densities', 'radius', method='exact')
        self.declare_partials('penalized_heat_loads', 'control_points', method='exact')
        self.declare_partials('penalized_heat_loads', 'radius', method='exact')
        self.declare_partials('penalized_heat_loads', 'heat_load', method='exact')

    def compute(self, inputs, outputs):

        # Get the Mesh parameters
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get the inputs
        control_points = jnp.array(inputs['control_points'])
        radius     = jnp.array(inputs['radius'])
        heat_load = jnp.array(inputs['heat_load'])

        # Compute the pseudo-densities
        densities, penalized_densities, penalized_heat_loads = self._compute_primal(mesh_centers, mesh_size,
                                                control_points, radius,
                                                kernel_centers, kernel_radii, heat_load)

        # Write the outputs
        outputs['densities'] = densities
        outputs['penalized_densities'] = penalized_densities
        outputs['penalized_heat_loads'] = penalized_heat_loads

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        # Get constant mesh parameters.
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])
        # Get design inputs.
        control_points = jnp.array(inputs['control_points'])
        radius = jnp.array(inputs['radius'])
        heat_load = jnp.array(inputs['heat_load'])
        # Pack all inputs in the order expected by _compute_primal.
        primals = (mesh_centers, mesh_size, control_points, radius, kernel_centers, kernel_radii, heat_load)

        if mode == 'fwd':
            # In forward mode, we build a tangent tuple.
            t_mesh_centers = jnp.zeros_like(mesh_centers)
            t_mesh_size = jnp.zeros_like(mesh_size)
            t_kernel_centers = jnp.zeros_like(kernel_centers)
            t_kernel_radii = jnp.zeros_like(kernel_radii)
            # The design-dependent inputs use the provided directional derivatives.
            tangents = (t_mesh_centers,
                        t_mesh_size,
                        d_inputs['control_points'],
                        d_inputs['radius'],
                        t_kernel_centers,
                        t_kernel_radii,
                        d_inputs['heat_load'])
            # jvp returns (primal_out, tangent_out)
            _, tangent_out = jvp(self._compute_primal, primals, tangents)
            # tangent_out is a tuple: (densities_t, penalized_densities_t, penalized_heat_loads_t)
            d_outputs['densities'] = tangent_out[0]
            d_outputs['penalized_densities'] = tangent_out[1]
            d_outputs['penalized_heat_loads'] = tangent_out[2]

        elif mode == 'rev':
            # In reverse mode, use vjp to get the pullback.
            primal_out, pullback = vjp(self._compute_primal, *primals)
            # The cotangent for outputs is provided as a tuple.
            cotangent = (d_outputs['densities'],
                         d_outputs['penalized_densities'],
                         d_outputs['penalized_heat_loads'])
            grads = pullback(cotangent)
            # grads is a tuple with the same order as primals:
            # (mesh_centers, mesh_size, control_points, radius, kernel_centers, kernel_radii, heat_load)
            # Only assign derivatives to design inputs.
            d_inputs['control_points'] = grads[2]
            d_inputs['radius'] = grads[3]
            d_inputs['heat_load'] = grads[6]

    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        cyl_points, cyl_radii,
                        kernel_centers, kernel_radii,
                        heat_load):

        # Decompose control points into start and end points
        start_points, end_points, radii = create_cylinders(cyl_points, cyl_radii)

        densities, penalized_densities = project_interconnect(mesh_centers, mesh_size,
                                                      cyl_points, cyl_radii,
                                                      kernel_centers, kernel_radii)

        # Heat load
        penalized_heat_loads = heat_load * penalized_densities

        return densities, penalized_densities, penalized_heat_loads



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
            self.add_input(f'heat_loads_{i}', shape_by_conn=True)


        # Set the outputs
        self.add_output('aggregated_densities', copy_shape='densities_0')
        self.add_output('aggregated_heat_loads', copy_shape='heat_loads_0')
        self.add_output('max_density', val=0.0)

    def setup_partials(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the partials
        for i in range(n_projections):
            self.declare_partials('aggregated_densities', f'densities_{i}')
            self.declare_partials('aggregated_heat_loads', f'heat_loads_{i}')
            self.declare_partials('max_density', f'densities_{i}')


    def compute(self, inputs, outputs):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        densities = [jnp.array(inputs[f'densities_{i}']) for i in range(n_projections)]
        heat_loads = [jnp.array(inputs[f'heat_loads_{i}']) for i in range(n_projections)]

        # Calculate the values
        aggregated_densities, aggregated_heat_loads, max_density = self._compute_primal(densities, heat_loads, rho_min)

        # Write the outputs
        outputs['aggregated_densities'] = aggregated_densities
        outputs['aggregated_heat_loads'] = aggregated_heat_loads
        outputs['max_density'] = max_density

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):
        """
        Compute the Jacobian-vector product (forward mode) or the
        vector-Jacobian product (reverse mode) for the aggregator.
        The primals are:
            densities_list, heat_loads_list, rho_min
        and the outputs are:
            aggregated_densities, aggregated_heat_loads, max_density.
        """
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Assemble the list inputs from the OpenMDAO inputs dictionary.
        densities = [jnp.array(inputs[f'densities_{i}']) for i in range(n_projections)]
        heat_loads = [jnp.array(inputs[f'heat_loads_{i}']) for i in range(n_projections)]

        # Our _compute_primal function takes a tuple: (densities, heat_loads, rho_min)
        primals = (densities, heat_loads, rho_min)

        if mode == 'fwd':
            # In forward mode, build the corresponding tangent tuple.
            # For each densities input, the tangent is provided in d_inputs.
            tan_densities = [jnp.array(d_inputs[f'densities_{i}']) for i in range(n_projections)]
            tan_heat_loads = [jnp.array(d_inputs[f'heat_loads_{i}']) for i in range(n_projections)]
            tan_rho_min = jnp.zeros_like(rho_min)  # assume rho_min is constant
            tangents = (tan_densities, tan_heat_loads, tan_rho_min)

            # Compute the forward Jacobian-vector product.
            _, tangent_out = jvp(self._compute_primal, primals, tangents)

            # tangent_out is a tuple with three entries.
            d_outputs['aggregated_densities'] = tangent_out[0]
            d_outputs['aggregated_heat_loads'] = tangent_out[1]
            d_outputs['max_density'] = tangent_out[2]

        elif mode == 'rev':
            # In reverse mode, use the VJP (vector-Jacobian product).
            primal_out, pullback = vjp(self._compute_primal, *primals)
            # d_outputs contains cotangents for each output.
            cotan_agg_dens = d_outputs['aggregated_densities']
            cotan_agg_heat = d_outputs['aggregated_heat_loads']
            cotan_max = d_outputs['max_density']
            # Call pullback with a tuple of cotangents.
            grads = pullback((cotan_agg_dens, cotan_agg_heat, cotan_max))
            # grads is a tuple with three entries corresponding to:
            # 0: gradient with respect to densities (which is a list of arrays)
            # 1: gradient with respect to heat_loads (list of arrays)
            # 2: gradient with respect to rho_min (scalar)
            grad_densities, grad_heat_loads, grad_max_density = grads
            # TODO Remove for-loops
            for i in range(n_projections):
                d_inputs[f'densities_{i}'] = grad_densities[i]
                d_inputs[f'heat_loads_{i}'] = grad_heat_loads[i]

            # If there are other inputs (e.g., element_length) that are not varied, assign zeros as needed.

    @staticmethod
    def _compute_primal(densities, heat_loads, rho_min):

        # Aggregate the pseudo-densities
        # aggregated_densities = jnp.zeros_like(densities[0])
        # aggregated_heat_loads = jnp.zeros_like(heat_loads[0])
        # for density in densities:
        #     aggregated_densities += density

        aggregated_densities = jnp.sum(jnp.stack(densities, axis=0), axis=0)
        aggregated_heat_loads = jnp.sum(jnp.stack(heat_loads, axis=0), axis=0)

        # Ensure that no pseudo-density is below the minimum value
        # aggregated_densities = jnp.maximum(aggregated_densities, rho_min)
        aggregated_densities = jnp.where(aggregated_densities < rho_min, rho_min, aggregated_densities)

        # Calculate the maximum pseudo-density
        max_density = kreisselmeier_steinhauser_max(aggregated_densities.flatten(), rho=100)
        # Manual TODO Change
        # max_density = aggregated_densities.flatten()[132:133]

        # TODO Min
        # TODO Max above 1?

        return aggregated_densities, aggregated_heat_loads, max_density

