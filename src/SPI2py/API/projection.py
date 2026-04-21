# Standard imports
import numpy as np
from functools import partial
import jax.numpy as jnp
from jax import jvp, vjp
from openmdao.api import ExplicitComponent, Group

# SPI2py imports
from ..models.geometry.cylinders import create_cylinders
from ..models.projection.projection import project_component, project_capsules
from ..models.utilities.aggregation import kreisselmeier_steinhauser_max
from ..models.utilities.visualization import plot_grid


class Projections(Group):
    pass



class ProjectMDBDComponent(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid
    """

    def initialize(self):

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
            t_radii = d_inputs['radii'] if 'radii' in d_inputs else jnp.zeros_like(radii)
            t_heat_load = d_inputs['heat_load'] if 'heat_load' in d_inputs else jnp.zeros_like(heat_load)

            tangents = (t_mesh_centers, t_mesh_size, t_centers, t_radii, t_kernel_centers, t_kernel_radii, t_heat_load)

            # Compute the Jacobian-vector product (JVP) using JAX.
            _, tangent_out = jvp(self._compute_primal, primals, tangents)

            # Unpack and accumulate the outputs.
            d_outputs['densities'] += tangent_out[0]
            d_outputs['penalized_densities'] += tangent_out[1]
            d_outputs['penalized_heat_loads'] += tangent_out[2]

        elif mode == 'rev':
            primal_out, pullback = vjp(self._compute_primal, *primals)

            # The cotangent for outputs is provided in d_outputs.
            cotangent = (d_outputs['densities'],
                         d_outputs['penalized_densities'],
                         d_outputs['penalized_heat_loads'])

            # Compute the pullback (adjoint).
            grads = pullback(cotangent)

            # Only assign derivatives to design inputs.
            # 0 = mesh_centers, 1 = mesh_size,
            # 2 = centers, 3 = radii,
            # 4 = kernel_centers, 5 = kernel_radii,
            # 6 = heat_load
            d_inputs['centers'] += grads[2]
            d_inputs['radii'] += grads[3]
            d_inputs['heat_load'] += grads[6]


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


class ProjectLinearSplineComponent(ExplicitComponent):

    def initialize(self):

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
        self.add_output('densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_densities', compute_shape=lambda shapes: (nx, ny, nz))
        self.add_output('penalized_heat_loads', compute_shape=lambda shapes: (nx, ny, nz))

    def setup_partials(self):
        self.declare_partials('densities', ['start_points', 'end_points', 'radii'], method='exact')
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
        densities, penalized_densities, penalized_heat_loads = self._compute_primal(mesh_centers, mesh_size,
                                                                         kernel_centers, kernel_radii,
                                                                         start_points, end_points, radii,
                                                                         heat_load)

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
        start_points = jnp.array(inputs['start_points'])
        end_points = jnp.array(inputs['end_points'])
        radii = jnp.array(inputs['radii'])
        heat_load = jnp.array(inputs['heat_load'])

        # Freeze the constant (primal) parameters using functools.partial.
        # The _compute_primal function now only expects the design inputs.
        frozen_compute_primal = partial(self._compute_primal,
                                        mesh_centers, mesh_size,
                                        kernel_centers, kernel_radii)

        # Pack design inputs.
        primals = (start_points, end_points, radii, heat_load)

        if mode == 'fwd':
            # Get tangents for design inputs (defaulting to zeros if not provided).
            t_start_points = (jnp.array(d_inputs['start_points'])
                              if 'start_points' in d_inputs else jnp.zeros_like(start_points))
            t_end_points = (jnp.array(d_inputs['end_points'])
                            if 'end_points' in d_inputs else jnp.zeros_like(end_points))
            t_radii = jnp.array(d_inputs['radii']) if 'radii' in d_inputs else jnp.zeros_like(radii)
            t_heat_load = (jnp.array(d_inputs['heat_load'])
                           if 'heat_load' in d_inputs else jnp.zeros_like(heat_load))
            tangents = (t_start_points, t_end_points, t_radii, t_heat_load)

            # Compute the forward Jacobian-vector product.
            _, tangent_out = jvp(frozen_compute_primal, primals, tangents)

            # tangent_out is assumed to be a tuple of (densities_t, penalized_densities_t, penalized_heat_loads_t).
            d_outputs['densities'] += tangent_out[0]
            d_outputs['penalized_densities'] += tangent_out[1]
            d_outputs['penalized_heat_loads'] += tangent_out[2]

        elif mode == 'rev':
            # Compute the vector-Jacobian product (pullback) for the design inputs.
            _, pullback = vjp(frozen_compute_primal, *primals)
            cotangent = (d_outputs['densities'],
                         d_outputs['penalized_densities'],
                         d_outputs['penalized_heat_loads'])
            grads = pullback(cotangent)

            # Assign derivatives only to the design inputs.
            # Here grads is a tuple: (grad_start_points, grad_end_points, grad_radii, grad_heat_load)
            d_inputs['start_points'] += grads[0]
            d_inputs['end_points'] += grads[1]
            d_inputs['radii'] += grads[2]
            d_inputs['heat_load'] += grads[3]


    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        kernel_centers, kernel_radii,
                        start_points, end_points, radii,
                        heat_load):

        densities, penalized_densities = project_capsules(mesh_centers, mesh_size,
                                                          kernel_centers, kernel_radii,
                                                          start_points, end_points, radii)

        # Heat load
        penalized_heat_loads = heat_load * penalized_densities

        return densities, penalized_densities, penalized_heat_loads


class ProjectInterconnect(ExplicitComponent):

    def initialize(self):

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
                                                                                    kernel_centers, kernel_radii,
                                                                                    control_points, radius,
                                                                                    heat_load)

        # Write the outputs
        outputs['densities'] = densities
        outputs['penalized_densities'] = penalized_densities
        outputs['penalized_heat_loads'] = penalized_heat_loads

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode, discrete_inputs=None):

        # Freeze constant mesh and kernel parameters.
        mesh_size = jnp.atleast_1d(self.options['mesh_size'])
        mesh_centers = jnp.array(self.options['mesh_centers'])
        kernel_centers = jnp.array(self.options['kernel_centers'])
        kernel_radii = jnp.array(self.options['kernel_radii'])

        # Get design inputs.
        control_points = jnp.array(inputs['control_points'])
        radius = jnp.array(inputs['radius'])
        heat_load = jnp.array(inputs['heat_load'])

        # Freeze the static parameters in _compute_primal.
        frozen_compute_primal = partial(
            self._compute_primal,
            mesh_centers, mesh_size, kernel_centers, kernel_radii
        )

        # Now, frozen_compute_primal only expects (control_points, radius, heat_load).
        primals = (control_points, radius, heat_load)

        if mode == 'fwd':
            t_control_points = d_inputs['control_points'] if 'control_points' in d_inputs else jnp.zeros_like(
                control_points)
            t_radius = d_inputs['radius'] if 'radius' in d_inputs else jnp.zeros_like(radius)
            t_heat_load = d_inputs['heat_load'] if 'heat_load' in d_inputs else jnp.zeros_like(heat_load)
            tangents = (t_control_points, t_radius, t_heat_load)

            # Compute the forward jacobian-vector product.
            _, tangent_out = jvp(frozen_compute_primal, primals, tangents)
            # Expected tangent_out is a tuple: (densities_t, penalized_densities_t, penalized_heat_loads_t)
            d_outputs['densities'] += tangent_out[0]
            d_outputs['penalized_densities'] += tangent_out[1]
            d_outputs['penalized_heat_loads'] += tangent_out[2]

        elif mode == 'rev':
            _, pullback = vjp(frozen_compute_primal, *primals)
            cotangent = (d_outputs['densities'],
                         d_outputs['penalized_densities'],
                         d_outputs['penalized_heat_loads'])
            grads = pullback(cotangent)
            # Assign derivatives only to design inputs.
            d_inputs['control_points'] += grads[0]
            d_inputs['radius'] += grads[1]
            d_inputs['heat_load'] += grads[2]

    @staticmethod
    def _compute_primal(mesh_centers, mesh_size,
                        kernel_centers, kernel_radii,
                        cyl_points, cyl_radii,
                        heat_load):

        # TODO Fix this
        cyl_radii = cyl_radii[0]

        # Decompose control points into start and end points
        start_points, end_points, radii = create_cylinders(cyl_points, cyl_radii)

        densities, penalized_densities = project_capsules(mesh_centers, mesh_size,
                                                          kernel_centers, kernel_radii,
                                                          start_points, end_points, radii)

        # Heat load
        penalized_heat_loads = heat_load * penalized_densities

        return densities, penalized_densities, penalized_heat_loads

    def draw(self, plotter, subplot, prob):
        centers      = self.options['mesh_centers']
        element_size = self.options['mesh_size']
        densities    = prob.get_val(self.pathname + '.' + 'densities')
        plot_grid(plotter, subplot, centers, element_size, densities=densities)


class ProjectionAggregator(ExplicitComponent):

    def initialize(self):
        self.options.declare('mode', default='individual', values=('individual', 'combined'),
                             desc='Projection aggregation interface to use')
        self.options.declare('n_projections', types=int, desc='Number of precomputed projections', default=0)
        self.options.declare('n_components', types=int, desc='Number of MDBD component primitive inputs', default=0)
        self.options.declare('n_interconnects', types=int, desc='Number of interconnect primitive inputs', default=0)
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)

        # Mesh parameters
        self.options.declare('mesh_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('kernel_centers', default=None, types=(jnp.ndarray, np.ndarray, type(None)),
                             desc='Points representing the mesh kernel for combined mode')
        self.options.declare('kernel_radii', default=None, types=(jnp.ndarray, np.ndarray, type(None)),
                             desc='Radii of kernel points for combined mode')

    def setup(self):
        mode = self.options['mode']
        n_projections = self.options['n_projections']
        n_components = self.options['n_components']
        n_interconnects = self.options['n_interconnects']

        if mode == 'individual':
            if n_projections < 1:
                raise ValueError("ProjectionAggregator individual mode requires n_projections >= 1.")

            for i in range(n_projections):
                self.add_input(f'densities_{i}', shape_by_conn=True)
                self.add_input(f'heat_loads_{i}', shape_by_conn=True)

            self.add_output('aggregated_densities', copy_shape='densities_0')
            self.add_output('aggregated_heat_loads', copy_shape='heat_loads_0')

        else:
            if n_components + n_interconnects < 1:
                raise ValueError("ProjectionAggregator combined mode requires at least one primitive input.")
            if self.options['kernel_centers'] is None or self.options['kernel_radii'] is None:
                raise ValueError("ProjectionAggregator combined mode requires kernel_centers and kernel_radii.")

            for i in range(n_components):
                self.add_input(f'component_centers_{i}', shape_by_conn=True)
                self.add_input(f'component_radii_{i}', shape_by_conn=True)
                self.add_input(f'component_heat_load_{i}', val=0.0)

            for i in range(n_interconnects):
                self.add_input(f'interconnect_points_{i}', shape_by_conn=True)
                self.add_input(f'interconnect_radius_{i}', shape_by_conn=True)
                self.add_input(f'interconnect_heat_load_{i}', val=0.0)

            nx, ny, nz = self.options['mesh_centers'].shape[:3]
            self.add_output('aggregated_densities', shape=(nx, ny, nz))
            self.add_output('aggregated_heat_loads', shape=(nx, ny, nz))

        self.add_output('max_density', val=0.0)

    def setup_partials(self):
        mode = self.options['mode']
        n_projections = self.options['n_projections']
        n_components = self.options['n_components']
        n_interconnects = self.options['n_interconnects']

        if mode == 'individual':
            for i in range(n_projections):
                self.declare_partials('aggregated_densities', f'densities_{i}')
                self.declare_partials('aggregated_heat_loads', f'heat_loads_{i}')
                self.declare_partials('max_density', f'densities_{i}')
        else:
            for i in range(n_components):
                self.declare_partials('aggregated_densities', f'component_centers_{i}')
                self.declare_partials('aggregated_densities', f'component_radii_{i}')
                self.declare_partials('aggregated_heat_loads', f'component_centers_{i}')
                self.declare_partials('aggregated_heat_loads', f'component_radii_{i}')
                self.declare_partials('aggregated_heat_loads', f'component_heat_load_{i}')
                self.declare_partials('max_density', f'component_centers_{i}')
                self.declare_partials('max_density', f'component_radii_{i}')

            for i in range(n_interconnects):
                self.declare_partials('aggregated_densities', f'interconnect_points_{i}')
                self.declare_partials('aggregated_densities', f'interconnect_radius_{i}')
                self.declare_partials('aggregated_heat_loads', f'interconnect_points_{i}')
                self.declare_partials('aggregated_heat_loads', f'interconnect_radius_{i}')
                self.declare_partials('aggregated_heat_loads', f'interconnect_heat_load_{i}')
                self.declare_partials('max_density', f'interconnect_points_{i}')
                self.declare_partials('max_density', f'interconnect_radius_{i}')


    def compute(self, inputs, outputs):
        rho_min = self.options['rho_min']

        if self.options['mode'] == 'individual':
            densities, heat_loads = self._individual_inputs(inputs)
            aggregated_densities, aggregated_heat_loads, max_density = self._compute_individual_primal(
                densities, heat_loads, rho_min)
        else:
            primals = self._combined_inputs(inputs)
            aggregated_densities, aggregated_heat_loads, max_density = self._compute_combined_primal(
                jnp.array(self.options['mesh_centers']),
                jnp.atleast_1d(self.options['mesh_size']),
                jnp.array(self.options['kernel_centers']),
                jnp.array(self.options['kernel_radii']),
                *primals,
                rho_min)

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
        if self.options['mode'] == 'individual':
            self._compute_individual_jacvec_product(inputs, d_inputs, d_outputs, mode)
        else:
            self._compute_combined_jacvec_product(inputs, d_inputs, d_outputs, mode)

    def _individual_inputs(self, inputs):
        n_projections = self.options['n_projections']
        densities = [jnp.array(inputs[f'densities_{i}']) for i in range(n_projections)]
        heat_loads = [jnp.array(inputs[f'heat_loads_{i}']) for i in range(n_projections)]
        return densities, heat_loads

    def _combined_inputs(self, inputs):
        n_components = self.options['n_components']
        n_interconnects = self.options['n_interconnects']

        component_centers = [jnp.array(inputs[f'component_centers_{i}']) for i in range(n_components)]
        component_radii = [jnp.array(inputs[f'component_radii_{i}']) for i in range(n_components)]
        component_heat_loads = [jnp.array(inputs[f'component_heat_load_{i}']) for i in range(n_components)]
        interconnect_points = [jnp.array(inputs[f'interconnect_points_{i}']) for i in range(n_interconnects)]
        interconnect_radii = [jnp.array(inputs[f'interconnect_radius_{i}']) for i in range(n_interconnects)]
        interconnect_heat_loads = [jnp.array(inputs[f'interconnect_heat_load_{i}']) for i in range(n_interconnects)]

        return (component_centers, component_radii, component_heat_loads,
                interconnect_points, interconnect_radii, interconnect_heat_loads)

    def _compute_individual_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']
        densities, heat_loads = self._individual_inputs(inputs)
        primals = (densities, heat_loads, rho_min)

        if mode == 'fwd':
            tan_densities = [
                jnp.array(d_inputs[f'densities_{i}'])
                if f'densities_{i}' in d_inputs else jnp.zeros_like(densities[i])
                for i in range(n_projections)
            ]
            tan_heat_loads = [
                jnp.array(d_inputs[f'heat_loads_{i}'])
                if f'heat_loads_{i}' in d_inputs else jnp.zeros_like(heat_loads[i])
                for i in range(n_projections)
            ]
            tangents = (tan_densities, tan_heat_loads, jnp.zeros_like(rho_min))
            _, tangent_out = jvp(self._compute_individual_primal, primals, tangents)

            d_outputs['aggregated_densities'] += tangent_out[0]
            d_outputs['aggregated_heat_loads'] += tangent_out[1]
            d_outputs['max_density'] += tangent_out[2]

        elif mode == 'rev':
            _, pullback = vjp(self._compute_individual_primal, *primals)
            grads = pullback((d_outputs['aggregated_densities'],
                              d_outputs['aggregated_heat_loads'],
                              d_outputs['max_density']))
            for i in range(n_projections):
                d_inputs[f'densities_{i}'] += grads[0][i]
                d_inputs[f'heat_loads_{i}'] += grads[1][i]

    def _compute_combined_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
        n_components = self.options['n_components']
        n_interconnects = self.options['n_interconnects']
        rho_min = self.options['rho_min']
        primals = self._combined_inputs(inputs)

        frozen_compute_primal = partial(
            self._compute_combined_primal,
            jnp.array(self.options['mesh_centers']),
            jnp.atleast_1d(self.options['mesh_size']),
            jnp.array(self.options['kernel_centers']),
            jnp.array(self.options['kernel_radii']),
            rho_min=rho_min)

        if mode == 'fwd':
            tangents = self._combined_tangents(d_inputs, primals)
            _, tangent_out = jvp(frozen_compute_primal, primals, tangents)

            d_outputs['aggregated_densities'] += tangent_out[0]
            d_outputs['aggregated_heat_loads'] += tangent_out[1]
            d_outputs['max_density'] += tangent_out[2]

        elif mode == 'rev':
            _, pullback = vjp(frozen_compute_primal, *primals)
            grads = pullback((d_outputs['aggregated_densities'],
                              d_outputs['aggregated_heat_loads'],
                              d_outputs['max_density']))

            for i in range(n_components):
                self._add_if_present(d_inputs, f'component_centers_{i}', grads[0][i])
                self._add_if_present(d_inputs, f'component_radii_{i}', grads[1][i])
                self._add_if_present(d_inputs, f'component_heat_load_{i}', grads[2][i])

            for i in range(n_interconnects):
                self._add_if_present(d_inputs, f'interconnect_points_{i}', grads[3][i])
                self._add_if_present(d_inputs, f'interconnect_radius_{i}', grads[4][i])
                self._add_if_present(d_inputs, f'interconnect_heat_load_{i}', grads[5][i])

    def _combined_tangents(self, d_inputs, primals):
        (component_centers, component_radii, component_heat_loads,
         interconnect_points, interconnect_radii, interconnect_heat_loads) = primals

        return (
            self._tangent_list(d_inputs, 'component_centers', component_centers),
            self._tangent_list(d_inputs, 'component_radii', component_radii),
            self._tangent_list(d_inputs, 'component_heat_load', component_heat_loads),
            self._tangent_list(d_inputs, 'interconnect_points', interconnect_points),
            self._tangent_list(d_inputs, 'interconnect_radius', interconnect_radii),
            self._tangent_list(d_inputs, 'interconnect_heat_load', interconnect_heat_loads),
        )

    @staticmethod
    def _tangent_list(d_inputs, prefix, values):
        return [
            jnp.array(d_inputs[f'{prefix}_{i}'])
            if f'{prefix}_{i}' in d_inputs else jnp.zeros_like(value)
            for i, value in enumerate(values)
        ]

    @staticmethod
    def _add_if_present(d_inputs, name, value):
        if name in d_inputs:
            d_inputs[name] += value

    @staticmethod
    def _compute_individual_primal(densities, heat_loads, rho_min):
        aggregated_densities = jnp.sum(jnp.stack(densities, axis=0), axis=0)
        aggregated_heat_loads = jnp.sum(jnp.stack(heat_loads, axis=0), axis=0)

        aggregated_densities = jnp.where(aggregated_densities < rho_min, rho_min, aggregated_densities)
        max_density = kreisselmeier_steinhauser_max(aggregated_densities.flatten(), rho=100)

        return aggregated_densities, aggregated_heat_loads, max_density

    @staticmethod
    def _compute_primal(densities, heat_loads, rho_min):
        return ProjectionAggregator._compute_individual_primal(densities, heat_loads, rho_min)

    @staticmethod
    def _compute_combined_primal(mesh_centers, mesh_size,
                                 kernel_centers, kernel_radii,
                                 component_centers, component_radii, component_heat_loads,
                                 interconnect_points, interconnect_radii, interconnect_heat_loads,
                                 rho_min):
        density_fields = []
        heat_load_fields = []

        for centers, radii, heat_load in zip(component_centers, component_radii, component_heat_loads):
            _, penalized_densities = project_component(mesh_centers, mesh_size,
                                                       centers, radii,
                                                       kernel_centers, kernel_radii)
            density_fields.append(penalized_densities)
            heat_load_fields.append(heat_load * penalized_densities)

        for points, radius, heat_load in zip(interconnect_points, interconnect_radii, interconnect_heat_loads):
            start_points, end_points, radii = create_cylinders(points, radius)
            _, penalized_densities = project_capsules(mesh_centers, mesh_size,
                                                      kernel_centers, kernel_radii,
                                                      start_points, end_points, radii)
            density_fields.append(penalized_densities)
            heat_load_fields.append(heat_load * penalized_densities)

        return ProjectionAggregator._compute_individual_primal(density_fields, heat_load_fields, rho_min)

    def draw(self, plotter, subplot, prob):
        centers      = self.options['mesh_centers']
        element_size = self.options['mesh_size']
        densities    = prob.get_val(self.pathname + '.' + 'aggregated_densities')
        plot_grid(plotter, subplot, centers, element_size, densities=densities)

