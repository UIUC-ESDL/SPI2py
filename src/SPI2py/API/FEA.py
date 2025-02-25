from jax import numpy as jnp

from openmdao.api import ExplicitComponent, IndepVarComp

from SPI2py.models.physics.distributed.mesh import generate_mesh_vec
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel


class Mesh(IndepVarComp):
    def initialize(self):

        self.options.declare('x_bounds', types=tuple, desc='X Bounds of the mesh')
        self.options.declare('y_bounds', types=tuple, desc='Y Bounds of the mesh')
        self.options.declare('z_bounds', types=tuple, desc='Z Bounds of the mesh')
        self.options.declare('element_size', types=(int, float), desc='Size of the mesh elements')

    def setup(self):

        # Get the options
        x_min, x_max = self.options['x_bounds']
        y_min, y_max = self.options['y_bounds']
        z_min, z_max = self.options['z_bounds']
        element_size = self.options['element_size']

        # Define the mesh grid positions
        nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh_vec(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
        centers = centers.reshape(nx, ny, nz, 1, 3)

        # Calculate the kernel volume fraction
        volume_element = element_size ** 3

        # Declare the outputs
        self.add_output('element_size', val=element_size)
        self.add_output('element_volume', val=volume_element)
        self.add_output('mesh_centers', val=centers)
        self.add_output('mesh_nodes', val=nodes)
        self.add_output('mesh_elements', val=elements)
        self.add_output('n_el_x', val=nx)
        self.add_output('n_el_y', val=ny)
        self.add_output('n_el_z', val=nz)


class FEA(ExplicitComponent):

    def initialize(self):

        # General parameters


        # Mesh parameters
        self.options.declare('element_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('mesh_centers', types=jnp.ndarray, desc='Centers of the mesh elements')


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
        sphere_radii = jnp.array(inputs['sphere_radii'])

        # Compute the pseudo-densities
        pseudo_densities = self._compute_primal(mesh_centers, element_size, sphere_positions, sphere_radii,
                                                kernel_points, kernel_radii)

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
