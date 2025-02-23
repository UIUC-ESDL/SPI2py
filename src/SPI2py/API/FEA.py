from jax import numpy as jnp
from openmdao.core.indepvarcomp import IndepVarComp

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
        # nodes, elements = generate_mesh_vec(nx, ny, nz, lx, ly, lz)
        nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh_vec(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
        centers = centers.reshape(nx, ny, nz, 1, 3)

        # Read the MDBD kernel
        uniform_8_kernel_positions, uniform_8_kernel_radii = create_uniform_kernel(1, mode='circumscription')
        kernel_positions = jnp.array(uniform_8_kernel_positions)
        kernel_radii = jnp.array(uniform_8_kernel_radii).reshape(-1, 1)

        # # Calculate the kernel volume fraction
        volume_element = element_size ** 3


        # Declare the outputs
        self.add_output('element_size', val=element_size)
        self.add_output('mesh_centers', val=centers)
        self.add_output('mesh_nodes', val=nodes)
        self.add_output('mesh_elements', val=elements)
        self.add_output('n_el_x', val=nx)
        self.add_output('n_el_y', val=ny)
        self.add_output('n_el_z', val=nz)

        # Outputs for additional info
        # self.add_output('element_volume', val=volume_element)
        # self.add_output('kernel_volume', val=volume_kernel)
        # self.add_output('volume_approximation_error', val=volume_approximation_error)
