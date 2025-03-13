import jax.numpy as jnp
import jax
from jax import jvp, vjp
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


class FEASolverComponent(ExplicitComponent):
    def setup(self):
        # Example FEA system with 10 nodes
        n = 10
        self.add_input("density", shape=(n,), desc="Material density")
        self.add_output("temperature", shape=(n,), desc="Computed temperature field")

        # Declare that we are using matrix-free derivatives
        self.declare_partials(of="temperature", wrt="density", method="cs")  # Using complex-step initially
        self.declare_partials(of="temperature", wrt="density", method="exact")  # Exact derivatives

    def compute(self, inputs, outputs):
        """ Compute the FEA solution. """
        density = inputs["density"]
        outputs["temperature"] = self.fea_solve(density)

    def compute_jacvec_prod(self, inputs, d_inputs, d_outputs, mode):
        """ Computes matrix-free vector-Jacobian products (VJP). """
        density = inputs["density"]

        if mode == "fwd":
            if "density" in d_inputs:
                d_outputs["temperature"] += self.jvp_fea(density, d_inputs["density"])

        elif mode == "rev":
            if "temperature" in d_outputs:
                d_inputs["density"] += self.vjp_fea(density, d_outputs["temperature"])

    def fea_solve(self, density):
        """ Example: Simple 1D FEA system with a stiffness matrix. """
        K = jnp.diag(jnp.full(density.shape[0], 2.0)) - jnp.diag(jnp.full(density.shape[0] - 1, 1.0), k=1) - jnp.diag(
            jnp.full(density.shape[0] - 1, 1.0), k=-1)
        f = jnp.ones_like(density)  # Heat source term
        u = jax.scipy.sparse.linalg.cg(K, f)[0]  # Solve Ku = f
        return u

    def jvp_fea(self, density, d_density):
        """ Computes forward-mode JVP for FEA. """
        jvp_fn = jvp(self.fea_solve, (density,), (d_density,))
        return jvp_fn[1]  # Extracts the JVP result

    def vjp_fea(self, density, d_temperature):
        """ Computes reverse-mode VJP for FEA. """
        vjp_fn = vjp(self.fea_solve, density)[1]
        return vjp_fn(d_temperature)[0]  # Extracts the VJP result


