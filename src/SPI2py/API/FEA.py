import jax.numpy as jnp
import jax
from jax import jvp, vjp
from jax.scipy.sparse.linalg import cg
from jax.experimental.sparse import coo_fromdense
from openmdao.api import ExplicitComponent, IndepVarComp

from SPI2py.models.physics.distributed.mesh import generate_mesh_vec
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.assembly import assemble_global_stiffness_matrix, apply_boundary_conditions
from SPI2py.models.physics.distributed.assembly import DirichletBC, RobinBC

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


class ExplicitFEA(ExplicitComponent):

    def initialize(self):

        # Mesh parameters
        self.options.declare('nodes', types=jnp.ndarray, desc='Nodes of the mesh')
        self.options.declare('elements', types=jnp.ndarray, desc='Elements of the mesh')
        self.options.declare('el_size', types=(int, float), desc='Size of the mesh elements', default=1.0)
        self.options.declare('el_centers', types=jnp.ndarray, desc='Centers of the mesh elements')
        self.options.declare('dirichlet_nodes', types=jnp.ndarray)
        self.options.declare('dirichlet_values', types=jnp.ndarray)
        self.options.declare('robin_nodes', types=jnp.ndarray)
        self.options.declare('robin_h', types=float)
        self.options.declare('robin_T_inf', types=float)
        self.options.declare('robin_area', types=float)

    def setup(self):


        self.add_input("density", shape_by_conn=True, desc="Material density")
        self.add_input("heat_loads", shape_by_conn=True)

        n_el = self.options['nodes'].shape[0]
        self.add_output("temperature", shape=(n_el,), desc="Computed temperature field")

    # def setup_partials(self):
    #     # Declare that we are using matrix-free derivatives
    #     self.declare_partials(of="temperature", wrt="density", method="exact")
    #     self.declare_partials(of="temperature", wrt="heat_loads", method="exact")

    def compute(self, inputs, outputs):

        # Unpack the options
        nodes = self.options['nodes']
        elements = self.options['elements']
        el_size = self.options['el_size']
        el_centers = self.options['el_centers']
        dirichlet_nodes = self.options['dirichlet_nodes']
        dirichlet_values = self.options['dirichlet_values']
        robin_nodes = self.options['robin_nodes']
        robin_h = self.options['robin_h']
        robin_T_inf = self.options['robin_T_inf']
        robin_area = self.options['robin_area']

        # Unpack the inputs
        density = jnp.array(inputs["density"])
        heat_loads = jnp.array(inputs["heat_loads"])

        T = self._compute_primal(density, heat_loads, nodes, elements, dirichlet_nodes, robin_nodes, el_size)

        outputs["temperature"] = T

    # def compute_jacvec_prod(self, inputs, d_inputs, d_outputs, mode):
    #     """ Computes matrix-free vector-Jacobian products (VJP). """
    #     density = inputs["density"]
    #
    #     if mode == "fwd":
    #         if "density" in d_inputs:
    #             d_outputs["temperature"] += self.jvp_fea(density, d_inputs["density"])
    #
    #     elif mode == "rev":
    #         if "temperature" in d_outputs:
    #             d_inputs["density"] += self.vjp_fea(density, d_outputs["temperature"])

    @staticmethod
    def _compute_primal(density, heat_loads, nodes, elements, dirichlet_nodes, robin_nodes, el_size):

        density = density.flatten()

        base_k = 1.0



        dirichlet_bc_1 = DirichletBC(dirichlet_nodes, T=200)
        robin_bc_1 = RobinBC(robin_nodes, h=10.0, T_inf=200, area=el_size**2)
        boundary_conditions = [dirichlet_bc_1, robin_bc_1]

        # Assemble the global stiffness matrix and load vector.
        K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k)

        # # Convert heat loads from element-wise to node-wise
        # # For each element, add (heat_load/number_of_nodes) to each of its 8 nodes.
        # # heat_loads = heat_loads.flatten()
        # nodes_per_elem = 8
        # heat_load_per_element = 1.0
        # element_contrib = (heat_load_per_element * density) / nodes_per_elem
        # elem_contrib_flat = element_contrib.flatten()
        # node_contrib = jnp.repeat(elem_contrib_flat, nodes_per_elem)
        # f = f.at[elements.flatten()].add(node_contrib)
        # f = heat_loads

        # Assume 'heat_loads' is an array of shape (n_el,) with the heat load for each element.
        nodes_per_elem = 8
        heat_loads = heat_loads.flatten()
        # 'density' is assumed to be an array of shape (n_el,) or broadcastable to it.
        element_contrib = (heat_loads * density) / nodes_per_elem
        elem_contrib_flat = element_contrib.flatten()
        node_contrib = jnp.repeat(elem_contrib_flat, nodes_per_elem)
        f = f.at[elements.flatten()].add(node_contrib)

        # Apply the boundary conditions and partition the system.
        K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, idx_f, idx_p = apply_boundary_conditions(K, f, boundary_conditions)

        # Solve the partitioned system for the unknown displacements.
        # K_ff @ u_f + K_fp @ u_p = f_f
        # K_ff @ u_f = f_f - K_fp @ u_p
        # u_f = K_ff^-1 @ (f_f - K_fp @ u_p)
        # u_f = jnp.linalg.solve(K_ff, f_f - K_fp @ u_p)

        # Convert K_ff to a sparse format for efficient solving
        # K_ff = BCOO.from_scipy_sparse(coo_matrix(K_ff))
        K_ff = coo_fromdense(K_ff)

        # K_ff = BCOO.fromdense(K_ff)

        # Solve the partitioned system for the unknown displacements using Conjugate Gradient (CG)
        def fea_solve(rhs):
            u_f, _ = cg(K_ff, rhs, tol=1e-8, maxiter=500)
            return u_f

        u_f = fea_solve(f_f - K_fp @ u_p)  # Solving K_ff @ u_f = (f_f - K_fp @ u_p)

        # Reassemble the full solution.
        n_nodes = K.shape[0]
        u = jnp.zeros(n_nodes)
        u = u.at[idx_f].set(u_f)
        u = u.at[idx_p].set(u_p)

        return u

    # def jvp_fea(self, density, d_density):
    #     """ Computes forward-mode JVP for FEA. """
    #     jvp_fn = jvp(self.fea_solve, (density,), (d_density,))
    #     return jvp_fn[1]  # Extracts the JVP result
    #
    # def vjp_fea(self, density, d_temperature):
    #     """ Computes reverse-mode VJP for FEA. """
    #     vjp_fn = vjp(self.fea_solve, density)[1]
    #     return vjp_fn(d_temperature)[0]  # Extracts the VJP result


class BoundaryConditionAggregator(ExplicitComponent):
    pass