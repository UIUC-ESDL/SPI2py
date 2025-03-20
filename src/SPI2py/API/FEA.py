import jax.numpy as jnp
import jax
from jax import jvp, vjp
from jax.scipy.sparse.linalg import cg
from jax.experimental.sparse import BCOO
from jax.experimental.sparse import coo_fromdense
from openmdao.api import ExplicitComponent, IndepVarComp

from SPI2py.models.physics.distributed.mesh import generate_mesh_vec
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.assembly import assemble_global_stiffness_matrix, apply_boundary_conditions
from SPI2py.models.physics.distributed.assembly import DirichletBC, RobinBC
from SPI2py.models.utilities.aggregation import kreisselmeier_steinhauser_max, kreisselmeier_steinhauser_min


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

    def setup(self):

        self.add_input("density", shape_by_conn=True, desc="Material density")
        self.add_input("heat_loads", shape_by_conn=True)

        n_el = self.options['nodes'].shape[0]
        self.add_output("temperature", shape=(n_el,), desc="Computed temperature field")
        self.add_output("max_temperature", val=0.0, desc="Computed maximum temperature")

    def setup_partials(self):
        # Declare that we are using matrix-free derivatives
        self.declare_partials(of="temperature", wrt="density", method="exact")
        self.declare_partials(of="temperature", wrt="heat_loads", method="exact")
        self.declare_partials(of="max_temperature", wrt="density", method="exact")
        self.declare_partials(of="max_temperature", wrt="heat_loads", method="exact")

    def compute(self, inputs, outputs):

        # Unpack the options
        nodes = self.options['nodes']
        elements = self.options['elements']
        el_size = self.options['el_size']
        el_centers = self.options['el_centers']
        robin_nodes = self.options['robin_nodes']
        robin_h = self.options['robin_h']
        robin_T_inf = self.options['robin_T_inf']
        dirichlet_nodes = self.options['dirichlet_nodes']
        dirichlet_values = self.options['dirichlet_values']

        robin_area = el_size ** 2

        # Unpack the inputs
        density = jnp.array(inputs["density"])
        heat_loads = jnp.array(inputs["heat_loads"])

        temp, max_temp = self._compute_primal(density, heat_loads, nodes, elements,
                                    robin_nodes, robin_h, robin_T_inf, robin_area,
                                    dirichlet_nodes, dirichlet_values)

        # def f(density, heat_loads, nodes, elements,
        #       robin_nodes, robin_h, robin_T_inf, robin_area,
        #       dirichlet_nodes, dirichlet_values):
        #     temp, max_temp = self._compute_primal(density, heat_loads, nodes, elements,
        #                                           robin_nodes, robin_h, robin_T_inf, robin_area,
        #                                           dirichlet_nodes, dirichlet_values)
        #     return max_temp
        # from jax import jacrev
        # j = jacrev(f, argnums=(0, 1))
        # j(density, heat_loads, nodes, elements,
        #   robin_nodes, robin_h, robin_T_inf, robin_area,
        #   dirichlet_nodes, dirichlet_values)
        outputs["temperature"] = temp
        outputs["max_temperature"] = max_temp

    @staticmethod
    def _compute_primal(density, heat_loads, nodes, elements,
                        r_nodes, r_h, r_T_inf, r_area,
                        d_nodes, d_T):

        # Flatten the inputs
        density = density.flatten()
        heat_loads = heat_loads.flatten()

        # Set the base thermal conductivity
        # TODO What?
        base_k = 1.0


        # Assemble the global stiffness matrix and load vector.
        K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k)

        # Apply the heat loads to the system.
        nodes_per_elem = 8
        element_contrib = (heat_loads * density) / nodes_per_elem
        node_contrib = jnp.repeat(element_contrib, nodes_per_elem)
        f = f.at[elements.flatten()].add(node_contrib)

        # Apply the boundary conditions and partition the system.
        K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, idx_f, idx_p = apply_boundary_conditions(K, f,
                                                                                        r_nodes, r_h, r_T_inf, r_area,
                                                                                        d_nodes, d_T)

        # Solve the partitioned system for the unknown displacements.
        # K_ff @ u_f + K_fp @ u_p = f_f
        # K_ff @ u_f = f_f - K_fp @ u_p
        # u_f = K_ff^-1 @ (f_f - K_fp @ u_p)
        # u_f = jnp.linalg.solve(K_ff, f_f - K_fp @ u_p)

        # Convert K_ff to a sparse format for efficient solving
        # K_ff = BCOO.from_scipy_sparse(coo_matrix(K_ff))
        # K_ff = coo_fromdense(K_ff)
        K_ff = BCOO.fromdense(K_ff)

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

        # Calculate the max temp
        # u_max = kreisselmeier_steinhauser_max(u)
        # TODO Reset
        u_max = kreisselmeier_steinhauser_min(u)

        return u, u_max

    def compute_jacvec_prod(self, inputs, d_inputs, d_outputs, mode):
        """
        Compute the matrix-free Jacobian-vector product for both outputs:
          - "temperature" (full field) and
          - "max_temperature" (a scalar, e.g., from a Kreisselmeier–Steinhauser function).
        """
        # Get current input values.
        density = jnp.array(inputs["density"])
        heat_loads = jnp.array(inputs["heat_loads"])

        # Unpack options.
        nodes = self.options['nodes']
        elements = self.options['elements']
        el_size = self.options['el_size']
        robin_nodes = self.options['robin_nodes']
        robin_h = self.options['robin_h']
        robin_T_inf = self.options['robin_T_inf']
        dirichlet_nodes = self.options['dirichlet_nodes']
        dirichlet_values = self.options['dirichlet_values']
        robin_area = el_size ** 2

        # Prepare the full set of "primal" arguments.
        primals = (density, heat_loads, nodes, elements,
                   robin_nodes, robin_h, robin_T_inf, robin_area,
                   dirichlet_nodes, dirichlet_values)

        if mode == "fwd":
            # Build tangent (directional) inputs only for the active variables.
            tan_density = jnp.array(d_inputs["density"]) if "density" in d_inputs and d_inputs[
                "density"] is not None else jnp.zeros_like(density)
            tan_heat_loads = jnp.array(d_inputs["heat_loads"]) if "heat_loads" in d_inputs and d_inputs[
                "heat_loads"] is not None else jnp.zeros_like(heat_loads)
            # For all other parameters, the directional derivative is zero.
            tan_nodes = jnp.zeros_like(nodes)
            tan_elements = jnp.zeros_like(elements)
            tan_robin_nodes = jnp.zeros_like(robin_nodes)
            tan_robin_h = 0.0
            tan_robin_T_inf = 0.0
            tan_robin_area = 0.0  # since robin_area is computed from el_size
            tan_dirichlet_nodes = jnp.zeros_like(dirichlet_nodes)
            tan_dirichlet_values = jnp.zeros_like(dirichlet_values)
            tangents = (tan_density, tan_heat_loads, tan_nodes, tan_elements,
                        tan_robin_nodes, tan_robin_h, tan_robin_T_inf, tan_robin_area,
                        tan_dirichlet_nodes, tan_dirichlet_values)
            # Compute forward-mode JVP. Since _compute_primal returns a tuple (u, u_max),
            # tangent_out will be a tuple with the corresponding directional derivatives.
            _, tangent_out = jvp(self._compute_primal, primals, tangents)
            d_outputs["temperature"] = tangent_out[0]
            d_outputs["max_temperature"] = tangent_out[1]

        elif mode == "rev":
            # Compute the reverse-mode VJP.
            # _compute_primal returns a tuple (u, u_max)
            primal_out, pullback = vjp(self._compute_primal, *primals)
            cotan_temperature = d_outputs["temperature"] if "temperature" in d_outputs and d_outputs[
                "temperature"] is not None else jnp.zeros_like(primal_out[0])
            cotan_max_temperature = d_outputs["max_temperature"] if "max_temperature" in d_outputs and d_outputs[
                "max_temperature"] is not None else 0.0
            # The pullback now expects a tuple of cotangents.
            grads = pullback((cotan_temperature, cotan_max_temperature))
            # grads is a tuple with gradients for each input argument;
            # we only need to accumulate contributions for our optimization variables:
            # d_inputs["density"] = grads[0]
            # d_inputs["heat_loads"] = grads[1]
            d_inputs["density"] = d_inputs.get("density", 0) + grads[0]
            d_inputs["heat_loads"] = d_inputs.get("heat_loads", 0) + grads[1]


class BoundaryConditionAggregator(ExplicitComponent):
    pass