import numpy as np
from functools import partial
import jax.numpy as jnp
import jax
from jax import jvp, vjp
from jax.scipy.sparse.linalg import cg
from jax.experimental.sparse import BCOO
from jax.experimental.sparse import coo_fromdense
from openmdao.api import ExplicitComponent, IndepVarComp
# from scipy.sparse.linalg import spsolve

from SPI2py.models.physics.distributed.mesh import generate_mesh_vec
from SPI2py.models.projection.mesh_kernels import create_uniform_kernel
from SPI2py.models.physics.distributed.assembly import assemble_base_global_stiffness, apply_boundary_conditions, update_global_stiffness #, assemble_global_load_vector
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


# class ExplicitFEA(ExplicitComponent):
#
#     def initialize(self):
#         # Mesh parameters.
#         self.options.declare('nodes', types=np.ndarray, desc='Nodes of the mesh')
#         self.options.declare('elements', types=np.ndarray,
#                              desc='Elements connectivity of the mesh (8 nodes per element)')
#         self.options.declare('el_size', types=(int, float), desc='Edge length of uniform hexahedral elements',
#                              default=1.0)
#         self.options.declare('el_centers', types=np.ndarray, desc='Centers of the mesh elements')
#         self.options.declare('dirichlet_nodes', types=np.ndarray, desc='Indices of nodes with Dirichlet BCs')
#         self.options.declare('dirichlet_values', types=np.ndarray, desc='Prescribed temperature at Dirichlet nodes')
#         self.options.declare('robin_nodes', types=np.ndarray, desc='Indices of nodes with Robin BCs')
#         self.options.declare('robin_h', types=float, desc='Robin convection coefficient')
#         self.options.declare('robin_T_inf', types=float, desc='Ambient temperature for Robin BCs')
#         # Penalty parameter.
#         self.options.declare('penalty_beta', types=float, desc='Penalty parameter for Dirichlet BC enforcement',
#                              default=1e3)
#
#     def setup(self):
#         # Inputs.
#         self.add_input("density", shape_by_conn=True, desc="Element density (ersatz material)")
#         self.add_input("heat_loads", shape_by_conn=True, desc="Heat load per element")
#         # Outputs.
#         n_nodes = self.options['nodes'].shape[0]
#         self.add_output("temperature", shape=(n_nodes,), desc="Computed temperature field")
#         self.add_output("max_temperature", val=0.0, desc="Maximum temperature in the domain")
#
#     def setup_partials(self):
#         # Declare that we are using matrix-free derivatives.
#         self.declare_partials(of="temperature", wrt="density", method="exact")
#         self.declare_partials(of="temperature", wrt="heat_loads", method="exact")
#         self.declare_partials(of="max_temperature", wrt="density", method="exact")
#         self.declare_partials(of="max_temperature", wrt="heat_loads", method="exact")
#
#     def compute(self, inputs, outputs):
#         # Unpack options.
#         nodes = self.options['nodes']
#         elements = self.options['elements']
#         el_size = self.options['el_size']
#         robin_nodes = self.options['robin_nodes']
#         robin_h = self.options['robin_h']
#         robin_T_inf = self.options['robin_T_inf']
#         dirichlet_nodes = self.options['dirichlet_nodes']
#         dirichlet_values = self.options['dirichlet_values']
#         penalty_beta = self.options['penalty_beta']
#         # For Robin BC, assume the "area" associated with a node is the element face area.
#         # For a uniform element, we use el_size^2.
#         robin_area = el_size ** 2
#
#         # Unpack inputs.
#         density = inputs["density"].flatten()  # one density per element
#         heat_loads = inputs["heat_loads"].flatten()  # one heat load per element
#
#         temp, max_temp = self._compute_primal(density, heat_loads, nodes, elements,
#                                               robin_nodes, robin_h, robin_T_inf, robin_area,
#                                               dirichlet_nodes, dirichlet_values,
#                                               penalty_beta, el_size)
#         outputs["temperature"] = temp
#         outputs["max_temperature"] = max_temp
#
#     @staticmethod
#     def _compute_primal(density, heat_loads, nodes, elements,
#                         r_nodes, r_h, r_T_inf, r_area,
#                         d_nodes, d_T, beta, el_size):
#         """
#         Assemble the global system for a uniform hexahedral mesh using two-point Gauss quadrature,
#         apply heat loads, enforce boundary conditions via a penalty formulation, and solve for the temperature.
#
#         Parameters:
#           density  : (n_elements,) ersatz density values.
#           heat_loads: (n_elements,) heat load per element.
#           nodes    : (n_nodes x dim) array of node coordinates.
#           elements : (n_elements x 8) connectivity array.
#           r_nodes, r_h, r_T_inf, r_area: Robin BC parameters.
#           d_nodes, d_T: Dirichlet BC nodes and prescribed temperature.
#           beta     : Penalty parameter.
#           el_size  : Edge length of each element.
#
#         Returns:
#           u      : computed temperature field (n_nodes,).
#           u_max  : maximum temperature (scalar), here using a simple np.max.
#         """
#         # Assemble global stiffness matrix and initial load vector.
#         K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k=1.0, el_size=el_size)
#         # Assemble the load vector from the heat loads.
#         f_load = assemble_global_load_vector(nodes, elements, heat_loads, density, el_size)
#         f = f + f_load
#
#         # Apply boundary conditions.
#         K_bc, f_bc = apply_boundary_conditions(K, f, r_nodes, r_h, r_T_inf, r_area,
#                                                       d_nodes, d_T, beta)
#         # Solve the sparse system.
#         u = spsolve(K_bc, f_bc)
#         u_max = np.max(u)
#         return u, u_max
#
#     # def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):
#     #     # For brevity, the implementation of JVP/VJP is omitted here.
#     #     # One could use complex step (or other methods) externally.
#     #     pass

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
        self.options.declare('base_k', types=float, desc='Base thermal conductivity', default=1.0)



    def setup(self):

        self.add_input("density", shape_by_conn=True, desc="Material density")
        self.add_input("heat_loads", shape_by_conn=True)

        n_el = self.options['nodes'].shape[0]
        self.add_output("temperature", shape=(n_el,), desc="Computed temperature field")
        self.add_output("max_temperature", val=0.0, desc="Computed maximum temperature")

        # Preassemble the base global stiffness matrix once (for density=1).
        nodes = jnp.array(self.options['nodes'])
        elements = jnp.array(self.options['elements'])
        base_k = self.options['base_k']
        # assemble_base_global_stiffness returns a sparse matrix in BCOO format and an auxiliary mapping.
        self._K_base, self._elem_indices = assemble_base_global_stiffness(nodes, elements, base_k)

    def setup_partials(self):
        # Declare that we are using matrix-free derivatives
        self.declare_partials(of="temperature", wrt="density", method="exact")
        self.declare_partials(of="temperature", wrt="heat_loads", method="exact")
        self.declare_partials(of="max_temperature", wrt="density", method="exact")
        self.declare_partials(of="max_temperature", wrt="heat_loads", method="exact")

    def compute(self, inputs, outputs):

        # Unpack the options
        nodes = jnp.array(self.options['nodes'])
        elements = jnp.array(self.options['elements'])
        el_size = jnp.array(self.options['el_size'])
        el_centers = jnp.array(self.options['el_centers'])
        robin_nodes = jnp.array(self.options['robin_nodes'])
        robin_h = jnp.array(self.options['robin_h'])
        robin_T_inf = jnp.array(self.options['robin_T_inf'])
        dirichlet_nodes = jnp.array(self.options['dirichlet_nodes'])
        dirichlet_values = jnp.array(self.options['dirichlet_values'])

        robin_area = jnp.array(el_size ** 2)

        # Unpack the inputs
        density = jnp.array(inputs["density"])
        heat_loads = jnp.array(inputs["heat_loads"])

        # Retrieve the preassembled base stiffness matrix and mapping.
        K_base = self._K_base
        elem_indices = self._elem_indices
        # penal = self.options['penal']

        temp, max_temp = self._compute_primal(density, heat_loads, nodes, elements,
                                    robin_nodes, robin_h, robin_T_inf, robin_area,
                                    dirichlet_nodes, dirichlet_values, K_base, elem_indices)

        outputs["temperature"] = temp
        outputs["max_temperature"] = max_temp

    @staticmethod
    def _compute_primal(density, heat_loads, nodes, elements,
                        r_nodes, r_h, r_T_inf, r_area,
                        d_nodes, d_T, K_base, elem_indices):
        """
        Compute the primal (temperature) solution and a measure of the maximum
        temperature using a sparse solver. This function assembles the global system,
        applies heat loads, and then enforces boundary conditions using a penalty
        formulation rather than explicit partitioning.

        Parameters:
          density      : material density field (array, to be flattened)
          heat_loads   : distributed heat loads (array, to be flattened)
          nodes        : node coordinates (array)
          elements     : element connectivity (array)
          r_nodes      : indices for nodes with Robin BC
          r_h, r_T_inf, r_area : Robin BC parameters
          d_nodes      : indices for nodes with Dirichlet BC (prescribed temperature)
          d_T          : prescribed temperatures at d_nodes

        Returns:
          u    : computed temperature (or displacement) field (dense vector)
          u_max: a scalar computed via a Kreisselmeier–Steinhauser (KS) function (here using a min-version)
        """
        # Update the global stiffness matrix using current densities.
        K_updated = update_global_stiffness(K_base, elem_indices, density)

        # Assemble the global load vector from heat loads.
        nodes_per_elem = 8
        # Distribute each element's heat load to its nodes (simple lumping).
        element_contrib = (heat_loads * density) / nodes_per_elem
        f = jnp.zeros(nodes.shape[0])
        f = f.at[elements.flatten()].add(jnp.repeat(element_contrib, nodes_per_elem))

        # Apply boundary conditions (both Robin and Dirichlet).
        K_updated, f = apply_boundary_conditions(K_updated, f,
                                                 r_nodes, r_h, r_T_inf, r_area,
                                                 d_nodes, d_T, beta=1e3)

        # Solve the global system using a sparse solver (e.g., conjugate gradient).
        u, _ = cg(K_updated, f, tol=1e-8, maxiter=500)

        # Compute a scalar measure of the maximum temperature (e.g., using a KS function).
        u_max = kreisselmeier_steinhauser_max(u)

        # # Flatten the inputs if needed.
        # density = density.flatten()
        # heat_loads = heat_loads.flatten()
        #
        # # Set the base thermal conductivity (or other base material property)
        # # base_k = 1.0
        #
        # # Assemble the global stiffness matrix and load vector.
        # # This routine should return K in a sparse format (e.g., BCOO) and f as a dense jnp.array.
        # # TODO Assembly Kf in setup to avoid re-meshing!
        # # K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k)
        #
        # # Apply the heat loads.
        # # For an 8-node hexahedral element, distribute each element's heat load evenly to its nodes.
        # nodes_per_elem = 8
        # element_contrib = (heat_loads * density) / nodes_per_elem
        # # Repeat each element's contribution for each of its nodes.
        # node_contrib = jnp.repeat(element_contrib, nodes_per_elem)
        # # Assume 'elements' is an array of shape (n_elements, nodes_per_elem).
        # f = f.at[elements.flatten()].add(node_contrib)
        #
        # # Instead of partitioning the system, apply boundary conditions as sparse additions.
        # K, f = apply_boundary_conditions(K, f, r_nodes, r_h, r_T_inf, r_area,
        #                                         d_nodes, d_T, beta=1e3)
        #
        # # Solve the full sparse system: K * u = f.
        # # We assume K is now in BCOO format. Define a function for the iterative solver.
        # def fea_solve(rhs):
        #     # Use an iterative solver (e.g., Conjugate Gradient) suitable for symmetric positive-definite systems.
        #     u_sol, _ = cg(K, rhs, tol=1e-8, maxiter=500)
        #     return u_sol
        #
        # u = fea_solve(f)
        #
        # # For post-processing, compute a scalar measure of the solution.
        # # For example, use a Kreisselmeier–Steinhauser function (here using a min-version as a placeholder).
        # u_max = kreisselmeier_steinhauser_max(u)

        return u, u_max

    # @staticmethod
    # def _compute_primal(density, heat_loads, nodes, elements,
    #                     r_nodes, r_h, r_T_inf, r_area,
    #                     d_nodes, d_T):
    #
    #     # Flatten the inputs
    #     density = density.flatten()
    #     heat_loads = heat_loads.flatten()
    #
    #     # Set the base thermal conductivity
    #     # TODO What?
    #     base_k = 1.0
    #
    #
    #     # Assemble the global stiffness matrix and load vector.
    #     K, f = assemble_global_stiffness_matrix(nodes, elements, density, base_k)
    #
    #     # Apply the heat loads to the system.
    #     nodes_per_elem = 8
    #     element_contrib = (heat_loads * density) / nodes_per_elem
    #     node_contrib = jnp.repeat(element_contrib, nodes_per_elem)
    #     f = f.at[elements.flatten()].add(node_contrib)
    #
    #     # Apply the boundary conditions and partition the system.
    #     K_ff, K_fp, K_pf, K_pp, f_f, f_p, u_p, idx_f, idx_p = apply_boundary_conditions(K, f,
    #                                                                                     r_nodes, r_h, r_T_inf, r_area,
    #                                                                                     d_nodes, d_T)
    #
    #     # Solve the partitioned system for the unknown displacements.
    #     # K_ff @ u_f + K_fp @ u_p = f_f
    #     # K_ff @ u_f = f_f - K_fp @ u_p
    #     # u_f = K_ff^-1 @ (f_f - K_fp @ u_p)
    #     # u_f = jnp.linalg.solve(K_ff, f_f - K_fp @ u_p)
    #
    #     # Convert K_ff to a sparse format for efficient solving
    #     # K_ff = BCOO.from_scipy_sparse(coo_matrix(K_ff))
    #     # K_ff = coo_fromdense(K_ff)
    #     K_ff = BCOO.fromdense(K_ff)
    #
    #     # Solve the partitioned system for the unknown displacements using Conjugate Gradient (CG)
    #     def fea_solve(rhs):
    #         u_f, _ = cg(K_ff, rhs, tol=1e-8, maxiter=500)
    #         return u_f
    #
    #     u_f = fea_solve(f_f - K_fp @ u_p)  # Solving K_ff @ u_f = (f_f - K_fp @ u_p)
    #
    #     # Reassemble the full solution.
    #     n_nodes = K.shape[0]
    #     u = jnp.zeros(n_nodes)
    #     u = u.at[idx_f].set(u_f)
    #     u = u.at[idx_p].set(u_p)
    #
    #     # Calculate the max temp
    #     # u_max = kreisselmeier_steinhauser_max(u)
    #     # TODO Reset
    #     u_max = kreisselmeier_steinhauser_min(u)
    #
    #     return u, u_max

    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # Unpack dynamic (differentiable) inputs.
        density = jnp.array(inputs["density"])
        heat_loads = jnp.array(inputs["heat_loads"])

        # Unpack static options.
        nodes = jnp.array(self.options['nodes'])
        elements = jnp.array(self.options['elements'])
        el_size = jnp.array(self.options['el_size'])
        robin_nodes = jnp.array(self.options['robin_nodes'])
        robin_h = jnp.array(self.options['robin_h'])
        robin_T_inf = jnp.array(self.options['robin_T_inf'])
        dirichlet_nodes = jnp.array(self.options['dirichlet_nodes'])
        dirichlet_values = jnp.array(self.options['dirichlet_values'])
        robin_area = jnp.array(el_size ** 2)

        # Freeze all static arguments via partial so that only density and heat_loads are inputs.
        frozen_compute_primal = partial(
            self._compute_primal,
            nodes=nodes,
            elements=elements,
            r_nodes=robin_nodes,
            r_h=robin_h,
            r_T_inf=robin_T_inf,
            r_area=robin_area,
            d_nodes=dirichlet_nodes,
            d_T=dirichlet_values
        )


        if mode == "fwd":
            # Build tangents only for differentiable inputs.
            tan_density = (jnp.array(d_inputs["density"])
                           if "density" in d_inputs and d_inputs["density"] is not None
                           else jnp.zeros_like(density))

            tan_heat_loads = (jnp.array(d_inputs["heat_loads"])
                              if "heat_loads" in d_inputs and d_inputs["heat_loads"] is not None
                              else jnp.zeros_like(heat_loads))

            primals = (density, heat_loads)
            tangents = (tan_density, tan_heat_loads)

            # Call jax.jvp on the frozen function.
            _, tangent_out = jvp(frozen_compute_primal, primals, tangents)

            # Assign the computed derivatives to the outputs.
            d_outputs["temperature"] += tangent_out[0]
            d_outputs["max_temperature"] += tangent_out[1]

        elif mode == "rev":
            # Get the VJP (pullback) for the frozen function.
            _, pullback = vjp(frozen_compute_primal, density, heat_loads)
            cotangent = (d_outputs["temperature"], d_outputs["max_temperature"])
            grads = pullback(cotangent)

            # Accumulate the gradients into d_inputs.
            d_inputs["density"] += grads[0]
            d_inputs["heat_loads"] += grads[1]


class BoundaryConditionAggregator(ExplicitComponent):
    pass