# Standard library imports
from functools import partial
import jax.numpy as jnp
from jax import jvp, vjp
from openmdao.api import ExplicitComponent, IndepVarComp

# Local imports
from SPI2py.models.physics.distributed.mesh import generate_mesh
from SPI2py.models.physics.distributed.assembly import assemble_base_global_system_penalty, apply_bc_penalty,  update_global_system_penalty
from SPI2py.models.physics.distributed.assembly import assemble_global_system_partition
from SPI2py.models.physics.distributed.solver import solve_system_partition, solve_system_penalty
from SPI2py.models.utilities.aggregation import kreisselmeier_steinhauser_max


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
        nodes, elements, centers, nx, ny, nz, lx, ly, lz = generate_mesh(x_min, x_max, y_min, y_max, z_min, z_max, element_size=element_size)
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
        self.options.declare('dirichlet_T', types=float)
        self.options.declare('robin_nodes', types=jnp.ndarray)
        self.options.declare('robin_h', types=float)
        self.options.declare('robin_T_inf', types=float)
        self.options.declare('base_k', types=float, desc='Base thermal conductivity', default=1.0)
        self.options.declare('fea_solution_scheme', types=str, desc='FEA solution scheme', default='partition')

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
        nodes    = jnp.array(self.options['nodes'])
        elements = jnp.array(self.options['elements'])
        k        = jnp.array(self.options['base_k'])
        r_nodes  = jnp.array(self.options['robin_nodes'])
        r_h      = self.options['robin_h']
        r_T_inf  = self.options['robin_T_inf']
        r_area   = self.options['el_size'] ** 2
        d_nodes  = jnp.array(self.options['dirichlet_nodes'])
        d_T_arr  = self.options['dirichlet_T'] * jnp.ones(len(d_nodes))

        # Unpack the inputs
        density    = jnp.array(inputs["density"])
        heat_loads = jnp.array(inputs["heat_loads"])

        idx = jnp.arange(nodes.shape[0])
        idx_p = d_nodes
        idx_f = jnp.setdiff1d(idx, idx_p)

        temp, max_temp = self._compute_primal(density, heat_loads,
                                              nodes, elements,
                                              k,
                                              r_nodes, r_h, r_T_inf, r_area,
                                              d_nodes, d_T_arr,
                                              idx_f, idx_p)

        outputs["temperature"] = temp
        outputs["max_temperature"] = max_temp

    @staticmethod
    def _compute_primal(densities, heat_loads,
                        nodes, elements,
                        k,
                        r_nodes, r_h, r_T_inf, r_area,
                        d_nodes, d_T,
                        idx_f, idx_p):

        heat_nodes = jnp.arange(elements.shape[0])

        # Flatten the inputs
        densities = densities.flatten()
        heat_loads = heat_loads.flatten()

        # Initialize global stiffness matrix and force vector
        K, f, u = assemble_global_system_partition(nodes, elements,
                                                   k=k,
                                                   pseudo_densities=densities,
                                                   r_nodes=r_nodes,
                                                   r_h=r_h,
                                                   r_T_inf=r_T_inf,
                                                   r_area=r_area,
                                                   heat_nodes=heat_nodes,
                                                   heat_loads=heat_loads,
                                                   d_nodes=d_nodes,
                                                   d_T=d_T)

        u = solve_system_partition(K, f, u, idx_f, idx_p)

        # Compute a scalar measure of the maximum temperature (e.g., using a KS function).
        u_max = kreisselmeier_steinhauser_max(u)

        return u, u_max


    def compute_jacvec_product(self, inputs, d_inputs, d_outputs, mode):

        # Unpack dynamic (differentiable) inputs.
        density = jnp.array(inputs["density"])
        heat_loads = jnp.array(inputs["heat_loads"])

        # Unpack static options.
        nodes = jnp.array(self.options['nodes'])
        elements = jnp.array(self.options['elements'])
        d_nodes = jnp.array(self.options['dirichlet_nodes'])
        d_T_arr = self.options['dirichlet_T'] * jnp.ones(len(d_nodes))

        K_base = self._K_base
        f_base = self._f_base
        elem_indices = self._elem_indices

        idx = jnp.arange(nodes.shape[0])
        idx_p = d_nodes
        idx_f = jnp.setdiff1d(idx, idx_p)

        u_p = d_T_arr



        # Freeze all static arguments via partial so that only density and heat_loads are inputs.
        frozen_compute_primal = partial(self._compute_primal,
                                        nodes=nodes, elements=elements,
                                        K_base=K_base, f_base=f_base, u_p=u_p,
                                        elem_indices=elem_indices, idx_f=idx_f, idx_p=idx_p)


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


