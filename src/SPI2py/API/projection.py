import jax.numpy as np
from jax import jacfwd, jacrev
from jax.scipy.stats import gaussian_kde
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from ..models.physics.continuum.geometric_projection import projection_volume
from ..models.kinematics.distance_calculations import signed_distances_spheres_spheres_np

class Mesh(IndepVarComp):
    def initialize(self):

        self.options.declare('element_length', types=float, desc='Length of a cubic element')
        self.options.declare('n_el_x', types=int, desc='Number of elements along the x axis')
        self.options.declare('n_el_y', types=int, desc='Number of elements along the y axis')
        self.options.declare('n_el_z', types=int, desc='Number of elements along the z axis')

    def setup(self):

        # Get the options
        element_length = self.options['element_length']
        n_el_x = self.options['n_el_x']
        n_el_y = self.options['n_el_y']
        n_el_z = self.options['n_el_z']

        # Define the properties of an element
        element_half_length = element_length / 2

        # Define the properties of the mesh
        x_center_positions = np.linspace(0 + element_half_length, element_length * n_el_x - element_half_length, n_el_x)
        y_center_positions = np.linspace(0 + element_half_length, element_length * n_el_y - element_half_length, n_el_y)
        z_center_positions = np.linspace(0 + element_half_length, element_length * n_el_z - element_half_length, n_el_z)
        mesh_element_center_positions = np.meshgrid(x_center_positions, y_center_positions, z_center_positions)

        mesh_shape = np.zeros((n_el_x, n_el_y, n_el_z))

        # Declare the outputs
        self.add_output('mesh_element_length', val=element_length)
        self.add_output('mesh_element_center_positions', val=mesh_element_center_positions)
        self.add_output('mesh_shape', val=mesh_shape)


class Projections(Group):
    def initialize(self):
        self.options.declare('n_comp_projections', types=int, desc='Number of component projections')
        self.options.declare('n_int_projections', types=int, desc='Number of interconnect projections')
        self.options.declare('rho_min', types=float, desc='Minimum value of the density', default=1e-3)

    def setup(self):

        # Get the options
        n_comp_projections = self.options['n_comp_projections']
        n_int_projections = self.options['n_int_projections']
        rho_min = self.options['rho_min']

        # Projection counter
        i=0

        # TODO Can I automatically connect this???
        # Add the projection components
        for j in range(n_comp_projections):
            self.add_subsystem('projection_' + str(i), Projection(rho_min=rho_min))
            # self.connect(f'system.components.comp_{j}.transformed_sphere_positions', 'projection_' + str(i) + '.points')
            i += 1

        # Add the interconnect projection components
        for j in range(n_int_projections):
            self.add_subsystem('projection_' + str(i), Projection(rho_min=rho_min))
            # self.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', 'projection_int_' + str(i) + '.points')
            i += 1


class Projection(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid

    TODO Deal with objects outside of mesh!
    """

    def initialize(self):
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)

    def setup(self):

        # Mesh Inputs
        self.add_input('mesh_element_length', val=0)
        self.add_input('mesh_shape', shape_by_conn=True)
        self.add_input('mesh_element_center_positions', shape_by_conn=True)

        # Object Inputs
        self.add_input('points', shape_by_conn=True)
        self.add_input('reference_density', 0.0, desc='Number of points per a 1x1x1 cube')

        # Outputs
        # TODO Fix
        self.add_output('mesh_element_pseudo_densities', copy_shape='mesh_shape')
        self.add_output('volume', val=0.0)

    def setup_partials(self):
        self.declare_partials('mesh_element_pseudo_densities', 'points')

    def compute(self, inputs, outputs):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        mesh_element_length = inputs['mesh_element_length']
        mesh_element_center_positions = inputs['mesh_element_center_positions']

        # Get the Object inputs
        points = inputs['points']
        reference_density = inputs['reference_density']

        # Convert the input to a JAX numpy array
        points = np.array(points)
        mesh_element_center_positions = np.array(mesh_element_center_positions)

        # Project
        element_pseudo_densities = self._project(points, mesh_element_length, mesh_element_center_positions, rho_min, reference_density)

        # Calculate the volume
        volume = np.sum(element_pseudo_densities) * mesh_element_length ** 3

        # Write the outputs
        outputs['mesh_element_pseudo_densities'] = element_pseudo_densities
        outputs['volume'] = volume

    # def compute_partials(self, inputs, partials):
    #
    #     # Get the options
    #     rho_min = self.options['rho_min']
    #
    #     # Get the inputs
    #     points = inputs['points']
    #     element_length = inputs['mesh_element_length']
    #     element_min_pseudo_densities = inputs['element_min_pseudo_densities']
    #     element_center_positions = inputs['element_center_positions']
    #
    #     # Convert the input to a JAX numpy array
    #     points = np.array(points)
    #     element_min_pseudo_densities = np.array(element_min_pseudo_densities)
    #     element_center_positions = np.array(element_center_positions)
    #
    #     # Calculate the Jacobian of the kernel
    #     grad_kernel = jacfwd(self._project, argnums=0)
    #     grad_kernel_val = grad_kernel(points, element_center_positions, element_min_pseudo_densities)
    #
    #     partials['mesh_element_pseudo_densities', 'points'] = grad_kernel_val # TODO Check all the transposing... (?)

    @staticmethod
    def _project(points, mesh_element_length, mesh_element_center_positions, rho_min, reference_density):
        """
        Projects the points to the mesh and calculates the pseudo-densities
        """

        grid_x, grid_y, grid_z = mesh_element_center_positions
        grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

        element_volume = mesh_element_length ** 3

        # # Scale the relative density to the mesh
        # relative_density_volume = 1
        # scale_factor = element_volume / relative_density_volume
        # points_per_mesh_element = scale_factor * reference_density

        # Perform KDE
        kernel = gaussian_kde(points.T, bw_method='scott')
        density_values = kernel(grid_coords).reshape(grid_x.shape)
        points_per_cell = density_values * element_volume * len(points)

        pseudo_densities = density_values + rho_min
        # Normalize the pseudo-densities
        min_density = np.min(pseudo_densities)
        max_density = np.max(pseudo_densities)
        pseudo_densities = (pseudo_densities - min_density) / (max_density - min_density)

        # Clip the minimum pseudo-densities to rho_min (to avoid numerical issues associated w/ zero densities)
        pseudo_densities = np.clip(pseudo_densities, rho_min, 1)

        return pseudo_densities