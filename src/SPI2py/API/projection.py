import jax.numpy as np
from jax import jacfwd, jacrev
from jax.scipy.stats import gaussian_kde
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from ..models.physics.continuum.geometric_projection import projection_volume
from ..models.kinematics.distance_calculations import signed_distances_spheres_spheres_np

class Mesh(IndepVarComp):
    def initialize(self):

        self.options.declare('bounds', types=tuple, desc='Bounds of the mesh')
        self.options.declare('n_elements_per_unit_length', types=float, desc='Number of elements per unit length')

    def setup(self):

        # Get the options
        # element_length = self.options['element_length']
        # n_el_x = self.options['n_el_x']
        # n_el_y = self.options['n_el_y']
        # n_el_z = self.options['n_el_z']
        bounds = self.options['bounds']
        n_elements_per_unit_length = self.options['n_elements_per_unit_length']

        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min

        n_el_x = int(n_elements_per_unit_length * x_len)
        n_el_y = int(n_elements_per_unit_length * y_len)
        n_el_z = int(n_elements_per_unit_length * z_len)

        element_length = 1 / n_elements_per_unit_length

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
        self.add_output('n_el_x', val=n_el_x)
        self.add_output('n_el_y', val=n_el_y)
        self.add_output('n_el_z', val=n_el_z)


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
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)

        # Outputs
        self.add_output('mesh_element_pseudo_densities', copy_shape='mesh_shape')
        self.add_output('volume', val=0.0)

    def setup_partials(self):
        self.declare_partials('mesh_element_pseudo_densities', 'sphere_positions')

    def compute(self, inputs, outputs):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        mesh_element_length = inputs['mesh_element_length']
        mesh_element_center_positions = inputs['mesh_element_center_positions']

        # Get the Object inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']

        # Convert the input to a JAX numpy array
        mesh_element_center_positions = np.array(mesh_element_center_positions)
        sphere_positions = np.array(sphere_positions)
        sphere_radii = np.array(sphere_radii)

        # Project
        element_pseudo_densities = self._project(sphere_positions, sphere_radii, mesh_element_length, mesh_element_center_positions, rho_min)

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
    def _project(sphere_positions, sphere_radii, mesh_element_length, mesh_element_center_positions, rho_min):
        """
        Projects the points to the mesh and calculates the pseudo-densities
        """

        _, nx, ny, nz = mesh_element_center_positions.shape

        # Reshape the mesh_centers to (nx*ny*nz, 3)
        mesh_centers = mesh_element_center_positions.transpose(1, 2, 3, 0).reshape(-1, 3)

        # mesh_radii = np.ones((mesh_centers.shape[0], 1)) * mesh_element_length / 2
        mesh_radii = np.zeros((mesh_centers.shape[0], 1))

        # Initialize influence map
        influence_map = np.zeros(mesh_radii.shape)

        # Spread of the influence, potentially adjust based on application
        sigma = 1.0


        for i in range(len(sphere_positions)):

            # Calculate adjusted distances from sphere center to each cell center
            # distances = np.linalg.norm(mesh_centers - sphere['center'], axis=-1) - sphere['radius']
            distances = -signed_distances_spheres_spheres_np(sphere_positions[i, np.newaxis], sphere_radii[i, np.newaxis], mesh_centers, mesh_radii)

            # Ensure non-negative distances for the influence calculation
            distances = np.maximum(distances, 0)

            # Apply Gaussian kernel, adjusting influence based on adjusted distance
            influence = np.exp(-distances ** 2 / (2 * sigma ** 2))

            # Sum the influences of this sphere onto the mesh cells
            influence_map += influence.T

        pseudo_densities = influence_map

        # Reshape back to (nx, ny, nz, 1)
        pseudo_densities = pseudo_densities.reshape(nx, ny, nz, 1)

        # Transpose to get (3, nx, ny, nz)
        pseudo_densities = pseudo_densities.transpose(3, 0, 1, 2)

        # Normalize the pseudo-densities
        # pseudo_densities = (1 - rho_min) / (1 + np.exp(-1 * (pseudo_densities - 0.5))) + rho_min

        max_pd = np.max(pseudo_densities)
        min_pd = np.min(pseudo_densities)

        pseudo_densities = (pseudo_densities - min_pd) / (max_pd - min_pd)

        max_pd = np.max(pseudo_densities)
        min_pd = np.min(pseudo_densities)

        return pseudo_densities