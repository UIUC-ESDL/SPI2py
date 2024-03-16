import jax.numpy as np
from jax import jacfwd, jacrev
from jax.scipy.stats import gaussian_kde
from openmdao.api import ExplicitComponent, Group
from ..models.physics.continuum.geometric_projection import projection_volume
from ..models.kinematics.distance_calculations import signed_distances_spheres_spheres_np


class Projections(Group):
    def initialize(self):
        self.options.declare('n_comp_projections', types=int, desc='Number of component projections')
        self.options.declare('n_int_projections', types=int, desc='Number of interconnect projections')
        self.options.declare('min_xyz', types=(int, float), desc='Minimum value of the x-, y-, and z-axis')
        self.options.declare('max_xyz', types=(int, float), desc='Maximum value of the x-, y-, and z-axis')
        self.options.declare('n_el_xyz', types=int, desc='Number of elements along the x-, y-, and z-axis')
        self.options.declare('rho_min', types=float, desc='Minimum value of the density', default=1e-3)

    def setup(self):

        # Get the options
        n_comp_projections = self.options['n_comp_projections']
        n_int_projections = self.options['n_int_projections']
        min_xyz = self.options['min_xyz']
        max_xyz = self.options['max_xyz']
        n_el_xyz = self.options['n_el_xyz']
        rho_min = self.options['rho_min']

        # Projection counter
        i=0

        # TODO Can I automatically connect this???
        # Add the projection components
        for j in range(n_comp_projections):
            self.add_subsystem('projection_' + str(i), Projection(min_xyz=min_xyz, max_xyz=max_xyz, n_el_xyz=n_el_xyz, rho_min=rho_min))
            # self.connect(f'system.components.comp_{j}.transformed_sphere_positions', 'projection_' + str(i) + '.points')
            i += 1

        # Add the interconnect projection components
        for j in range(n_int_projections):
            self.add_subsystem('projection_' + str(i), Projection(min_xyz=min_xyz, max_xyz=max_xyz, n_el_xyz=n_el_xyz, rho_min=rho_min))
            # self.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', 'projection_int_' + str(i) + '.points')
            i += 1


class Projection(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid

    TODO Deal with objects outside of mesh!
    """

    def initialize(self):
        self.options.declare('min_xyz', types=(int, float), desc='Minimum value of the x-, y-, and z-axis')
        self.options.declare('max_xyz', types=(int, float), desc='Maximum value of the x-, y-, and z-axis')
        self.options.declare('n_el_xyz', types=int, desc='Number of elements along the x-, y-, and z-axis')
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)

    def setup(self):

        # Get the options
        min_xyz = self.options['min_xyz']
        max_xyz = self.options['max_xyz']
        n_el_xyz = self.options['n_el_xyz']
        rho_min = self.options['rho_min']

        # Initialize the mesh
        element_min_pseudo_densities = rho_min * np.ones((n_el_xyz, n_el_xyz, n_el_xyz))

        # Calculate the center point of each element
        element_length = (max_xyz - min_xyz) / n_el_xyz
        element_half_length = element_length / 2
        x_center_positions = np.linspace(min_xyz + element_half_length, max_xyz - element_half_length, n_el_xyz)
        y_center_positions = np.linspace(min_xyz + element_half_length, max_xyz - element_half_length, n_el_xyz)
        z_center_positions = np.linspace(min_xyz + element_half_length, max_xyz - element_half_length, n_el_xyz)
        element_center_positions = np.meshgrid(x_center_positions, y_center_positions, z_center_positions)

        # sampling_spheres_positions = np.array([element_center_positions[0].ravel(), element_center_positions[1].ravel(), element_center_positions[2].ravel()]).T
        # sampling_sphere_radii = element_half_length * np.ones(len(sampling_spheres_positions))
        # sampling_sphere_min_pseudo_densities = rho_min * np.ones(len(sampling_spheres_positions))

        # Define inputs and output
        # TODO Combine center positions and densities into a single input
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)
        self.add_input('element_min_pseudo_densities', val=element_min_pseudo_densities)
        self.add_input('element_center_positions', val=element_center_positions)
        # self.add_input('sampling_spheres_positions', val=sampling_spheres_positions)
        # self.add_input('sampling_sphere_radii', val=sampling_sphere_radii)
        # self.add_input('sampling_sphere_min_pseudo_densities', val=sampling_sphere_min_pseudo_densities)
        self.add_output('element_pseudo_densities', val=element_min_pseudo_densities)

    def setup_partials(self):
        self.declare_partials('element_pseudo_densities', 'sphere_positions')

    def compute(self, inputs, outputs):

        # Get the options
        min_xyz = self.options['min_xyz']
        max_xyz = self.options['max_xyz']
        n_el_xyz = self.options['n_el_xyz']
        element_length = (max_xyz - min_xyz) / n_el_xyz

        # Get the inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']
        element_min_pseudo_densities = inputs['element_min_pseudo_densities']
        element_center_positions = inputs['element_center_positions']


        # Convert the input to a JAX numpy array
        sphere_positions = np.array(sphere_positions)
        sphere_radii = np.array(sphere_radii)
        element_min_pseudo_densities = np.array(element_min_pseudo_densities)
        element_center_positions = np.array(element_center_positions)

        # Project
        sampling_sphere_densities = self._project(sphere_positions, sphere_radii,
                                                  element_center_positions, element_min_pseudo_densities)

        outputs['sampling_sphere_densities'] = sampling_sphere_densities

    # def compute_partials(self, inputs, partials):
    #
    #     # Get the inputs
    #     points = inputs['points']
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
    #     partials['element_pseudo_densities', 'points'] = grad_kernel_val # TODO Check all the transposing... (?)

    @staticmethod
    def calculate_density(min1, max1, min2, max2):
        """
        Calculate the overlap volume between two cubes.
        """
        overlap_x = max(0, min(max1[0], max2[0]) - max(min1[0], min2[0]))
        overlap_y = max(0, min(max1[1], max2[1]) - max(min1[1], min2[1]))
        overlap_z = max(0, min(max1[2], max2[2]) - max(min1[2], min2[2]))

        cell_volume = (max1[0] - min1[0]) * (max1[1] - min1[1]) * (max1[2] - min1[2])

        overlap_volume = overlap_x * overlap_y * overlap_z

        density = overlap_volume / cell_volume

        return

    @staticmethod
    def _project(sphere_positions, sphere_radii, element_center_positions, element_min_pseudo_densities):
        """
        Projects the points to the mesh and calculates the pseudo-densities
        """

        # Calculate how much the bounding box of each sphere overlaps with each element
        sphere_bounding_boxes = np.array([sphere_positions - sphere_radii, sphere_positions + sphere_radii]).T




        # Perform KDE
        # kernel = gaussian_kde(points.T, bw_method='scott')
        # density_values = kernel(grid_coords).reshape(grid_x.shape)
        # pseudo_densities = density_values + min_pseudo_densities
        # Normalize the pseudo-densities
        # min_density = np.min(pseudo_densities)
        # max_density = np.max(pseudo_densities)
        # pseudo_densities = (pseudo_densities - min_density) / (max_density - min_density)

        return pseudo_densities