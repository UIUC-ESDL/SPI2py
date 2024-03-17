import jax.numpy as np
from jax import jacfwd, jacrev
from jax.scipy.stats import gaussian_kde
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from ..models.physics.continuum.geometric_projection import projection_volume
from ..models.kinematics.distance_calculations import signed_distances_spheres_spheres_np

class Mesh(IndepVarComp):
    def initialize(self):
        self.options.declare('min_xyz', types=(int, float), desc='Minimum value of the x-, y-, and z-axis')
        self.options.declare('max_xyz', types=(int, float), desc='Maximum value of the x-, y-, and z-axis')
        self.options.declare('n_el_xyz', types=int, desc='Number of elements along the x-, y-, and z-axis')

    def setup(self):
        min_xyz = self.options['min_xyz']
        max_xyz = self.options['max_xyz']
        n_el_xyz = self.options['n_el_xyz']

        # Define the grid
        mesh_element_length = (max_xyz - min_xyz) / n_el_xyz
        mesh_element_half_length = mesh_element_length / 2
        x_center_positions = np.linspace(min_xyz + mesh_element_half_length, max_xyz - mesh_element_half_length, n_el_xyz)
        y_center_positions = np.linspace(min_xyz + mesh_element_half_length, max_xyz - mesh_element_half_length, n_el_xyz)
        z_center_positions = np.linspace(min_xyz + mesh_element_half_length, max_xyz - mesh_element_half_length, n_el_xyz)
        mesh_element_center_positions = np.meshgrid(x_center_positions, y_center_positions, z_center_positions)

        self.add_output('min_xyz', val=min_xyz)
        self.add_output('max_xyz', val=max_xyz)
        self.add_output('n_el_xyz', val=n_el_xyz)
        self.add_output('mesh_element_length', val=mesh_element_length)
        self.add_output('mesh_element_center_positions', val=mesh_element_center_positions)



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
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)

    def setup(self):

        # Mesh Inputs
        self.add_input('min_xyz', val=0)
        self.add_input('max_xyz', val=0)
        self.add_input('n_el_xyz', val=0)
        self.add_input('mesh_element_length', val=0)
        self.add_input('mesh_element_center_positions', shape_by_conn=True)

        # Object Inputs
        self.add_input('points', shape_by_conn=True)

        # Outputs
        self.add_output('mesh_element_pseudo_densities', shape_by_conn=True)
        self.add_output('volume', val=0.0)

    def setup_partials(self):
        self.declare_partials('element_pseudo_densities', 'points')

    def compute(self, inputs, outputs):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the inputs
        element_length = inputs['mesh_element_length']
        points = inputs['points']
        element_min_pseudo_densities = inputs['element_min_pseudo_densities']
        element_center_positions = inputs['element_center_positions']

        # Convert the input to a JAX numpy array
        points = np.array(points)
        element_center_positions = np.array(element_center_positions)

        # Project
        element_pseudo_densities = self._project(points, element_center_positions, rho_min)

        # Calculate the volume
        volume = np.sum(element_pseudo_densities) * element_length ** 3

        # Write the outputs
        outputs['element_pseudo_densities'] = element_pseudo_densities
        outputs['volume'] = volume

    # def compute_partials(self, inputs, partials):
    #
    #     # Get the options
    #     min_xyz = self.options['min_xyz']
    #     max_xyz = self.options['max_xyz']
    #     n_el_xyz = self.options['n_el_xyz']
    #     element_length = (max_xyz - min_xyz) / n_el_xyz
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
    def _project(points, mesh_element_center_positions, rho_min):
        """
        Projects the points to the mesh and calculates the pseudo-densities
        """

        grid_x, grid_y, grid_z = mesh_element_center_positions
        grid_coords = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()])

        # Perform KDE
        kernel = gaussian_kde(points.T, bw_method='scott')
        density_values = kernel(grid_coords).reshape(grid_x.shape)
        pseudo_densities = density_values + rho_min

        # Normalize the pseudo-densities
        min_density = np.min(pseudo_densities)
        max_density = np.max(pseudo_densities)
        pseudo_densities = (pseudo_densities - min_density) / (max_density - min_density)

        # Clip the minimum pseudo-densities to rho_min (to avoid numerical issues associated w/ zero densities)
        pseudo_densities = np.clip(pseudo_densities, rho_min, 1)

        return pseudo_densities