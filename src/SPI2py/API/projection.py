import jax.numpy as np
from jax import jacfwd, jacrev
from jax.scipy.stats import gaussian_kde
from openmdao.core.explicitcomponent import ExplicitComponent


class Projection(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid
    """

    def initialize(self):
        self.options.declare('min_xyz', types=(int, float), desc='Minimum value of the x-, y-, and z-axis')
        self.options.declare('max_xyz', types=(int, float), desc='Maximum value of the x-, y-, and z-axis')
        self.options.declare('n_el_xyz', types=int, desc='Number of elements along the x-, y-, and z-axis')
        self.options.declare('rho_min', types=float, desc='Minimum value of the density', default=3e-3)

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
        element_center_positions = np.meshgrid(x_center_positions, y_center_positions, z_center_positions, indexing='ij')

        # Define inputs and output
        # self.add_input('points', shape_by_conn=True)
        self.add_input('points', shape=(10, 3))
        self.add_input('element_min_pseudo_densities', val=element_min_pseudo_densities)
        self.add_input('element_center_positions', val=element_center_positions)
        self.add_output('element_pseudo_densities', val=element_min_pseudo_densities)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # # Get the input arrays
        # input_vector = inputs['input_vector']
        #
        # # Convert the input to a JAX numpy array
        # input_vector = np.array(input_vector)
        #
        # # Stack inputs vertically
        # aggregated_output = self._compute_aggregation(input_vector)
        #
        # # Convert the output to a numpy array
        # aggregated_output = aggregated_output.detach().numpy()
        #
        # # Set the output
        # outputs['aggregated_output'] = aggregated_output

        outputs['element_pseudo_densities'] = 1

    # def compute_partials(self, inputs, partials):

        # # Get the input arrays
        # input_vector = inputs['input_vector']
        #
        # # Convert the input to a JAX numpy array
        # input_vector = np.array(input_vector)
        #
        # # Calculate the partial derivatives
        # jac_aggregated_output = jacrev(self._compute_aggregation)(input_vector)
        #
        # # Convert the partial derivatives to numpy arrays
        # jac_aggregated_output_np = jac_aggregated_output.detach().numpy()
        #
        # # Set the partial derivatives
        # partials['aggregated_output', 'input_vector'] = jac_aggregated_output_np

    # @staticmethod
    # def _project_points_to_mesh(points, element_center_positions, min_pseudo_densities):
    #     """
    #     Projects the points to the mesh and calculates the pseudo-densities
    #     """
    #
    #     # Would something this simple work?
    #     # # Calculate the distances from each point to each element center
    #     # distances = np.linalg.norm(points[:, None, :] - element_center_positions[None, :, :], axis=-1)
    #     # # Calculate the pseudo-densities
    #     # pseudo_densities = np.min(distances, axis=0)
    #     # # Calculate the minimum pseudo-densities
    #     # pseudo_densities = np.maximum(pseudo_densities, min_pseudo_densities)
    #
    #     # Perform KDE
    #     kernel = gaussian_kde(points.T, bw_method='scott')
    #     density_values = kernel(grid_positions).reshape(x_grid.shape)
    #
    #
    #     grad_kernel = jacfwd(lambda x: gaussian_kde(x, bw_method='scott'))
    #     grad_kernel_val = grad_kernel(points.T)
    #
    #
    #     return pseudo_densities