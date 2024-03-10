import jax.numpy as np
from jax import jacfwd, jacrev
from jax.scipy.stats import gaussian_kde
from openmdao.core.explicitcomponent import ExplicitComponent


class Projection(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid
    """

    def initialize(self):
        self.options.declare('x_min', types=float, desc='Minimum value of the x-axis')
        self.options.declare('x_max', types=float, desc='Maximum value of the x-axis')
        self.options.declare('y_min', types=float, desc='Minimum value of the y-axis')
        self.options.declare('y_max', types=float, desc='Maximum value of the y-axis')
        self.options.declare('z_min', types=float, desc='Minimum value of the z-axis')
        self.options.declare('z_max', types=float, desc='Maximum value of the z-axis')
        self.options.declare('n_el_x', types=int)
        self.options.declare('n_el_y', types=int)
        self.options.declare('n_el_z', types=int)
        self.options.declare('rho_min', types=float, description='Minimum value of the density', default=3e-3)

    def setup(self):

        # Get the options
        x_min = self.options['x_min']
        x_max = self.options['x_max']
        y_min = self.options['y_min']
        y_max = self.options['y_max']
        z_min = self.options['z_min']
        z_max = self.options['z_max']
        n_el_x = self.options['n_el_x']
        n_el_y = self.options['n_el_y']
        n_el_z = self.options['n_el_z']
        rho_min = self.options['rho_min']

        # Initialize the mesh
        element_min_pseudo_densities = rho_min * np.ones((n_el_x, n_el_y, n_el_z))

        # Calculate the center point of each element
        dx = (x_max - x_min) / n_el_x
        dy = (y_max - y_min) / n_el_y
        dz = (z_max - z_min) / n_el_z
        x = np.linspace(x_min + dx/2, x_max - dx/2, n_el_x)
        y = np.linspace(y_min + dy/2, y_max - dy/2, n_el_y)
        z = np.linspace(z_min + dz/2, z_max - dz/2, n_el_z)
        element_center_positions = np.meshgrid(x, y, z, indexing='ij')  # TODO Verify correct indexing

        # Define inputs and output
        self.add_input('points', shape_by_conn=True)
        self.add_input('min_pseudo_densities', val=element_min_pseudo_densities)
        self.add_input('element_center_positions', val=element_center_positions)
        self.add_output('pseudo_densities', val=element_min_pseudo_densities)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):

        # Get the input arrays
        input_vector = inputs['input_vector']

        # Convert the input to a JAX numpy array
        input_vector = np.array(input_vector)

        # Stack inputs vertically
        aggregated_output = self._compute_aggregation(input_vector)

        # Convert the output to a numpy array
        aggregated_output = aggregated_output.detach().numpy()

        # Set the output
        outputs['aggregated_output'] = aggregated_output

    def compute_partials(self, inputs, partials):
        # Get the input arrays
        input_vector = inputs['input_vector']

        # Convert the input to a JAX numpy array
        input_vector = np.array(input_vector)

        # Calculate the partial derivatives
        jac_aggregated_output = jacrev(self._compute_aggregation)(input_vector)

        # Convert the partial derivatives to numpy arrays
        jac_aggregated_output_np = jac_aggregated_output.detach().numpy()

        # Set the partial derivatives
        partials['aggregated_output', 'input_vector'] = jac_aggregated_output_np

    @staticmethod
    def _project_points_to_mesh(points, element_center_positions, min_pseudo_densities):
        """
        Projects the points to the mesh and calculates the pseudo-densities
        """

        # Would something this simple work?
        # # Calculate the distances from each point to each element center
        # distances = np.linalg.norm(points[:, None, :] - element_center_positions[None, :, :], axis=-1)
        # # Calculate the pseudo-densities
        # pseudo_densities = np.min(distances, axis=0)
        # # Calculate the minimum pseudo-densities
        # pseudo_densities = np.maximum(pseudo_densities, min_pseudo_densities)

        # Perform KDE
        kernel = gaussian_kde(points.T, bw_method='scott')
        density_values = kernel(grid_positions).reshape(x_grid.shape)


        grad_kernel = jacfwd(lambda x: gaussian_kde(x, bw_method='scott'))
        grad_kernel_val = grad_kernel(points.T)


        return pseudo_densities