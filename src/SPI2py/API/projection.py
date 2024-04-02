import jax.numpy as jnp
from jax import jacfwd, jacrev
from openmdao.api import ExplicitComponent, Group
from openmdao.core.indepvarcomp import IndepVarComp

from ..models.projection.projection import calculate_pseudo_densities
from ..models.projection.mesh_kernels import mdbd_kernel_positions, mdbd_kernel_radii


class Mesh(IndepVarComp):
    def initialize(self):

        self.options.declare('bounds', types=tuple, desc='Bounds of the mesh')
        self.options.declare('n_elements_per_unit_length', types=float, desc='Number of elements per unit length')
        self.options.declare('mesh_kernel_min_radius', types=float)

    def setup(self):

        # Get the options
        bounds = self.options['bounds']
        n_elements_per_unit_length = self.options['n_elements_per_unit_length']
        mesh_kernel_min_radius = self.options['mesh_kernel_min_radius']

        # Determine the number of elements in each direction
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        n_el_x = int(n_elements_per_unit_length * (x_max - x_min))
        n_el_y = int(n_elements_per_unit_length * (y_max - y_min))
        n_el_z = int(n_elements_per_unit_length * (z_max - z_min))

        element_length = 1 / n_elements_per_unit_length
        element_half_length = element_length / 2

        # Define the mesh grid positions
        x_grid_positions = jnp.linspace(x_min, x_max, n_el_x + 1)
        y_grid_positions = jnp.linspace(y_min, y_max, n_el_y + 1)
        z_grid_positions = jnp.linspace(z_min, z_max, n_el_z + 1)
        x_grid, y_grid, z_grid = jnp.meshgrid(x_grid_positions, y_grid_positions, z_grid_positions, indexing='ij')
        grid = jnp.stack((x_grid, y_grid, z_grid), axis=-1)

        # Define the mesh center points
        x_center_positions = jnp.linspace(x_min + element_half_length, element_length * n_el_x - element_half_length, n_el_x)
        y_center_positions = jnp.linspace(y_min + element_half_length, element_length * n_el_y - element_half_length, n_el_y)
        z_center_positions = jnp.linspace(z_min + element_half_length, element_length * n_el_z - element_half_length, n_el_z)
        x_centers, y_centers, z_centers = jnp.meshgrid(x_center_positions, y_center_positions, z_center_positions, indexing='ij')


        # Read the MDBD kernel
        kernel_positions = jnp.array(mdbd_kernel_positions)
        kernel_radii = jnp.array(mdbd_kernel_radii).reshape(-1, 1)

        # Truncate the number of spheres based on the minimum radius
        kernel_positions = kernel_positions[kernel_radii.flatten() > mesh_kernel_min_radius]
        kernel_radii = kernel_radii[kernel_radii.flatten() > mesh_kernel_min_radius]

        # Scale the sphere positions
        kernel_positions = kernel_positions * element_length
        kernel_radii = kernel_radii * element_length

        meshgrid_centers = jnp.stack((x_centers, y_centers, z_centers), axis=-1)
        meshgrid_centers_expanded = jnp.expand_dims(meshgrid_centers, axis=3)


        all_points = meshgrid_centers_expanded + kernel_positions
        all_radii = jnp.zeros((n_el_x, n_el_y, n_el_z, 1, 1)) + kernel_radii


        # Declare the outputs
        self.add_output('element_length', val=element_length)
        self.add_output('centers', val=meshgrid_centers)
        self.add_output('grid', val=grid)
        self.add_output('n_el_x', val=n_el_x)
        self.add_output('n_el_y', val=n_el_y)
        self.add_output('n_el_z', val=n_el_z)
        self.add_output('sample_points', val=all_points)
        self.add_output('sample_radii', val=all_radii)


class Projections(Group):
    def initialize(self):
        self.options.declare('n_comp_projections', types=int, desc='Number of component projections')
        self.options.declare('n_int_projections', types=int, desc='Number of interconnect projections')
        self.options.declare('rho_min', types=float, desc='Minimum value of the density', default=1e-5)

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
        self.options.declare('color', types=str, desc='Color of the projection', default='blue')

    def setup(self):

        # Mesh Inputs
        self.add_input('element_length', val=0)
        self.add_input('centers', shape_by_conn=True)
        self.add_input('sample_points', shape_by_conn=True)
        self.add_input('sample_radii', shape_by_conn=True)

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)

        # Outputs
        self.add_output('pseudo_densities', compute_shape=lambda shapes: (shapes['centers'][0],shapes['centers'][1],shapes['centers'][2]))
        self.add_output('true_volume', val=0.0)
        self.add_output('projected_volume', val=0.0)

    def setup_partials(self):
        self.declare_partials('pseudo_densities', 'sphere_positions')

    def compute(self, inputs, outputs):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        element_length = inputs['element_length']
        centers = inputs['centers']
        sample_points = inputs['sample_points']
        sample_radii = inputs['sample_radii']

        # Get the Object inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']

        # Convert the input to Jax arrays
        element_length = jnp.array(element_length)
        centers = jnp.array(centers)
        sample_points = jnp.array(sample_points)
        sample_radii = jnp.array(sample_radii)


        sphere_positions = jnp.array(sphere_positions)
        sphere_radii = jnp.array(sphere_radii)

        # Project
        pseudo_densities= self._project(sphere_positions, sphere_radii,
                                                 element_length,
                                                 centers, sample_points, sample_radii, rho_min)

        # Calculate the volume
        true_volume = self._true_volume(sphere_positions, sphere_radii)

        # Calculate the projected volume
        projected_volume = self._projected_volume(pseudo_densities, element_length)

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities
        outputs['true_volume'] = true_volume
        outputs['projected_volume'] = projected_volume

    def compute_partials(self, inputs, partials):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        element_length = inputs['element_length']
        centers = inputs['centers']
        sample_points = inputs['sample_points']
        sample_radii = inputs['sample_radii']

        # Get the Object inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']

        # Convert the input to Jax arrays
        element_length = jnp.array(element_length)
        centers = jnp.array(centers)
        sample_points = jnp.array(sample_points)
        sample_radii = jnp.array(sample_radii)

        sphere_positions = jnp.array(sphere_positions)
        sphere_radii = jnp.array(sphere_radii)

        # Calculate the Jacobian of the kernel
        grad_sphere_positions = jacfwd(self._project, argnums=0)
        grad_sphere_positions_val = grad_sphere_positions(sphere_positions, sphere_radii, element_length,
                                                 centers,sample_points, sample_radii, rho_min)

        # Set the partials
        partials['pseudo_densities', 'sphere_positions'] = grad_sphere_positions_val


    @staticmethod
    def _project(sphere_positions, sphere_radii, element_length, centers, sample_points, sample_radii, rho_min):
        pseudo_densities = calculate_pseudo_densities(sphere_positions, sphere_radii, element_length, centers, sample_points, sample_radii, rho_min)
        return pseudo_densities

    @staticmethod
    def _true_volume(sphere_positions, sphere_radii):
        return jnp.sum(4 / 3 * jnp.pi * sphere_radii ** 3)

    @staticmethod
    def _projected_volume(pseudo_densities, element_length):
        return jnp.sum(pseudo_densities) * element_length ** 3

