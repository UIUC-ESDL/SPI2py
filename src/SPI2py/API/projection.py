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

    def setup(self):

        # Get the options
        n_comp_projections = self.options['n_comp_projections']
        n_int_projections = self.options['n_int_projections']
        n_projections = n_comp_projections + n_int_projections

        # Create the projection aggregator
        self.add_subsystem('aggregator', ProjectionAggregator(n_projections=n_projections))

        # Projection counter
        i=0


        # TODO Can I automatically connect this???
        # Add the projection components
        for j in range(n_comp_projections):
            self.add_subsystem(f'projection_{i}', Projection())
            # self.connect(f'system.components.comp_{j}.transformed_sphere_positions', 'projection_' + str(i) + '.points')
            self.connect(f'projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
            self.connect(f'projection_{i}.true_volumes', f'aggregator.true_volumes_{i}')
            self.connect(f'projection_{i}.projected_volumes', f'aggregator.projected_volumes_{i}')
            i += 1

        # Add the interconnect projection components
        for j in range(n_int_projections):
            self.add_subsystem(f'projection_{i}', Projection())
            # self.connect(f'system.interconnects.int_{j}.transformed_sphere_positions', 'projection_int_' + str(i) + '.points')
            self.connect(f'projection_{i}.pseudo_densities', f'aggregator.pseudo_densities_{i}')
            self.connect(f'projection_{i}.true_volumes', f'aggregator.true_volumes_{i}')
            self.connect(f'projection_{i}.projected_volumes', f'aggregator.projected_volumes_{i}')
            i += 1


class Projection(ExplicitComponent):
    """
    Calculates the pseudo-density of a set of points in a 3D grid

    TODO Deal with objects outside of mesh!
    """

    def initialize(self):
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
        self.add_output('pseudo_densities', compute_shape=lambda shapes: (shapes['centers'][0], shapes['centers'][1], shapes['centers'][2]))
        self.add_output('true_volumes', copy_shape='sphere_radii')
        self.add_output('projected_volumes', compute_shape=lambda shapes: (shapes['centers'][0], shapes['centers'][1], shapes['centers'][2]))

    def setup_partials(self):
        self.declare_partials('pseudo_densities', 'sphere_positions')

    def compute(self, inputs, outputs):

        # Get the Mesh inputs
        element_length   = jnp.array(inputs['element_length'])
        sample_points    = jnp.array(inputs['sample_points'])
        sample_radii     = jnp.array(inputs['sample_radii'])
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii     = jnp.array(inputs['sphere_radii'])

        # Compute the pseudo-densities
        pseudo_densities = self._project(sphere_positions, sphere_radii, sample_points, sample_radii)

        # Compute the true volume
        true_volumes = self._true_volumes(sphere_positions, sphere_radii)

        # Compute the projected volume
        projected_volumes = self._projected_volumes(pseudo_densities, element_length)

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities
        outputs['true_volumes'] = true_volumes
        outputs['projected_volumes'] = projected_volumes

    def compute_partials(self, inputs, partials):

        # Get the inputs
        element_length   = jnp.array(inputs['element_length'])
        sample_points    = jnp.array(inputs['sample_points'])
        sample_radii     = jnp.array(inputs['sample_radii'])
        sphere_positions = jnp.array(inputs['sphere_positions'])
        sphere_radii     = jnp.array(inputs['sphere_radii'])

        # Calculate the Jacobian of the pseudo-densities
        jac_pseudo_densities = jacfwd(self._project)(sphere_positions, sphere_radii, sample_points, sample_radii)

        # Calculate the Jacobian of the projected volumes
        pseudo_densities = self._project(sphere_positions, sphere_radii,  sample_points, sample_radii)
        jac_projected_volumes = jacfwd(self._projected_volumes)(pseudo_densities, element_length)

        # Set the partials
        partials['pseudo_densities', 'sphere_positions'] = jac_pseudo_densities
        partials['projected_volumes', 'projected_volumes'] = jac_projected_volumes


    @staticmethod
    def _project(sphere_positions, sphere_radii, sample_points, sample_radii):
        pseudo_densities = calculate_pseudo_densities(sphere_positions, sphere_radii, sample_points, sample_radii)
        return pseudo_densities

    @staticmethod
    def _true_volumes(sphere_positions, sphere_radii):
        return (4 / 3) * jnp.pi * sphere_radii ** 3

    @staticmethod
    def _projected_volumes(pseudo_densities, element_length):
        return pseudo_densities * element_length ** 3


class ProjectionAggregator(ExplicitComponent):

    def initialize(self):
        self.options.declare('n_projections', types=int, desc='Number of projections')
        self.options.declare('rho_min', types=(int, float), desc='Minimum value of the density', default=3e-3)

    def setup(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the inputs
        self.add_input('element_length', val=0)

        for i in range(n_projections):
            self.add_input(f'pseudo_densities_{i}', shape_by_conn=True)
            self.add_input(f'true_volumes_{i}', shape_by_conn=True)
            self.add_input(f'projected_volumes_{i}', shape_by_conn=True)

        # Set the outputs
        self.add_output('pseudo_densities', copy_shape='pseudo_densities_0')
        self.add_output('true_volume', val=0.0)
        self.add_output('projected_volume', val=0.0)

    def setup_partials(self):

        # Get the options
        n_projections = self.options['n_projections']

        # Set the partials
        for i in range(n_projections):
            self.declare_partials('pseudo_densities', f'pseudo_densities_{i}')
            self.declare_partials('projected_volume', f'projected_volumes_{i}')


    def compute(self, inputs, outputs):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        pseudo_densities = [jnp.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections)]
        true_volumes = [inputs[f'true_volumes_{i}'] for i in range(n_projections)]
        projected_volumes = [inputs[f'projected_volumes_{i}'] for i in range(n_projections)]

        # Calculate the values
        pseudo_densities = self._aggregate_pseudo_densities(pseudo_densities, rho_min)
        true_volume = self._aggregate_true_volume(true_volumes)
        projected_volume = self._aggregate_projected_volume(projected_volumes)

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities
        outputs['true_volume'] = true_volume
        outputs['projected_volume'] = projected_volume

    def compute_partials(self, inputs, partials):

        # Get the options
        n_projections = self.options['n_projections']
        rho_min = self.options['rho_min']

        # Get the inputs
        pseudo_densities = [jnp.array(inputs[f'pseudo_densities_{i}']) for i in range(n_projections)]
        projected_volumes = [inputs[f'projected_volumes_{i}'] for i in range(n_projections)]

        # Calculate the partial derivatives
        jac_pseudo_densities = jacfwd(self._aggregate_pseudo_densities)(pseudo_densities, rho_min)
        jac_projected_volume = jacfwd(self._aggregate_projected_volume)(projected_volumes)

        # Set the partial derivatives
        jacs = zip(jac_pseudo_densities, jac_projected_volume)
        for i, jac_pseudo_densities_i, jac_projected_volume_i in enumerate(jacs):
            partials['pseudo_densities', f'pseudo_densities_{i}'] = jac_pseudo_densities_i
            partials['projected_volume', f'projected_volume_{i}'] = jac_projected_volume_i

    @staticmethod
    def _aggregate_pseudo_densities(pseudo_densities, rho_min):

        aggregate_pseudo_densities = jnp.zeros_like(pseudo_densities[0])

        for pseudo_density in pseudo_densities:
            aggregate_pseudo_densities += pseudo_density

        # Apply the minimum and maximum density constraints
        aggregate_pseudo_densities = jnp.maximum(aggregate_pseudo_densities, rho_min)
        aggregate_pseudo_densities = jnp.minimum(aggregate_pseudo_densities, 1.0)

        return aggregate_pseudo_densities

    @staticmethod
    def _aggregate_true_volume(true_volumes):
        true_volume = 0.0
        for volumes in true_volumes:
            true_volume += jnp.sum(volumes)
        return true_volume

    @staticmethod
    def _aggregate_projected_volume(projected_volumes):
        projected_volume = 0.0
        for volumes in projected_volumes:
            projected_volume += jnp.sum(volumes)
        return projected_volume
