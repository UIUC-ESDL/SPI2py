import torch
import numpy as np
import os
from torch.func import jacfwd
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from ..models.kinematics.distance_calculations import distances_points_points
from ..models.kinematics.encapsulation import overlap_volume_spheres_spheres, overlap_volume_sphere_sphere
from ..models.geometry.finite_sphere_method import read_xyzr_file

class Mesh(IndepVarComp):
    def initialize(self):

        self.options.declare('bounds', types=tuple, desc='Bounds of the mesh')
        self.options.declare('n_elements_per_unit_length', types=float, desc='Number of elements per unit length')
        self.options.declare('mdbd_unit_cube_filepath', types=str)
        self.options.declare('mdbd_unit_cube_min_radius', types=float)

    def setup(self):

        # Get the options
        bounds = self.options['bounds']
        n_elements_per_unit_length = self.options['n_elements_per_unit_length']
        mdbd_unit_cube_filepath = self.options['mdbd_unit_cube_filepath']
        mdbd_unit_cube_min_radius = self.options['mdbd_unit_cube_min_radius']

        # Determine the number of elements in each direction
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        x_len = x_max - x_min
        y_len = y_max - y_min
        z_len = z_max - z_min

        n_el_x = int(n_elements_per_unit_length * x_len)
        n_el_y = int(n_elements_per_unit_length * y_len)
        n_el_z = int(n_elements_per_unit_length * z_len)

        element_length = 1 / n_elements_per_unit_length
        element_half_length = element_length / 2

        # Define the mesh grid positions
        x_grid_positions = torch.linspace(x_min, x_max, n_el_x + 1)
        y_grid_positions = torch.linspace(y_min, y_max, n_el_y + 1)
        z_grid_positions = torch.linspace(z_min, z_max, n_el_z + 1)
        x_grid, y_grid, z_grid = torch.meshgrid(x_grid_positions, y_grid_positions, z_grid_positions, indexing='ij')
        grid = torch.stack((x_grid, y_grid, z_grid), dim=-1)

        # Define the mesh center points
        x_center_positions = torch.linspace(x_min + element_half_length, element_length * n_el_x - element_half_length, n_el_x)
        y_center_positions = torch.linspace(y_min + element_half_length, element_length * n_el_y - element_half_length, n_el_y)
        z_center_positions = torch.linspace(z_min + element_half_length, element_length * n_el_z - element_half_length, n_el_z)
        x_centers, y_centers, z_centers = torch.meshgrid(x_center_positions, y_center_positions, z_center_positions, indexing='ij')

        # Read the unit cube MDBD kernel
        SPI2py_path = os.path.dirname(os.path.dirname(__file__))
        mdbd_unit_cube_filepath = os.path.join('models\\projection', mdbd_unit_cube_filepath)
        mdbd_unit_cube_filepath = os.path.join(SPI2py_path, mdbd_unit_cube_filepath)


        # TODO Remove num spheres...
        kernel_positions, kernel_radii = read_xyzr_file(mdbd_unit_cube_filepath, num_spheres=10)

        kernel_positions = torch.tensor(kernel_positions, dtype=torch.float64)
        kernel_radii = torch.tensor(kernel_radii, dtype=torch.float64).view(-1, 1)

        # Truncate the number of spheres based on the minimum radius
        kernel_positions = kernel_positions[kernel_radii.flatten() > mdbd_unit_cube_min_radius]
        kernel_radii = kernel_radii[kernel_radii.flatten() > mdbd_unit_cube_min_radius]

        # Scale the sphere positions
        kernel_positions = kernel_positions * element_length
        kernel_radii = kernel_radii * element_length

        meshgrid_centers = torch.stack((x_centers, y_centers, z_centers), dim=-1)
        meshgrid_centers_expanded = meshgrid_centers.unsqueeze(3)

        all_points = meshgrid_centers_expanded + kernel_positions
        all_radii = torch.zeros((n_el_x, n_el_y, n_el_z, 1, 1)) + kernel_radii


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
        # TODO Fix copy shape, should be :,:,:,1... not 3(was copy x_centers)
        self.add_output('pseudo_densities', shape=(20,20,6))
        self.add_output('volume', val=0.0)

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

        # Convert the input to a tensor
        element_length = torch.tensor(element_length, dtype=torch.float64)
        centers = torch.tensor(centers, dtype=torch.float64)
        sample_points = torch.tensor(sample_points, dtype=torch.float64)
        sample_radii = torch.tensor(sample_radii, dtype=torch.float64)

        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64)

        # Project
        pseudo_densities = self._project(sphere_positions, sphere_radii,
                                                 element_length,
                                                 centers, sample_points, sample_radii, rho_min)

        # Calculate the volume
        volume = torch.sum(pseudo_densities) * element_length ** 3

        pseudo_densities = pseudo_densities.detach().numpy()
        volume = volume.detach().numpy()

        # Write the outputs
        outputs['pseudo_densities'] = pseudo_densities
        outputs['volume'] = volume

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

        # Convert the input to a tensor
        element_length = torch.tensor(element_length, dtype=torch.float64, requires_grad=False)
        centers = torch.tensor(centers, dtype=torch.float64, requires_grad=False)
        sample_points = torch.tensor(sample_points, dtype=torch.float64, requires_grad=False)
        sample_radii = torch.tensor(sample_radii, dtype=torch.float64, requires_grad=False)

        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=True)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64, requires_grad=False)

        # Calculate the Jacobian of the kernel
        grad_sphere_positions = jacfwd(self._project, argnums=0)
        grad_sphere_positions_val = grad_sphere_positions(sphere_positions, sphere_radii, element_length,
                                                 centers,sample_points, sample_radii, rho_min)

        # Convert the outputs to numpy arrays
        grad_sphere_positions_val = grad_sphere_positions_val.detach().numpy()

        partials['pseudo_densities', 'sphere_positions'] = grad_sphere_positions_val

    @staticmethod
    def _project(sphere_positions, sphere_radii, element_length, centers, sample_points, sample_radii, rho_min):
        """
        Projects the points to the mesh and calculates the pseudo-densities

        mesh_positions: (nx, ny, nz, n_mesh_points, 1, 3) tensor
        mesh_radii: (nx, ny, nz, n_mesh_points, 1) tensor
        object_positions_expanded: (1, 1, 1, 1, n_object_points, 3) tensor
        object_radii_expanded: (1, 1, 1, 1, n_object_points) tensor

        pseudo_densities: (nx, ny, nz, 1) tensor
        """

        nx, ny, nz, _ = centers.shape
        n_points = sphere_positions.shape[0]

        expected_element_volume = element_length ** 3
        element_volume = torch.sum((4/3) * torch.pi * sample_radii[0, 0, 0] ** 3)

        element_volumes = (4/3) * torch.pi * sample_radii ** 3

        # assert element_volume <= expected_element_volume

        # Initialize the pseudo-densities
        # pseudo_densities = torch.zeros((nx, ny, nz, 1))

        mesh_positions = sample_points
        mesh_radii = sample_radii
        object_positions = sphere_positions
        object_radii = sphere_radii
        object_radii_expanded = object_radii.transpose(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        distances = torch.norm(mesh_positions[..., None, :] - object_positions[None, None, None, None, :, :], dim=-1)

        volume_overlaps = overlap_volume_sphere_sphere(object_radii_expanded, mesh_radii, distances)

        volume_fractions = volume_overlaps / element_volumes

        pseudo_densities = torch.sum(volume_fractions, dim=(3, 4), keepdim=True).squeeze(3)

        # Clip the pseudo-densities
        pseudo_densities = torch.clip(pseudo_densities, min=rho_min, max=1)

        return pseudo_densities