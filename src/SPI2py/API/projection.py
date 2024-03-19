import torch
from torch.func import jacfwd
from openmdao.api import ExplicitComponent, Group, IndepVarComp
from ..models.physics.continuum.geometric_projection import projection_volume
from ..models.kinematics.distance_calculations import signed_distances_spheres_spheres, distances_points_points

class Mesh(IndepVarComp):
    def initialize(self):

        self.options.declare('bounds', types=tuple, desc='Bounds of the mesh')
        self.options.declare('n_elements_per_unit_length', types=float, desc='Number of elements per unit length')

    def setup(self):

        # Get the options
        bounds = self.options['bounds']
        n_elements_per_unit_length = self.options['n_elements_per_unit_length']

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

        # Define the mesh center points
        x_center_positions = torch.linspace(x_min + element_half_length, element_length * n_el_x - element_half_length, n_el_x)
        y_center_positions = torch.linspace(y_min + element_half_length, element_length * n_el_y - element_half_length, n_el_y)
        z_center_positions = torch.linspace(z_min + element_half_length, element_length * n_el_z - element_half_length, n_el_z)
        x_centers, y_centers, z_centers = torch.meshgrid(x_center_positions, y_center_positions, z_center_positions, indexing='ij')

        # Declare the outputs
        self.add_output('element_length', val=element_length)
        self.add_output('x_centers', val=x_centers)
        self.add_output('y_centers', val=y_centers)
        self.add_output('z_centers', val=z_centers)
        self.add_output('x_grid', val=x_grid)
        self.add_output('y_grid', val=y_grid)
        self.add_output('z_grid', val=z_grid)
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
        self.add_input('element_length', val=0)
        self.add_input('x_centers', shape_by_conn=True)
        self.add_input('y_centers', shape_by_conn=True)
        self.add_input('z_centers', shape_by_conn=True)

        # Object Inputs
        self.add_input('sphere_positions', shape_by_conn=True)
        self.add_input('sphere_radii', shape_by_conn=True)

        # Outputs
        self.add_output('element_pseudo_densities', copy_shape='x_centers')
        self.add_output('volume', val=0.0)

    def setup_partials(self):
        self.declare_partials('element_pseudo_densities', 'sphere_positions')

    def compute(self, inputs, outputs):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        element_length = inputs['element_length']
        x_centers = inputs['x_centers']
        y_centers = inputs['y_centers']
        z_centers = inputs['z_centers']

        # Get the Object inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']

        # Convert the input to a tensor
        element_length = torch.tensor(element_length, dtype=torch.float64)
        x_centers = torch.tensor(x_centers, dtype=torch.float64)
        y_centers = torch.tensor(y_centers, dtype=torch.float64)
        z_centers = torch.tensor(z_centers, dtype=torch.float64)

        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64)

        # Project
        element_pseudo_densities = self._project(sphere_positions, sphere_radii,
                                                 element_length,
                                                 x_centers, y_centers, z_centers, rho_min)

        # Calculate the volume
        volume = torch.sum(element_pseudo_densities) * element_length ** 3

        element_pseudo_densities = element_pseudo_densities.detach().numpy()
        volume = volume.detach().numpy()

        # Write the outputs
        outputs['element_pseudo_densities'] = element_pseudo_densities
        outputs['volume'] = volume

    def compute_partials(self, inputs, partials):

        # Get the options
        rho_min = self.options['rho_min']

        # Get the Mesh inputs
        element_length = inputs['element_length']
        x_centers = inputs['x_centers']
        y_centers = inputs['y_centers']
        z_centers = inputs['z_centers']

        # Get the Object inputs
        sphere_positions = inputs['sphere_positions']
        sphere_radii = inputs['sphere_radii']

        # Convert the input to a tensor
        element_length = torch.tensor(element_length, dtype=torch.float64, requires_grad=False)
        x_centers = torch.tensor(x_centers, dtype=torch.float64, requires_grad=False)
        y_centers = torch.tensor(y_centers, dtype=torch.float64, requires_grad=False)
        z_centers = torch.tensor(z_centers, dtype=torch.float64, requires_grad=False)

        sphere_positions = torch.tensor(sphere_positions, dtype=torch.float64, requires_grad=True)
        sphere_radii = torch.tensor(sphere_radii, dtype=torch.float64, requires_grad=False)

        # Calculate the Jacobian of the kernel
        grad_sphere_positions = jacfwd(self._project, argnums=0)
        grad_sphere_positions_val = grad_sphere_positions(sphere_positions, sphere_radii, element_length,
                                                 x_centers, y_centers, z_centers, rho_min)

        partials['element_pseudo_densities', 'sphere_positions'] = grad_sphere_positions_val

    @staticmethod
    def _project(sphere_positions, sphere_radii, element_length, x_centers, y_centers, z_centers, rho_min):
        """
        Projects the points to the mesh and calculates the pseudo-densities
        """

        nx, ny, nz = x_centers.shape

        # Reshape the mesh_centers to (nx*ny*nz, 3)
        mesh_centers = torch.stack((x_centers, y_centers, z_centers), dim=3).reshape(-1, 3)
        mesh_radii = element_length/2 * torch.ones((mesh_centers.shape[0], 1))

        # Initialize influence map
        pseudo_densities = torch.zeros(mesh_radii.shape)

        # Spread of the influence, potentially adjust based on application
        sigma = 0.025  #0.255  # 0.2125

        # def sphere_overlap(R, r, d):
        #     # min_volume = torch.zeros_like(r)
        #     # max_volume_1 = (4/3) * torch.pi * (R*torch.ones_like(r))**3
        #     # max_volume_2 = (4/3) * torch.pi * r**3
        #     # minimax_volume = torch.minimum(max_volume_1, max_volume_2)
        #
        #     term_1 = (R+r-d)**2
        #     term_2 = (d**2 + 2*d*r - 3*r**2 + 2*d*R + 6*r*R - 3*R**2)
        #     overlap_vol = (torch.pi * term_1 * term_2) / (12*d)
        #
        #     # overlap_vol = torch.clip(overlap_vol, min=min_volume, max=minimax_volume)
        #     return overlap_vol

        for i in range(len(sphere_positions)):

            # Calculate adjusted distances from sphere center to each cell center
            distances = -signed_distances_spheres_spheres(sphere_positions[i].unsqueeze(1),
                                                             sphere_radii[i].unsqueeze(1), mesh_centers, mesh_radii)

            # Ensure non-negative distances for the influence calculation
            distances = torch.maximum(distances, torch.zeros_like(distances))

            # Apply Gaussian kernel, adjusting influence based on adjusted distance
            sphere_pseudo_densities = torch.exp(-distances ** 2 / (2 * sigma ** 2))

            # Sum the influences of this sphere onto the mesh cells
            pseudo_densities += sphere_pseudo_densities.T

            # Calculate the distances between the sphere and the mesh centers
            # signed_distances = -signed_distances_spheres_spheres(sphere_positions[i].unsqueeze(1),
            #                                                      sphere_radii[i].unsqueeze(1), mesh_centers, mesh_radii)
            #
            # distances = distances_points_points(sphere_positions[i].unsqueeze(1), mesh_centers)
            #
            # # Only evaluate overlapping sphere pairs
            # condition = signed_distances < 0
            # overlapping_indices = torch.where(condition)[1]
            #
            # # Calculate the overlap volume
            # overlap_volume = sphere_overlap(sphere_radii[i], mesh_radii[overlapping_indices], distances.T[overlapping_indices])
            #
            # # Add the overlap volume to the influence map
            # pseudo_densities[overlapping_indices] += overlap_volume



        # Reshape back to (nx, ny, nz, 1)
        pseudo_densities = pseudo_densities.reshape(nx, ny, nz, 1)

        # # Transpose to get (3, nx, ny, nz)
        # pseudo_densities = pseudo_densities.permute(3, 0, 1, 2)

        # Normalize the pseudo-densities
        # max_pd = torch.max(pseudo_densities)
        # min_pd = torch.min(pseudo_densities)
        # pseudo_densities = (pseudo_densities - min_pd) / (max_pd - min_pd)

        return pseudo_densities