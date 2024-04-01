# import os
#
# from jax import numpy as jnp
# from openmdao.core.indepvarcomp import IndepVarComp
#
# # from SPI2py.models.geometry.spherical_decomposition import read_xyzr_file
#
#
# class Mesh(IndepVarComp):
#     def initialize(self):
#
#         self.options.declare('bounds', types=tuple, desc='Bounds of the mesh')
#         self.options.declare('n_elements_per_unit_length', types=float, desc='Number of elements per unit length')
#         self.options.declare('mdbd_unit_cube_filepath', types=str)
#         self.options.declare('mdbd_unit_cube_min_radius', types=float)
#
#     def setup(self):
#
#         # Get the options
#         bounds = self.options['bounds']
#         n_elements_per_unit_length = self.options['n_elements_per_unit_length']
#         mdbd_unit_cube_filepath = self.options['mdbd_unit_cube_filepath']
#         mdbd_unit_cube_min_radius = self.options['mdbd_unit_cube_min_radius']
#
#         # Determine the number of elements in each direction
#         x_min, x_max, y_min, y_max, z_min, z_max = bounds
#
#         x_len = x_max - x_min
#         y_len = y_max - y_min
#         z_len = z_max - z_min
#
#         n_el_x = int(n_elements_per_unit_length * x_len)
#         n_el_y = int(n_elements_per_unit_length * y_len)
#         n_el_z = int(n_elements_per_unit_length * z_len)
#
#         element_length = 1 / n_elements_per_unit_length
#         element_half_length = element_length / 2
#
#         # Define the mesh grid positions
#         x_grid_positions = jnp.linspace(x_min, x_max, n_el_x + 1)
#         y_grid_positions = jnp.linspace(y_min, y_max, n_el_y + 1)
#         z_grid_positions = jnp.linspace(z_min, z_max, n_el_z + 1)
#         x_grid, y_grid, z_grid = jnp.meshgrid(x_grid_positions, y_grid_positions, z_grid_positions, indexing='ij')
#         grid = jnp.stack((x_grid, y_grid, z_grid), axis=-1)
#
#         # Define the mesh center points
#         x_center_positions = jnp.linspace(x_min + element_half_length, element_length * n_el_x - element_half_length, n_el_x)
#         y_center_positions = jnp.linspace(y_min + element_half_length, element_length * n_el_y - element_half_length, n_el_y)
#         z_center_positions = jnp.linspace(z_min + element_half_length, element_length * n_el_z - element_half_length, n_el_z)
#         x_centers, y_centers, z_centers = jnp.meshgrid(x_center_positions, y_center_positions, z_center_positions, indexing='ij')
#
#         # Read the unit cube MDBD kernel
#         SPI2py_path = os.path.dirname(os.path.dirname(__file__))
#         mdbd_unit_cube_filepath = os.path.join('models\\projection', mdbd_unit_cube_filepath)
#         mdbd_unit_cube_filepath = os.path.join(SPI2py_path, mdbd_unit_cube_filepath)
#
#
#         # TODO Remove num spheres...
#         kernel_positions, kernel_radii = read_xyzr_file(mdbd_unit_cube_filepath, num_spheres=10)
#
#         # kernel_positions = torch.tensor(kernel_positions, dtype=torch.float64)
#         # kernel_radii = torch.tensor(kernel_radii, dtype=torch.float64).view(-1, 1)
#         kernel_positions = jnp.array(kernel_positions)
#         kernel_radii = jnp.array(kernel_radii).reshape(-1, 1)
#
#         # Truncate the number of spheres based on the minimum radius
#         kernel_positions = kernel_positions[kernel_radii.flatten() > mdbd_unit_cube_min_radius]
#         kernel_radii = kernel_radii[kernel_radii.flatten() > mdbd_unit_cube_min_radius]
#
#         # Scale the sphere positions
#         kernel_positions = kernel_positions * element_length
#         kernel_radii = kernel_radii * element_length
#
#         meshgrid_centers = jnp.stack((x_centers, y_centers, z_centers), axis=-1)
#         meshgrid_centers_expanded = jnp.expand_dims(meshgrid_centers, axis=3)
#
#
#         all_points = meshgrid_centers_expanded + kernel_positions
#         all_radii = jnp.zeros((n_el_x, n_el_y, n_el_z, 1, 1)) + kernel_radii
#
#
#         # Declare the outputs
#         self.add_output('element_length', val=element_length)
#         self.add_output('centers', val=meshgrid_centers)
#         self.add_output('grid', val=grid)
#         self.add_output('n_el_x', val=n_el_x)
#         self.add_output('n_el_y', val=n_el_y)
#         self.add_output('n_el_z', val=n_el_z)
#         self.add_output('sample_points', val=all_points)
#         self.add_output('sample_radii', val=all_radii)

def read_xyzr_file(filepath, num_spheres=100):
    """
    Reads a .xyzr file and returns the positions and radii of the spheres.

    TODO Remove num spheres

    :param filepath:
    :param num_spheres:
    :return: positions, radii
    """

    with open(filepath, 'r') as f:
        lines = f.readlines()

    if num_spheres is not None and num_spheres > len(lines):
        raise ValueError('num_spheres must be less than the number of spheres in the file')

    # Truncate the number of spheres as specified
    lines = lines[0:num_spheres]

    positions = []
    radii = []

    for line in lines:
        x, y, z, r = line.split()

        positions.append([float(x), float(y), float(z)])
        radii.append(float(r))

    return positions, radii
