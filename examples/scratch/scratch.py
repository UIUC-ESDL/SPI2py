import jax.numpy as jnp
import numpy as np
import pyvista as pv
from SPI2py.models.geometry.spherical_decomposition import refine_mdbd

# positions = [[0.0, 0.0, 0.0],
#              [-0.317, -0.317, -0.317],
#              [-0.317, -0.317, 0.317],
#              [-0.317, 0.317, -0.317],
#              [-0.317, 0.317, 0.317],
#              [0.317, -0.317, -0.317],
#              [0.317, -0.317, 0.317],
#              [0.317, 0.317, -0.317],
#              [0.317, 0.317, 0.317]]
#
# radii = [0.5, 0.317, 0.317, 0.317, 0.317, 0.317, 0.317, 0.317, 0.317]

# positions = [[0.0, 0.0, 0.0],
#              [-0.366, -0.366, -0.366],
#              [-0.366, -0.366, 0.366],
#              [-0.366, 0.366, -0.366],
#              [-0.366, 0.366, 0.366],
#              [0.366, -0.366, -0.366],
#              [0.366, -0.366, 0.366],
#              [0.366, 0.366, -0.366],
#              [0.366, 0.366, 0.366]]
# radii = [0.5, 0.134, 0.134, 0.134, 0.134, 0.134, 0.134, 0.134, 0.134]

# positions = [[0.0, 0.0, 0.0],
#              [-0.37, -0.37, -0.37],
#              [-0.37, -0.37, 0.37],
#              [-0.37, 0.37, -0.37],
#              [-0.37, 0.37, 0.37],
#              [0.37, -0.37, -0.37],
#              [0.37, -0.37, 0.37],
#              [0.37, 0.37, -0.37],
#              [0.37, 0.37, 0.37]]
# radii = [0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# positions_arr = jnp.array(positions)
# radii_arr = jnp.array(radii).reshape(-1, 1)
# xyzr_i = jnp.hstack((positions_arr, radii_arr))
#
# res = refine_mdbd(xyzr_i)

# print(res.x.reshape(-1, 4))

# Define the radius of the spheres
radius = 1/8

# Generate a 3D grid of points within the cube
x = np.linspace(-0.5+radius, 0.5-radius, 4)
y = np.linspace(-0.5+radius, 0.5-radius, 4)
z = np.linspace(-0.5+radius, 0.5-radius, 4)
positions = np.array(np.meshgrid(x, y, z)).T.reshape(-1,3)

# Create an array of radii
radii = np.full((64,), radius)


pos_list = positions.tolist()
rad_list = radii.tolist()

# Plot the spheres
p = pv.Plotter()
for i, pos in enumerate(positions):
    p.add_mesh(pv.Sphere(radius=radii[i], center=pos), color='cyan', opacity=0.5)

# Add the cube
p.add_mesh(pv.Box(bounds=(-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)), color='white', opacity=0.5)

p.show()





