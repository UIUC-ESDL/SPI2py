import jax.numpy as jnp
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

positions = [[0.0, 0.0, 0.0],
             [-0.37, -0.37, -0.37],
             [-0.37, -0.37, 0.37],
             [-0.37, 0.37, -0.37],
             [-0.37, 0.37, 0.37],
             [0.37, -0.37, -0.37],
             [0.37, -0.37, 0.37],
             [0.37, 0.37, -0.37],
             [0.37, 0.37, 0.37]]
radii = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

positions_arr = jnp.array(positions)
radii_arr = jnp.array(radii).reshape(-1, 1)
xyzr_i = jnp.hstack((positions_arr, radii_arr))

res = refine_mdbd(xyzr_i)

print(res.x.reshape(-1, 4))


# Plot the spheres
# p = pv.Plotter()
# for i, pos in enumerate(positions):
#     p.add_mesh(pv.Sphere(radius=radii[i], center=pos), color='cyan', opacity=0.5)
#
# p.show()


