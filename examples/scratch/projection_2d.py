# import numpy as np
# from scipy.stats import gaussian_kde
import numpy
import jax.numpy as np
from jax import grad, jacfwd, jacrev
from jax.scipy.stats import gaussian_kde

from time import time_ns

import matplotlib.pyplot as plt

"""
https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.stats.gaussian_kde.html
"""

# n_points = 100
# n_grid = 25
# grid_min = 0
# grid_max = 2
#
# # Generate a sample set of 2D points
# np.random.seed(0)  # For reproducibility
# points = np.random.rand(n_points, 2)
#
# # Plot the density values
# x_min = np.min(points[:, 0])
# x_max = np.max(points[:, 0])
# y_min = np.min(points[:, 1])
# y_max = np.max(points[:, 1])
#
# # Create a 2D grid
# x = np.linspace(grid_min, grid_max, n_grid)
# y = np.linspace(grid_min, grid_max, n_grid)
# x_grid, y_grid = np.meshgrid(x, y)
# grid_positions = np.vstack([x_grid.flatten(), y_grid.flatten()])
#
# # Perform KDE
# kernel = gaussian_kde(points.T, bw_method='scott')
# density_values = kernel(grid_positions).reshape(x_grid.shape)
#
#
# # Plot the results
# plt.figure(figsize=(8, 6))
# plt.imshow(np.rot90(density_values), cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
# plt.scatter(points[:, 0], points[:, 1], s=50, facecolor='white', edgecolor='black')
# plt.title('2D Gaussian KDE of Points')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.colorbar(label='Density')
#
# # Plot the grid lines
# x_lines = np.linspace(grid_min, grid_max, n_grid)
# y_lines = np.linspace(grid_min, grid_max, n_grid)
#
# # Draw grid lines
# for x in x_lines:
#     plt.vlines(x, grid_min, grid_max, colors='lightgrey', linestyles='--', linewidths=0.5)
# for y in y_lines:
#     plt.hlines(y, grid_min, grid_max, colors='lightgrey', linestyles='--', linewidths=0.5)
#
#
#
#
# plt.show()

# Generate a sample set of 2D points
numpy.random.seed(0) # For reproducibility
points = numpy.random.randn(5000, 3) # 100 points in 2D
points = np.array(points)

# Define the grid over the area of the points
x_min, x_max = -3, 3
y_min, y_max = -3, 3
z_min, z_max = -3, 3

x = np.linspace(x_min, x_max, 100)
y = np.linspace(y_min, y_max, 100)
z = np.linspace(z_min, z_max, 100)
# x_grid, y_grid = np.meshgrid(x, y)
# grid_positions = np.vstack([x_grid.flatten(), y_grid.flatten()])
x_grid, y_grid, z_grid = np.meshgrid(x, y, z)
grid_positions = np.vstack([x_grid.flatten(), y_grid.flatten(), z_grid.flatten()])

# Perform KDE
kernel = gaussian_kde(points.T, bw_method='scott')
density_values = kernel(grid_positions).reshape(x_grid.shape)

t1 = time_ns()

grad_kernel = jacfwd(lambda x: gaussian_kde(x, bw_method='scott'))
grad_kernel_val = grad_kernel(points.T)

t2 = time_ns()

# Print gradient size
print('Size:', grad_kernel_val.dataset.size)

print('Time (s):', (t2 - t1) / 1e9)


# # Plot the results
# plt.figure(figsize=(8, 6))
# plt.imshow(density_values, cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
# plt.scatter(points[:, 0], points[:, 1], s=50, facecolor='white', edgecolor='black')
# plt.title('2D Gaussian KDE of Points')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.colorbar(label='Density')
# plt.show()


# Define the gradient of the KDE
