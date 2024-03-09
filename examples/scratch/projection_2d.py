import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

"""
https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.stats.gaussian_kde.html
"""

n_points = 100
n_grid = 25
grid_min = 0
grid_max = 1

# Generate a sample set of 2D points
np.random.seed(0)  # For reproducibility
points = np.random.rand(n_points, 3)

# Plot the density values
x_min = np.min(points[:, 0])
x_max = np.max(points[:, 0])
y_min = np.min(points[:, 1])
y_max = np.max(points[:, 1])

# Create a 2D grid
x = np.linspace(grid_min, grid_max, n_grid)
y = np.linspace(grid_min, grid_max, n_grid)
x_grid, y_grid = np.meshgrid(x, y)
grid_positions = np.vstack([x_grid.flatten(), y_grid.flatten()])

# Perform KDE
kernel = gaussian_kde(points.T, bw_method='scott')
density_values = kernel(grid_positions).reshape(x_grid.shape)


# Plot the results
plt.figure(figsize=(8, 6))
plt.imshow(np.rot90(density_values), cmap=plt.cm.gist_earth_r, extent=[x_min, x_max, y_min, y_max])
plt.scatter(points[:, 0], points[:, 1], s=50, facecolor='white', edgecolor='black')
plt.title('2D Gaussian KDE of Points')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.colorbar(label='Density')

# Plot the grid lines
x_lines = np.linspace(grid_min, grid_max, n_grid)
y_lines = np.linspace(grid_min, grid_max, n_grid)

# Draw grid lines
for x in x_lines:
    plt.vlines(x, grid_min, grid_max, colors='lightgrey', linestyles='--', linewidths=0.5)
for y in y_lines:
    plt.hlines(y, grid_min, grid_max, colors='lightgrey', linestyles='--', linewidths=0.5)




plt.show()
