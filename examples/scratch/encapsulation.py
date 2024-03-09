import torch

from SPI2py.models.kinematics.distance_calculations import signed_distances_spheres_spheres

a = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float64)
a_radii = torch.tensor([1, 1], dtype=torch.float64)

# Non-interfering
b = torch.tensor([[0, 0, 3], [0, 0, 3]], dtype=torch.float64)

# Tangent
# b = torch.tensor([[0, 0, 2], [0, 0, 2]], dtype=torch.float64)

# Interfering
# b = torch.tensor([[0, 0, 0], [0, 0, 0]], dtype=torch.float64)
b_radii = torch.tensor([1, 1], dtype=torch.float64)

print(signed_distances_spheres_spheres(a, a_radii, b, b_radii))