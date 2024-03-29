import torch


def calculate_distances_torch(object_positions, mesh_positions):
    """
    Calculate the pairwise distances between object spheres and mesh spheres using PyTorch.
    """

    # Expanding dimensions to enable broadcasting for pairwise distance calculation
    dists = torch.norm(mesh_positions[..., None, :] - object_positions[None, None, None, None, :, :], dim=-1)

    return dists


def pseudo_density_function_torch(object_radii_expanded, mesh_radii, distances):

    # Example pseudo-density calculation (this needs to be aligned with your actual computation)
    # pseudo_density = torch.sum((1 / distances) * (object_radii_expanded + mesh_radii), dim=-1)
    pseudo_density = (1 / distances) * (object_radii_expanded + mesh_radii)

    return pseudo_density


# Example usage
nx, ny, nz, np = 4, 4, 4, 10  # Dimensions for the mesh
ns = 5  # Number of object spheres

# Example data in PyTorch tensors, potentially moving to GPU if available
object_positions = torch.rand(ns, 3)
object_radii = torch.rand(ns, 1)
mesh_positions = torch.rand(nx, ny, nz, np, 3)
mesh_radii = torch.rand(nx, ny, nz, np, 1)

# Calculate distances
distances = calculate_distances_torch(object_positions, mesh_positions)

# Adjust dimensions for broadcasting in pseudo-density calculation
object_radii_expanded = object_radii.transpose(0, 1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
# mesh_radii_expanded = mesh_radii.unsqueeze(-2)  # Adding an axis for ns spheres

# Calculate pseudo-density
pseudo_density = pseudo_density_function_torch(object_radii_expanded, mesh_radii, distances)

mysum = torch.sum(pseudo_density, dim=(3,4), keepdim=True).squeeze(3)

