"""
Nodes: Tuple; (component #, port #)
Edges: Tuples of component numbers
"""

import numpy as np

from .Components import Component
from .Interconnects import Interconnect
from .Systems import System

def get_src_indices(n_components, n_interconnects, n_segments_per_interconnect):

    # Check the inputs
    assert len(n_segments_per_interconnect) == n_interconnects

    # Pre-process interconnect parameters
    n_control_points_per_interconnect = [n_segments - 1 for n_segments in n_segments_per_interconnect]
    n_control_points = sum(n_control_points_per_interconnect)

    translations_dof = np.arange(3 * n_components).reshape(-1, 3)
    rotations_dof = np.arange(3 * n_components).reshape(-1, 3)
    control_points_dof = np.arange(3 * n_control_points).reshape(-1, 3)

    # Split up the indices
    indices_translations = np.split(translations_dof, n_components)
    indices_rotations = np.split(rotations_dof, n_components)
    indices_control_points = np.split(control_points_dof, n_control_points)

    # Convert arrays to lists
    indices_translations = [indices.flatten().tolist() for indices in indices_translations]
    indices_rotations = [indices.flatten().tolist() for indices in indices_rotations]
    indices_control_points = [indices.flatten().tolist() for indices in indices_control_points]

    # Combine control point indices based on the number of control points per interconnect
    indices_interconnects = []
    i = 0
    for n in n_control_points_per_interconnect:
        start = i
        stop = i + n
        indices_interconnect = []
        for indices in indices_control_points[start:stop]:
            indices_interconnect += indices
        indices_interconnects.append(indices_interconnect)

        i += n

    indices = (indices_translations, indices_rotations, indices_interconnects)

    return indices





def calculate_component_indices(n_components, n_spheres_per_component, n_ports_per_component):

    indices_sphere_positions = []
    indices_sphere_radii = []
    indices_port_positions = []

    shape_translation = []
    shape_rotation = []

    default_translation = []
    default_rotation = []

# @staticmethod
#     def _get_src_indices(n_components):
#
#         translations_dof = np.arange(3 * n_components).reshape(-1, 3)
#         rotations_dof = np.arange(3 * n_components).reshape(-1, 3)
#
#         # Split up the indices
#         indices_translations = np.split(translations_dof, n_components)
#         indices_rotations = np.split(rotations_dof, n_components)
#
#         # Convert arrays to lists
#         indices_translations = [indices.flatten().tolist() for indices in indices_translations]
#         indices_rotations = [indices.flatten().tolist() for indices in indices_rotations]
#
#         return indices_translations, indices_rotations
#
#     @staticmethod
#     def _get_sphere_indices(n_spheres):
#
#         indices_spheres = []
#         i = 0
#         for n_sphere in n_spheres:
#             n_variables = 3 * n_sphere  # 3 for x, y, z
#             indices_component_spheres = [j for j in range(i, i + n_variables)]
#             indices_spheres.append(indices_component_spheres)
#             i += 3 * n_sphere
#
#         indices_radii = []
#         i = 0
#         for n_sphere in n_spheres:
#             n_variables = n_sphere  # 1 for radius
#             indices_component_radii = [j for j in range(i, i + n_variables)]
#             indices_radii.append(indices_component_radii)
#             i += 1 * n_sphere
#
#         return indices_spheres, indices_radii



