"""
Heat and Mass Transfer, 5th Edition
Example 5-4 Heat Loss through Chimneys
"""

import jax.numpy as jnp
import matplotlib.pyplot as plt
from SPI2py.models.physics.distributed.mesh import generate_mesh, find_active_nodes, find_face_nodes
from SPI2py.models.physics.distributed.element import assemble_local_stiffness_matrix_scalar
from SPI2py.models.physics.distributed.assembly import assemble_global_system_partition
from SPI2py.models.physics.distributed.quadrature import shape_functions
from SPI2py.models.physics.distributed.solver import solve_system_partition


def test_case_1():
    """
    A simple 2x1x1 metal cube with its bottom face held at a fixed temperature (Dirichlet BC) and its top face subject
    to convection (Robin BC).
    """

    T_inf = 293  # ambient temperature (K)
    T_fixed = 573  # fixed temperature on bottom boundary (K)

    h_c = 21  # convection coefficient (W / (m^2 * K))
    k = 56.00  # thermal conductivity (W / (m * K))

    element_size = 0.075
    nodes, elements, _, _, _, _, _, _, _ = generate_mesh(0, 2, 0, 1, 0, 1, element_size=element_size)

    densities = jnp.ones(elements.shape[0])  # uniform density

    # Find active nodes and face nodes
    top_normal = jnp.array([0, 1, 0])
    bottom_normal = jnp.array([0, -1, 0])

    robin_nodes = find_face_nodes(nodes, top_normal)
    dirichlet_nodes = find_face_nodes(nodes, bottom_normal)

    idx = jnp.arange(nodes.shape[0])
    idx_p = dirichlet_nodes
    idx_f = jnp.setdiff1d(idx, idx_p)

    robin_area = element_size * element_size
    dirichlet_temperature = T_fixed * jnp.ones(dirichlet_nodes.shape[0])

    # Initialize global stiffness matrix and force vector
    K, f, u = assemble_global_system_partition(nodes, elements,
                                               k=k,
                                               pseudo_densities=densities,
                                               r_nodes=robin_nodes,
                                               r_h=h_c,
                                               r_T_inf=T_inf,
                                               r_area=robin_area,
                                               heat_nodes=None,
                                               heat_loads=None,
                                               d_nodes=dirichlet_nodes,
                                               d_T=dirichlet_temperature)

    u = solve_system_partition(K, f, u, idx_f, idx_p)

    # Simulation vals
    T_max_known = 573.0  # The bottom face is held at 573K
    T_min_known = 496.636  # The top face


    T_max = jnp.max(u)
    T_min = jnp.min(u)

    # The maximum temperature should be at the bottom face, which is held at T_fixed
    assert jnp.isclose(T_max, T_max_known)

    # The minimum temperature should be at the top face, which is subject to convection
    # Allow for a 5% tolerance as this is a non-conformal mesh, and the mesh is coarse for quick tests
    rtol = 0.05
    assert jnp.isclose(T_min, T_min_known, rtol=rtol)


def test_case_2():
    """
    Similar to test case 1, but a heat load is applied to the rightmost face
    """

    T_inf = 293  # ambient temperature (K)
    T_fixed = 573  # fixed temperature on bottom boundary (K)

    h_c = 21  # convection coefficient (W / (m^2 * K))
    k = 56.00  # thermal conductivity (W / (m * K))
    heat_source = 25000.00  # Heat generation (W / m^2)

    element_size = 0.075
    nodes, elements, _, _, _, _, _, _, _ = generate_mesh(0, 2, 0, 1, 0, 1, element_size=element_size)

    densities = jnp.ones(elements.shape[0])  # uniform density

    # Find active nodes and face nodes
    top_normal = jnp.array([0, 1, 0])
    bottom_normal = jnp.array([0, -1, 0])
    right_normal = jnp.array([1, 0, 0])

    robin_nodes = find_face_nodes(nodes, top_normal)
    dirichlet_nodes = find_face_nodes(nodes, bottom_normal)
    heat_nodes = find_face_nodes(nodes, right_normal)

    idx = jnp.arange(nodes.shape[0])
    idx_p = dirichlet_nodes
    idx_f = jnp.setdiff1d(idx, idx_p)

    robin_area = element_size * element_size
    dirichlet_temperature = T_fixed * jnp.ones(dirichlet_nodes.shape[0])

    # Face is 1 m^2. If there were for nodes, nodal contribution would be w/4.
    # heat (W / m^2) * area (m^2) = total heat (W) --> nodal contribution (W) / n_nodes
    face_heat_loads = heat_source * jnp.ones(heat_nodes.shape[0]) / heat_nodes.size

    # Initialize global stiffness matrix and force vector
    K, f, u = assemble_global_system_partition(nodes, elements,
                                               k=k,
                                               pseudo_densities=densities,
                                               r_nodes=robin_nodes,
                                               r_h=h_c,
                                               r_T_inf=T_inf,
                                               r_area=robin_area,
                                               heat_nodes=heat_nodes,
                                               heat_loads=face_heat_loads,
                                               d_nodes=dirichlet_nodes,
                                               d_T=dirichlet_temperature)

    u = solve_system_partition(K, f, u, idx_f, idx_p)

    # Simulation vals
    T_max_known = 797.176  # Center of rightmost face, perhaps 0.75m up out of 1m
    T_min_known = 513.514  # The top leftmost front and back corners.

    T_max = jnp.max(u)
    T_min = jnp.min(u)

    # Allow for a 7.5% tolerance as this is a non-conformal mesh, and the mesh is coarse for quick tests
    rtol = 0.075
    assert jnp.isclose(T_max, T_max_known, rtol=rtol)
    assert jnp.isclose(T_min, T_min_known, rtol=rtol)


def test_case_3():
    """
    Like test case 1, but with half the thermal conductivity, as scaled by pseudo-densities.
    With a fixed temperature bottom and a convection top, a lower conduction rate does not affect the max
    temperature, but does allow the top to cool more, resulting in a slightly lower min temperature.
    """

    T_inf = 293  # ambient temperature (K)
    T_fixed = 573  # fixed temperature on bottom boundary (K)

    h_c = 21  # convection coefficient (W / (m^2 * K))
    k = 56.00  # thermal conductivity (W / (m * K))

    element_size = 0.075
    nodes, elements, _, _, _, _, _, _, _ = generate_mesh(0, 2, 0, 1, 0, 1, element_size=element_size)

    densities = 0.5 * jnp.ones(elements.shape[0])  # uniform density

    # Find active nodes and face nodes
    top_normal = jnp.array([0, 1, 0])
    bottom_normal = jnp.array([0, -1, 0])

    robin_nodes = find_face_nodes(nodes, top_normal)
    dirichlet_nodes = find_face_nodes(nodes, bottom_normal)

    idx = jnp.arange(nodes.shape[0])
    idx_p = dirichlet_nodes
    idx_f = jnp.setdiff1d(idx, idx_p)

    robin_area = element_size * element_size
    dirichlet_temperature = T_fixed * jnp.ones(dirichlet_nodes.shape[0])

    # Initialize global stiffness matrix and force vector
    K, f, u = assemble_global_system_partition(nodes, elements,
                                               k=k,
                                               pseudo_densities=densities,
                                               r_nodes=robin_nodes,
                                               r_h=h_c,
                                               r_T_inf=T_inf,
                                               r_area=robin_area,
                                               heat_nodes=None,
                                               heat_loads=None,
                                               d_nodes=dirichlet_nodes,
                                               d_T=dirichlet_temperature)

    u = solve_system_partition(K, f, u, idx_f, idx_p)

    # Simulation vals
    T_max_known = 573.0  # The bottom face is held at 573K
    T_min_known = 451.823  # The top face


    T_max = jnp.max(u)
    T_min = jnp.min(u)

    # Allow for a 7.5% tolerance as this is a non-conformal mesh, and the mesh is coarse for quick tests
    rtol = 0.075
    assert jnp.isclose(T_max, T_max_known)
    assert jnp.isclose(T_min, T_min_known, rtol=rtol)