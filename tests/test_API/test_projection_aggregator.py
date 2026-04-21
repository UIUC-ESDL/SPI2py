import numpy as np
import jax.numpy as jnp

from SPI2py.API.projection import ProjectionAggregator
from SPI2py.models.geometry.cylinders import create_cylinders
from SPI2py.models.projection.mesh_kernels import default_projection_kernel
from SPI2py.models.projection.projection import project_capsules, project_component


def test_full_grid_projection_returns_zero_for_off_grid_component():
    mesh_centers = jnp.array([[[[[0.0, 0.0, 0.0]]]]])
    mesh_size = jnp.array([1.0])
    kernel_centers = jnp.array([[0.0, 0.0, 0.0]])
    kernel_radii = jnp.array([[0.5]])
    obj_centers = jnp.array([[10.0, 10.0, 10.0]])
    obj_radii = jnp.array([[0.1]])

    densities, penalized_densities = project_component(
        mesh_centers, mesh_size, obj_centers, obj_radii, kernel_centers, kernel_radii)

    np.testing.assert_allclose(densities, jnp.zeros((1, 1, 1)))
    np.testing.assert_allclose(penalized_densities, jnp.zeros((1, 1, 1)))


def test_project_component_uses_default_kernel_when_omitted():
    mesh_centers = jnp.array([[[[[0.0, 0.0, 0.0]]]]])
    mesh_size = jnp.array([1.0])
    obj_centers = jnp.array([[0.0, 0.0, 0.0]])
    obj_radii = jnp.array([[0.25]])
    kernel_centers, kernel_radii = default_projection_kernel()

    default_result = project_component(mesh_centers, mesh_size, obj_centers, obj_radii)
    explicit_result = project_component(
        mesh_centers, mesh_size, obj_centers, obj_radii, kernel_centers, kernel_radii)

    for default_value, explicit_value in zip(default_result, explicit_result):
        np.testing.assert_allclose(default_value, explicit_value)


def test_combined_aggregator_matches_individual_projection_math():
    mesh_centers = jnp.array([[[[[0.0, 0.0, 0.0]]],
                               [[[1.0, 0.0, 0.0]]]]])
    mesh_size = jnp.array([1.0])
    kernel_centers = jnp.array([[0.0, 0.0, 0.0]])
    kernel_radii = jnp.array([[0.5]])
    rho_min = 1e-2

    component_centers = [jnp.array([[0.0, 0.0, 0.0]])]
    component_radii = [jnp.array([[0.25]])]
    component_heat_loads = [jnp.array(2.0)]
    interconnect_points = [jnp.array([[0.0, 0.0, 0.0],
                                      [1.0, 0.0, 0.0]])]
    interconnect_radii = [jnp.array([[0.2], [0.2]])]
    interconnect_heat_loads = [jnp.array(3.0)]

    _, component_penalized = project_component(
        mesh_centers, mesh_size, component_centers[0], component_radii[0],
        kernel_centers, kernel_radii)

    start_points, end_points, radii = create_cylinders(interconnect_points[0], interconnect_radii[0])
    _, interconnect_penalized = project_capsules(
        mesh_centers, mesh_size, kernel_centers, kernel_radii,
        start_points, end_points, radii)

    expected = ProjectionAggregator._compute_individual_primal(
        [component_penalized, interconnect_penalized],
        [component_heat_loads[0] * component_penalized,
         interconnect_heat_loads[0] * interconnect_penalized],
        rho_min)
    actual = ProjectionAggregator._compute_combined_primal(
        mesh_centers, mesh_size, kernel_centers, kernel_radii,
        component_centers, component_radii, component_heat_loads,
        interconnect_points, interconnect_radii, interconnect_heat_loads,
        rho_min)

    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value)
