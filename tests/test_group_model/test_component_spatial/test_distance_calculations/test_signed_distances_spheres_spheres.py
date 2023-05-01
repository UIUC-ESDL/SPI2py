import numpy as np
from SPI2py.group_model.component_spatial.distance_calculations import signed_distances_spheres_spheres


def test_sphere_sphere_tangent_1():

    center_a = np.array([[0., 0., 0.]])
    radii_a  = np.array([0.5])

    center_b = np.array([[1., 0., 0.]])
    radii_b  = np.array([0.5])

    distances = signed_distances_spheres_spheres(center_a, radii_a, center_b, radii_b)
    expected_distances = np.array([0.0])

    assert np.allclose(distances, expected_distances)

def test_spheres_spheres_tangent_1():

    center_a = np.array([[0., 0., 0.], [0., 0., 0.]])
    radii_a  = np.array([0.5, 0.5])

    center_b = np.array([[1., 0., 0.], [1., 0., 0.]])
    radii_b  = np.array([0.5, 0.5])

    distances = signed_distances_spheres_spheres(center_a, radii_a, center_b, radii_b)
    expected_distances = np.array([0.0, 0.0, 0.0, 0.0])

    assert np.allclose(distances, expected_distances)