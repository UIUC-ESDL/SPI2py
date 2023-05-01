import numpy as np
from SPI2py.group_model.component_spatial.distance_calculations import signed_distances_capsules_capsules


def test_capsules_capsules_nonoverlap_1():

    a = np.array([[0., 0., 0.], [0., 0., 0.]])
    b = np.array([[1., 0., 0.], [1., 0., 0.]])
    ab_radii = np.array([0.5, 0.5])

    c = np.array([[0., 0., 2.], [0., 0., 2.]])
    d = np.array([[1., 0., 2.], [1., 0., 2.]])
    cd_radii = np.array([0.5, 0.5])

    distances = signed_distances_capsules_capsules(a, b, ab_radii, c, d, cd_radii)
    expected_distances = np.array([-1.0, -1.0, -1.0, -1.0])

    assert np.allclose(distances, expected_distances)

