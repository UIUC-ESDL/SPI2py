import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from SPI2py.API.system import LinearSplineComponent


def test_linear_spline_accepts_ports_alias_and_transforms_capsule():
    prob = om.Problem()
    prob.model.add_subsystem(
        'spline',
        LinearSplineComponent(
            start_points=[[0.0, 0.0, 0.0]],
            end_points=[[1.0, 0.0, 0.0]],
            radii=[0.2],
            ports=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            color='blue',
        ),
    )

    prob.setup()
    prob.set_val('spline.translation', [1.0, 2.0, 3.0])
    prob.run_model()

    np.testing.assert_allclose(prob.get_val('spline.updated_start_points'), [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(prob.get_val('spline.updated_end_points'), [[2.0, 2.0, 3.0]])
    np.testing.assert_allclose(prob.get_val('spline.updated_ports'), [[1.0, 2.0, 3.0], [2.0, 2.0, 3.0]])
    np.testing.assert_allclose(prob.get_val('spline.updated_radii'), [[0.2]])


def test_linear_spline_partials():
    prob = om.Problem()
    prob.model.add_subsystem(
        'spline',
        LinearSplineComponent(
            start_points=[[0.0, 0.0, 0.0]],
            end_points=[[1.0, 0.0, 0.0]],
            radii=[0.2],
            port_positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ),
    )

    prob.setup()
    prob.set_val('spline.translation', [1.0, 2.0, 3.0])
    prob.set_val('spline.rotation', [0.1, 0.2, 0.3])
    prob.run_model()

    partials = prob.check_partials(out_stream=None, method='fd', step=1e-6)
    assert_check_partials(partials, atol=1e-4, rtol=1e-4)


def test_linear_spline_allows_sphere_like_degenerate_capsule():
    prob = om.Problem()
    prob.model.add_subsystem(
        'spline',
        LinearSplineComponent(
            start_points=[[0.0, 0.0, 0.5]],
            end_points=[[0.0, 0.0, 0.5]],
            radii=[1.5],
            port_positions=[[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
            color='orange',
        ),
    )

    prob.setup()
    prob.set_val('spline.translation', [1.0, 2.0, 3.0])
    prob.run_model()

    np.testing.assert_allclose(prob.get_val('spline.updated_start_points'), [[1.0, 2.0, 3.5]])
    np.testing.assert_allclose(prob.get_val('spline.updated_end_points'), [[1.0, 2.0, 3.5]])
