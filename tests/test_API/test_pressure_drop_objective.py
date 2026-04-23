import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials

from SPI2py.API.objectives import PressureDrop
from SPI2py.models.physics.lumped.pressure_drop import calculate_pressure_drop


def test_pressure_drop_component_matches_model():
    coordinates = np.array([[0.0, 0.0, 0.0],
                            [5.0, 0.0, 0.0],
                            [5.0, 5.0, 0.0]])
    pipe_radius = np.array([0.05])
    flow_rate = np.array([0.001])

    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output('coordinates', val=coordinates)
    ivc.add_output('pipe_radius', val=pipe_radius)
    ivc.add_output('flow_rate', val=flow_rate)
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem(
        'pressure_drop',
        PressureDrop(),
        promotes_inputs=['coordinates', 'pipe_radius', 'flow_rate'],
    )
    prob.model.add_objective('pressure_drop.pressure_drop')

    prob.setup()
    prob.run_model()

    expected = calculate_pressure_drop(coordinates, pipe_radius[0], flow_rate=flow_rate[0])
    np.testing.assert_allclose(prob.get_val('pressure_drop.pressure_drop'), expected)


def test_pressure_drop_component_partials():
    coordinates = np.array([[0.0, 0.0, 0.0],
                            [5.0, 0.0, 0.0],
                            [5.0, 5.0, 0.0]])

    prob = om.Problem()
    ivc = om.IndepVarComp()
    ivc.add_output('coordinates', val=coordinates)
    ivc.add_output('pipe_radius', val=np.array([0.05]))
    ivc.add_output('flow_rate', val=np.array([0.001]))
    prob.model.add_subsystem('ivc', ivc, promotes=['*'])
    prob.model.add_subsystem(
        'pressure_drop',
        PressureDrop(),
        promotes_inputs=['coordinates', 'pipe_radius', 'flow_rate'],
    )

    prob.setup()
    prob.run_model()

    partials = prob.check_partials(out_stream=None, method='fd')
    assert_check_partials(partials, atol=1e-2, rtol=1e-2)
