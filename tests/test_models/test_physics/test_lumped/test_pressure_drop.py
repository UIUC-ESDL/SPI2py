import numpy as np
import pytest

from SPI2py.models.physics.lumped.pressure_drop import (
    WATER,
    bend_loss_coefficient,
    calculate_bend_angle,
    calculate_pressure_drop,
    smooth_pipe_friction_factor,
)


def test_straight_pipe_matches_darcy_weisbach_major_loss():
    coordinates = np.array([[0.0, 0.0, 0.0],
                            [10.0, 0.0, 0.0]])
    pipe_radius = 0.05
    flow_rate = 0.001

    pressure_drop = calculate_pressure_drop(
        coordinates=coordinates,
        pipe_radius=pipe_radius,
        flow_rate=flow_rate,
    )

    diameter = 2.0 * pipe_radius
    area = np.pi * pipe_radius ** 2
    velocity = flow_rate / area
    reynolds_number = WATER.density * velocity * diameter / WATER.dynamic_viscosity
    friction_factor = float(smooth_pipe_friction_factor(reynolds_number))
    head_loss = friction_factor * (10.0 / diameter) * velocity ** 2 / (2.0 * 9.81)
    expected = head_loss * WATER.density * 9.81

    np.testing.assert_allclose(pressure_drop, expected)


def test_zero_degree_bend_matches_equivalent_straight_pipe():
    split_straight = calculate_pressure_drop(
        coordinates=[(0.0, 0.0, 0.0),
                     (5.0, 0.0, 0.0),
                     (10.0, 0.0, 0.0)],
        pipe_radius=0.05,
        flow_rate=0.001,
    )
    single_straight = calculate_pressure_drop(
        coordinates=[(0.0, 0.0, 0.0),
                     (10.0, 0.0, 0.0)],
        pipe_radius=0.05,
        flow_rate=0.001,
    )

    np.testing.assert_allclose(split_straight, single_straight, rtol=1e-6)


def test_right_angle_bend_adds_minor_loss():
    bent = calculate_pressure_drop(
        coordinates=[(0.0, 0.0, 0.0),
                     (5.0, 0.0, 0.0),
                     (5.0, 5.0, 0.0)],
        pipe_radius=0.05,
        flow_rate=0.001,
    )
    straight_same_length = calculate_pressure_drop(
        coordinates=[(0.0, 0.0, 0.0),
                     (10.0, 0.0, 0.0)],
        pipe_radius=0.05,
        flow_rate=0.001,
    )

    assert bent > straight_same_length
    np.testing.assert_allclose(
        calculate_bend_angle((0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (5.0, 5.0, 0.0)),
        np.pi / 2.0,
    )


def test_laminar_flow_uses_laminar_friction_instead_of_erroring():
    pressure_drop = calculate_pressure_drop(
        coordinates=[(0.0, 0.0, 0.0),
                     (10.0, 0.0, 0.0)],
        pipe_radius=0.1,
        flow_rate=1e-6,
    )

    assert pressure_drop > 0.0


def test_invalid_inputs_raise_clear_errors():
    with pytest.raises(ValueError, match="flow_rate is required"):
        calculate_pressure_drop([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], pipe_radius=0.05)

    with pytest.raises(ValueError, match="pipe_radius must be greater than zero"):
        calculate_pressure_drop(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            pipe_radius=0.0,
            flow_rate=0.001,
        )

    with pytest.raises(ValueError, match="zero-length"):
        calculate_pressure_drop(
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
            pipe_radius=0.05,
            flow_rate=0.001,
        )


def test_return_details_reports_bend_angle():
    details = calculate_pressure_drop(
        coordinates=[(0.0, 0.0, 0.0),
                     (5.0, 0.0, 0.0),
                     (5.0, 5.0, 0.0)],
        pipe_radius=0.05,
        flow_rate=0.001,
        return_details=True,
    )

    assert details["total_pressure_drop"] > 0.0
    np.testing.assert_allclose(details["segments"][0]["bend_angle"], np.pi / 2.0)
    assert details["segments"][0]["minor_head_loss"] > 0.0
    assert details["segments"][1]["bend_angle"] == 0.0


def test_bend_loss_coefficient_is_zero_for_straight_path():
    np.testing.assert_allclose(bend_loss_coefficient(0.0, 0.02), 0.0)
