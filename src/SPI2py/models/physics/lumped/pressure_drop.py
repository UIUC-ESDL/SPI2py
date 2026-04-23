from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class Fluid:
    """Fluid properties for incompressible pressure-drop calculations."""

    name: str
    density: float
    dynamic_viscosity: float


# Predefined fluids
WATER = Fluid("Water", density=998.2, dynamic_viscosity=0.001002)
AIR = Fluid("Air", density=1.225, dynamic_viscosity=1.81e-5)


def smooth_pipe_friction_factor(reynolds_number):
    """
    Smooth-pipe Darcy friction factor with a blended transition region.

    Laminar flow uses 64/Re. Turbulent flow uses the Blasius correlation, which
    is appropriate for smooth pipes over ordinary turbulent Reynolds numbers.
    The transition from Re=2300 to Re=4000 is blended to avoid a hard model
    failure during optimization.
    """
    re = jnp.maximum(jnp.asarray(reynolds_number), 1e-12)
    laminar = 64.0 / re
    turbulent = 0.3164 / (re ** 0.25)

    transition = jnp.clip((re - 2300.0) / (4000.0 - 2300.0), 0.0, 1.0)
    transition = transition * transition * (3.0 - 2.0 * transition)
    return (1.0 - transition) * laminar + transition * turbulent


def bend_loss_coefficient(turn_angle, friction_factor, bend_radius_ratio=3.0):
    """
    Estimate bend-loss coefficient for a smooth elbow.

    The angle is the pipe turn angle in radians, where 0 is straight and pi/2 is
    a 90-degree bend. The ratio is bend radius divided by pipe diameter.
    """
    angle = jnp.clip(jnp.asarray(turn_angle), 0.0, jnp.pi)
    ratio = jnp.asarray(bend_radius_ratio)
    sin_half = jnp.sin(angle / 2.0)

    coefficient = (
        friction_factor * angle * ratio
        + (0.1 + 2.4 * friction_factor) * sin_half
        + (
            6.6
            * friction_factor
            * (sin_half + jnp.sqrt(jnp.maximum(sin_half, 0.0) + 1e-12))
        )
        / (ratio ** (4.0 * angle / jnp.pi))
    )
    return jnp.where(angle > 1e-10, coefficient, 0.0)


def calculate_bend_angle(coord1, coord2, coord3, d=None, tolc=1e-12):
    """
    Calculate the pipe turn angle at a bend in radians.

    Coordinates are interpreted as centerline stations. A straight run returns
    0.0 and a right-angle bend returns pi/2. If ``d`` is provided, the function
    returns ``(angle, tangent_length)`` for backward compatibility with older
    callers; pressure-drop calculations do not use that tangent length.
    """
    v1 = np.asarray(coord1, dtype=float) - np.asarray(coord2, dtype=float)
    v2 = np.asarray(coord3, dtype=float) - np.asarray(coord2, dtype=float)

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 <= tolc or norm2 <= tolc:
        raise ValueError("Bend angle requires nonzero adjacent segment lengths.")

    cos_included = np.dot(v1 / norm1, v2 / norm2)
    included_angle = np.arccos(np.clip(cos_included, -1.0, 1.0))
    turn_angle = float(np.pi - included_angle)

    if d is None:
        return turn_angle

    bend_radius = 3.0 * float(d)
    tangent_length = bend_radius * np.tan(turn_angle / 2.0)
    return turn_angle, tangent_length


def pressure_drop_primal(
    coordinates,
    pipe_radius,
    flow_rate,
    density=WATER.density,
    dynamic_viscosity=WATER.dynamic_viscosity,
    bend_radius_ratio=3.0,
):
    """
    JAX-compatible pressure-drop calculation.

    Coordinates are centerline stations. Major loss is computed over the
    polyline centerline length, and bend loss is added at each interior station.
    """
    coordinates = jnp.asarray(coordinates)
    pipe_radius = jnp.mean(jnp.ravel(jnp.asarray(pipe_radius)))
    flow_rate = jnp.mean(jnp.ravel(jnp.asarray(flow_rate)))
    density = jnp.asarray(density)
    dynamic_viscosity = jnp.asarray(dynamic_viscosity)

    pipe_diameter = 2.0 * pipe_radius
    cross_sectional_area = jnp.pi * pipe_radius ** 2
    flow_velocity = flow_rate / cross_sectional_area
    velocity_head = flow_velocity ** 2 / (2.0 * 9.81)

    segment_vectors = coordinates[1:] - coordinates[:-1]
    segment_lengths = jnp.linalg.norm(segment_vectors, axis=1)
    centerline_length = jnp.sum(segment_lengths)

    reynolds_number = (
        density * jnp.abs(flow_velocity) * pipe_diameter / dynamic_viscosity
    )
    friction_factor = smooth_pipe_friction_factor(reynolds_number)

    major_loss_head = (
        friction_factor * (centerline_length / pipe_diameter) * velocity_head
    )

    minor_loss_head = 0.0
    if coordinates.shape[0] > 2:
        incoming = coordinates[:-2] - coordinates[1:-1]
        outgoing = coordinates[2:] - coordinates[1:-1]
        incoming_norms = jnp.linalg.norm(incoming, axis=1, keepdims=True)
        outgoing_norms = jnp.linalg.norm(outgoing, axis=1, keepdims=True)
        incoming_unit = incoming / incoming_norms
        outgoing_unit = outgoing / outgoing_norms

        cos_included = jnp.sum(incoming_unit * outgoing_unit, axis=1)
        included_angles = jnp.arccos(jnp.clip(cos_included, -1.0, 1.0))
        turn_angles = jnp.pi - included_angles
        bend_coefficients = bend_loss_coefficient(
            turn_angles, friction_factor, bend_radius_ratio
        )
        minor_loss_head = jnp.sum(bend_coefficients) * velocity_head

    total_head_loss = major_loss_head + minor_loss_head
    total_pressure_drop = total_head_loss * density * 9.81
    return jnp.atleast_1d(total_pressure_drop)


def _validate_scalar(name, value):
    arr = np.asarray(value, dtype=float)
    if arr.size != 1:
        raise ValueError(f"{name} must be scalar.")
    scalar = float(arr.reshape(-1)[0])
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite.")
    if scalar <= 0.0:
        raise ValueError(f"{name} must be greater than zero.")
    return scalar


def _validate_coordinates(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_points, 3).")
    if coordinates.shape[0] < 2:
        raise ValueError("At least two coordinates are required.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must be finite.")

    segment_lengths = np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
    if np.any(segment_lengths <= 1e-12):
        raise ValueError("coordinates must not contain zero-length pipe segments.")
    return coordinates


def _pressure_drop_details(
    coordinates,
    pipe_radius,
    flow_rate,
    fluid,
    bend_radius_ratio,
):
    pipe_diameter = 2.0 * pipe_radius
    cross_sectional_area = np.pi * pipe_radius ** 2
    flow_velocity = flow_rate / cross_sectional_area
    reynolds_number = (
        fluid.density * abs(flow_velocity) * pipe_diameter / fluid.dynamic_viscosity
    )
    friction_factor = float(smooth_pipe_friction_factor(reynolds_number))
    velocity_head = flow_velocity ** 2 / (2.0 * 9.81)

    details = []
    for i, (start, end) in enumerate(zip(coordinates[:-1], coordinates[1:])):
        segment_length = np.linalg.norm(end - start)
        major_loss_head = (
            friction_factor * (segment_length / pipe_diameter) * velocity_head
        )

        bend_angle = 0.0
        minor_loss_head = 0.0
        if i < coordinates.shape[0] - 2:
            bend_angle = calculate_bend_angle(
                coordinates[i], coordinates[i + 1], coordinates[i + 2]
            )
            k_bend = float(
                bend_loss_coefficient(bend_angle, friction_factor, bend_radius_ratio)
            )
            minor_loss_head = k_bend * velocity_head

        segment_pressure_drop = (
            major_loss_head + minor_loss_head
        ) * fluid.density * 9.81

        details.append(
            {
                "segment": i + 1,
                "length": segment_length,
                "flow_velocity": flow_velocity,
                "reynolds_number": reynolds_number,
                "friction_factor": friction_factor,
                "major_head_loss": major_loss_head,
                "minor_head_loss": minor_loss_head,
                "bend_angle": bend_angle,
                "bend_angle_degrees": np.degrees(bend_angle),
                "segment_pressure_drop": segment_pressure_drop,
            }
        )
    return details


def calculate_pressure_drop(
    coordinates,
    pipe_radius,
    fluid=WATER,
    flow_rate=None,
    bend_radius_ratio=3.0,
    return_details=False,
):
    """
    Calculate pressure drop in a pipe with multiple segments and bends.

    The model assumes incompressible flow in a smooth pipe. Coordinates are
    interpreted as centerline stations. Major loss is computed using the
    Darcy-Weisbach equation, and each interior station contributes an empirical
    bend loss based on the turn angle.
    """
    coordinates = _validate_coordinates(coordinates)
    pipe_radius = _validate_scalar("pipe_radius", pipe_radius)
    bend_radius_ratio = _validate_scalar("bend_radius_ratio", bend_radius_ratio)
    if flow_rate is None:
        raise ValueError("flow_rate is required.")
    flow_rate = _validate_scalar("flow_rate", flow_rate)
    _validate_scalar("fluid.density", fluid.density)
    _validate_scalar("fluid.dynamic_viscosity", fluid.dynamic_viscosity)

    total_pressure_drop = float(
        pressure_drop_primal(
            coordinates,
            pipe_radius,
            flow_rate,
            fluid.density,
            fluid.dynamic_viscosity,
            bend_radius_ratio,
        )[0]
    )

    if not return_details:
        return total_pressure_drop

    return {
        "total_pressure_drop": total_pressure_drop,
        "fluid": fluid.name,
        "segments": _pressure_drop_details(
            coordinates, pipe_radius, flow_rate, fluid, bend_radius_ratio
        ),
    }
