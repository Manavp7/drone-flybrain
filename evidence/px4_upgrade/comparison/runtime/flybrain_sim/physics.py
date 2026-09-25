"""Acceleration-limited point-mass dynamics: not a six-DOF flight model.

Energy is a deliberately explicit analytic surrogate for regression comparisons.
Its numbers do not establish real payload capacity, endurance, or battery safety.
"""
from __future__ import annotations

import math
from .contracts import Scenario, Vec3, VehicleState
from .geometry import add, clamp_norm, mul, norm, sub

MAX_SPEED = 5.0
MAX_ACCELERATION = 3.0
RESPONSE_TIME = 0.55


def disturbance_at(scenario: Scenario, t: float) -> Vec3:
    """Exogenous deterministic schedule shared across controller variants."""
    if scenario.wind == (0.0, 0.0, 0.0):
        return (0.0, 0.0, 0.0)
    phase = (scenario.seed % 101) * 0.071
    gust = 0.75 + 0.25 * math.sin(t * 0.83 + phase)
    return mul(scenario.wind, 0.45 * gust)


def step(state: VehicleState, command_velocity: Vec3, scenario: Scenario) -> VehicleState:
    dt = scenario.dt
    if dt <= 0.0:
        raise ValueError("scenario.dt must be positive")
    if not all(math.isfinite(x) for x in command_velocity):
        command_velocity = (0.0, 0.0, 0.0)
    command = clamp_norm(command_velocity, MAX_SPEED)
    acceleration = add(mul(sub(command, state.velocity), 1.0 / RESPONSE_TIME),
                       disturbance_at(scenario, state.time))
    acceleration = clamp_norm(acceleration, MAX_ACCELERATION)
    velocity = clamp_norm(add(state.velocity, mul(acceleration, dt)), MAX_SPEED)
    # Trapezoidal integration avoids an instantaneous displacement at command onset.
    position = add(state.position, mul(add(state.velocity, velocity), 0.5 * dt))
    speed = norm(velocity)
    power_w = 180.0 + 7.0 * speed * speed + 16.0 * max(velocity[2], 0.0) + 4.0 * norm(acceleration)
    energy = power_w * dt / 3600.0
    return VehicleState(
        time=state.time + dt, position=position, velocity=velocity,
        battery_wh=max(0.0, state.battery_wh - energy),
        energy_used_wh=state.energy_used_wh + energy,
    )
