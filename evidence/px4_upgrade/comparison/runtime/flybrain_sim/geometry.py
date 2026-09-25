"""Deterministic box geometry for the reduced-order indoor simulator.

Swept collision expands each axis by the vehicle radius. This is conservative
near box corners rather than an exact swept-sphere calculation.
"""
from __future__ import annotations

import math
from .contracts import Box, Scenario, Vec3

VEHICLE_RADIUS = 0.45


def add(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def mul(a: Vec3, value: float) -> Vec3:
    return (a[0] * value, a[1] * value, a[2] * value)


def norm(a: Vec3) -> float:
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def distance(a: Vec3, b: Vec3) -> float:
    return norm(sub(a, b))


def unit(a: Vec3) -> Vec3:
    n = norm(a)
    return mul(a, 1.0 / n) if n > 1e-12 else (0.0, 0.0, 0.0)


def clamp_norm(a: Vec3, maximum: float) -> Vec3:
    n = norm(a)
    return mul(a, maximum / n) if n > maximum else a


def point_box_distance(point: Vec3, box: Box) -> float:
    """Euclidean distance from a point to a closed axis-aligned box."""
    squared = 0.0
    for i in range(3):
        delta = max(box.low[i] - point[i], 0.0, point[i] - box.high[i])
        squared += delta * delta
    return math.sqrt(squared)


def segment_intersects_box(start: Vec3, end: Vec3, box: Box,
                           radius: float = VEHICLE_RADIUS) -> bool:
    """Closed slab test; tangency counts as collision."""
    if radius < 0.0:
        raise ValueError("radius must be nonnegative")
    enter, leave = 0.0, 1.0
    for i in range(3):
        low, high = box.low[i] - radius, box.high[i] + radius
        delta = end[i] - start[i]
        if abs(delta) < 1e-12:
            if start[i] < low or start[i] > high:
                return False
        else:
            a = (low - start[i]) / delta
            b = (high - start[i]) / delta
            if a > b:
                a, b = b, a
            enter = max(enter, a)
            leave = min(leave, b)
            if enter > leave:
                return False
    return True


def swept_collision(start: Vec3, end: Vec3, obstacles: tuple[Box, ...],
                    radius: float = VEHICLE_RADIUS) -> bool:
    return any(segment_intersects_box(start, end, box, radius) for box in obstacles)


def clearance(point: Vec3, obstacles: tuple[Box, ...]) -> float:
    """Center-to-obstacle distance; subtract radius for body clearance."""
    return min((point_box_distance(point, box) for box in obstacles), default=math.inf)


def within_bounds(point: Vec3, bounds: Vec3,
                  radius: float = VEHICLE_RADIUS) -> bool:
    return all(radius <= point[i] <= bounds[i] - radius for i in range(3))


def bounds_clearance(point: Vec3, bounds: Vec3) -> float:
    return min(min(point[i], bounds[i] - point[i]) for i in range(3))


def obstacles_at(scenario: Scenario, t: float) -> tuple[Box, ...]:
    """Moving fixtures shuttle with a smooth eight-meter bounded excursion.

    Dynamic Box.velocity encodes the peak velocity, not a constant translation.
    Returned boxes carry the instantaneous velocity for observation consumers.
    """
    if scenario.category not in ("moving_obstacle", "compound"):
        return scenario.obstacles
    result = []
    phase = (scenario.seed % 31) * 0.17
    frequency = 0.24
    wave = math.sin(frequency * max(0.0, t) + phase)
    derivative = math.cos(frequency * max(0.0, t) + phase)
    for box in scenario.obstacles:
        if box.velocity == (0.0, 0.0, 0.0):
            result.append(box)
        else:
            shift = mul(box.velocity, wave / frequency)
            result.append(Box(box.id, add(box.low, shift), add(box.high, shift),
                              mul(box.velocity, derivative)))
    return tuple(result)
