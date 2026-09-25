"""Seeded abstract industrial inspection worlds, in meters and seconds."""
from __future__ import annotations

import random
from .contracts import Box, Scenario
from .geometry import point_box_distance

CATEGORIES = (
    "nominal", "wind", "sensor_noise", "sensor_dropout", "latency",
    "low_battery", "moving_obstacle", "clock_reset", "compute_stall", "compound",
)


def generate_scenario(seed: int, category: str | None = None) -> Scenario:
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    category = CATEGORIES[seed % len(CATEGORIES)] if category is None else category
    if category not in CATEGORIES:
        raise ValueError(f"unknown scenario category: {category}")
    # Geometry is independent of category, enabling comparisons of fault regimes.
    rng = random.Random(seed)
    home = (3.0, 3.0, 3.0)
    waypoints = (
        (39.0 + rng.uniform(-1, 1), 8.0 + rng.uniform(-1, 1), 4.0),
        (40.0 + rng.uniform(-1, 1), 28.0 + rng.uniform(-1, 1), 5.0),
        (9.0 + rng.uniform(-1, 1), 28.0 + rng.uniform(-1, 1), 4.0),
    )
    boxes = []
    # Ground-attached fixtures below z=9.2 leave a connected overhead free region.
    # Each waypoint has a clear vertical access column, so all are reachable.
    for i in range(10):
        for _ in range(100):
            x, y = rng.uniform(7.0, 40.0), rng.uniform(5.0, 28.0)
            width, depth = rng.uniform(2.0, 5.0), rng.uniform(2.0, 5.0)
            candidate = Box(f"fixture-{i:02d}", (x, y, 0.0),
                            (min(x + width, 44.0), min(y + depth, 32.0),
                             rng.uniform(3.0, 9.2)))
            # Test XY footprint using a point inside the box's vertical extent.
            if all(point_box_distance((p[0], p[1], 1.0), candidate) >= 1.8
                   for p in (home,) + waypoints):
                boxes.append(candidate)
                break
    if category in ("moving_obstacle", "compound"):
        boxes.append(Box("moving-inspection-fixture", (21.0, 5.0, 1.0),
                         (23.0, 7.0, 7.0), (0.72, 0.0, 0.0)))
    wind = (0.0, 0.0, 0.0)
    if category in ("wind", "compound"):
        wind = (rng.uniform(-1.4, 1.4), rng.uniform(-1.4, 1.4), rng.uniform(-0.2, 0.2))
    return Scenario(
        seed=seed, category=category, bounds=(48.0, 36.0, 14.0), home=home,
        waypoints=waypoints, obstacles=tuple(boxes), wind=wind,
        sensor_noise=0.32 if category in ("sensor_noise", "compound") else 0.02,
        dropout_probability=0.80 if category in ("sensor_dropout", "compound") else 0.0,
        latency_steps=6 if category in ("latency", "compound") else 0,
        fault_start=18.0 + (seed % 11), fault_duration=6.0 + (seed % 5),
        initial_battery_wh=rng.uniform(3.2, 4.8) if category == "low_battery" else 22.0,
    )
