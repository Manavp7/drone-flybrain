"""Shared SI-unit contracts. World coordinates: x/y horizontal, z up.

The simulator supplies geometric observations; it does not render camera images
or implement an actual visual-inertial estimator. These boundaries are explicit.
"""
from __future__ import annotations
from dataclasses import dataclass, field

Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class Box:
    id: str
    low: Vec3
    high: Vec3
    velocity: Vec3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Scenario:
    seed: int
    category: str
    bounds: Vec3
    home: Vec3
    waypoints: tuple[Vec3, ...]
    obstacles: tuple[Box, ...]
    wind: Vec3 = (0.0, 0.0, 0.0)
    sensor_noise: float = 0.02
    dropout_probability: float = 0.0
    latency_steps: int = 0
    fault_start: float = 25.0
    fault_duration: float = 8.0
    initial_battery_wh: float = 22.0
    max_time: float = 180.0
    dt: float = 0.2
    sensor_range: float = 18.0


@dataclass
class VehicleState:
    time: float
    position: Vec3
    velocity: Vec3
    battery_wh: float
    energy_used_wh: float = 0.0


@dataclass(frozen=True)
class Observation:
    capture_time: float
    receive_time: float
    position: Vec3
    velocity: Vec3
    obstacles: tuple[Box, ...]
    battery_wh: float
    valid: bool = True
    fault: str = ""


@dataclass(frozen=True)
class BrainOutput:
    speed_scale: float = 1.0
    caution: float = 0.0
    healthy: bool = True
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Guidance:
    velocity: Vec3
    mode: str
    reason: str
    valid_until: float


@dataclass
class EpisodeResult:
    seed: int
    category: str
    variant: str
    outcome: str
    mission_complete: bool
    returned_home: bool
    collision: bool
    geofence_violation: bool
    waypoints_completed: int
    waypoints_total: int
    simulated_seconds: float
    energy_wh: float
    minimum_clearance_m: float
    distance_m: float
    interventions: int
    stale_observations: int
    brain_rejections: int
    wall_seconds: float
    trajectory: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)


def scenario_dict(s: Scenario) -> dict:
    from dataclasses import asdict
    return asdict(s)
