"""One fixed, enclosed civilian inspection course and seeded fault regimes.

This module defines the test environment, not controller waypoints or control
hints. REFERENCE_ROUTE is an independent feasibility witness and is never passed
to the controller. Units are metres, seconds, and watt-hours.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import random

from flybrain_sim.contracts import Box, Scenario, Vec3

COURSE_ID = "enclosed-zigzag-three-aperture-v1"
PROFILES = (
    "nominal", "wind", "sensor_noise", "sensor_dropout", "latency",
    "low_battery", "moving_obstacle", "clock_reset", "compute_stall", "compound",
)
PROFILE_DEFINITIONS = {
    "nominal": {"description": "Fixed course with independent baseline sensor draws", "sensor_noise_rms_m": [0.02, 0.02]},
    "wind": {"description": "Seeded gust direction, amplitude, and phase", "wind_vector_horizontal_norm": [1.6, 3.6], "wind_vector_vertical": [-0.5, 0.5], "actual_acceleration_multiplier": [0.225, 0.45]},
    "sensor_noise": {"description": "Continuous independent bounded position/velocity error", "sensor_noise_rms_m": [0.20, 0.48], "uniform_position_bound_multiplier": math.sqrt(3.0)},
    "sensor_dropout": {"description": "Invalid observations during one fault window", "dropout_probability": [0.65, 0.98]},
    "latency": {"description": "Delayed geometric observations during one fault window", "latency_steps": [4, 12], "latency_seconds": [0.8, 2.4]},
    "low_battery": {"description": "Undercharged starts; safe early returns remain incomplete missions", "initial_battery_wh": [4.0, 8.0]},
    "moving_obstacle": {"description": "Room-two moving fixture with limited-range observations", "fixture_peak_speed_m_s": [0.75, 1.30], "excursion_m": [-4.5, 4.5], "sensor_range_m": 6.0},
    "clock_reset": {"description": "Twelve-second timestamp offset during one fault window", "clock_offset_seconds": -12.0},
    "compute_stall": {"description": "Missing-observation proxy; controller computation still executes", "controller_process_actually_stalled": False},
    "compound": {"description": "Wind, continuous noise, dropout, latency, clock offset, and moving fixture", "components": ["wind", "sensor_noise", "sensor_dropout", "latency", "clock_reset", "moving_obstacle"]},
}
BOUNDS = (48.0, 36.0, 10.0)
HOME = (4.0, 8.0, 3.5)
WAYPOINTS = ((19.0, 28.0, 5.5), (31.0, 8.0, 3.5), (43.0, 28.0, 5.5))
APERTURES = (
    {"id": "gate-1", "x": 12.0, "y": 8.0, "z": 3.5, "width": 4.2, "height": 4.2},
    {"id": "gate-2", "x": 24.0, "y": 28.0, "z": 5.5, "width": 4.2, "height": 4.2},
    {"id": "gate-3", "x": 36.0, "y": 8.0, "z": 3.5, "width": 4.2, "height": 4.2},
)


def _static_boxes() -> tuple[Box, ...]:
    boxes = [
        Box("floor", (0.0, 0.0, 0.0), (48.0, 36.0, 0.4)),
        Box("roof", (0.0, 0.0, 9.6), (48.0, 36.0, 10.0)),
        Box("west-wall", (0.0, 0.0, 0.0), (0.4, 36.0, 10.0)),
        Box("east-wall", (47.6, 0.0, 0.0), (48.0, 36.0, 10.0)),
        Box("south-wall", (0.0, 0.0, 0.0), (48.0, 0.4, 10.0)),
        Box("north-wall", (0.0, 35.6, 0.0), (48.0, 36.0, 10.0)),
    ]
    for aperture in APERTURES:
        name, x, y, z = (aperture[key] for key in ("id", "x", "y", "z"))
        y0, y1 = y - aperture["width"] / 2, y + aperture["width"] / 2
        z0, z1 = z - aperture["height"] / 2, z + aperture["height"] / 2
        boxes.extend((
            Box(f"{name}-sill", (x - 0.4, 0.0, 0.0), (x + 0.4, 36.0, z0)),
            Box(f"{name}-lintel", (x - 0.4, 0.0, z1), (x + 0.4, 36.0, 10.0)),
            Box(f"{name}-south-jamb", (x - 0.4, 0.0, z0), (x + 0.4, y0, z1)),
            Box(f"{name}-north-jamb", (x - 0.4, y1, z0), (x + 0.4, 36.0, z1)),
        ))
    return tuple(boxes)


STATIC_OBSTACLES = _static_boxes()

# Crossing a partition at any point outside its finite aperture is physically
# obstructed. The route below takes an explicitly generous centreline. Outbound
# witness endpoints are also the actual inspection targets; intermediate points
# are only for the independent geometry certificate.
OUTBOUND_REFERENCE_ROUTE = (
    HOME,
    (10.0, 8.0, 3.5), (14.0, 8.0, 3.5), WAYPOINTS[0],
    (22.0, 28.0, 5.5), (26.0, 28.0, 5.5), WAYPOINTS[1],
    (34.0, 8.0, 3.5), (38.0, 8.0, 3.5), WAYPOINTS[2],
)
REFERENCE_ROUTE = OUTBOUND_REFERENCE_ROUTE + tuple(reversed(OUTBOUND_REFERENCE_ROUTE[:-1]))


def static_geometry_sha256() -> str:
    document = {
        "course_id": COURSE_ID, "bounds": BOUNDS, "home": HOME,
        "waypoints": WAYPOINTS, "obstacles": [asdict(box) for box in STATIC_OBSTACLES],
    }
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_trial(seed: int, profile: str = "nominal") -> Scenario:
    """Keep physical geometry identical; randomise exogenous fault conditions.

    This intentionally does not reproduce the previous open-box world generator.
    A trial's seed controls sensor draws, gust phase/direction, dynamic phase,
    fault windows, and the numeric severities declared below.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if profile not in PROFILES:
        raise ValueError(f"unknown stress profile: {profile}")
    # Dedicated constant makes environment draws distinct from sensor draws.
    rng = random.Random(seed ^ 0x51A7F09)
    fault_start = rng.uniform(15.0, 65.0)
    fault_duration = rng.uniform(4.0, 12.0)
    angle = rng.uniform(-math.pi, math.pi)
    wind_strength = rng.uniform(1.6, 3.6)
    wind = ((wind_strength * math.cos(angle), wind_strength * math.sin(angle),
             rng.uniform(-0.5, 0.5)) if profile in ("wind", "compound")
            else (0.0, 0.0, 0.0))
    noise = rng.uniform(0.20, 0.48) if profile in ("sensor_noise", "compound") else 0.02
    dropout = rng.uniform(0.65, 0.98) if profile in ("sensor_dropout", "compound") else 0.0
    latency = rng.randint(4, 12) if profile in ("latency", "compound") else 0
    battery = rng.uniform(4.0, 8.0) if profile == "low_battery" else 22.0
    obstacles = STATIC_OBSTACLES
    if profile in ("moving_obstacle", "compound"):
        # Peak speed is encoded in Box.velocity. The custom bounded motion
        # function below converts it into instantaneous position and velocity.
        peak_speed = rng.uniform(0.75, 1.30)
        obstacles += (Box("moving-inspection-fixture", (17.0, 15.0, 0.4),
                          (19.0, 17.0, 6.6), (0.0, peak_speed, 0.0)),)
    return Scenario(
        seed=seed, category=profile, bounds=BOUNDS, home=HOME,
        waypoints=WAYPOINTS, obstacles=obstacles, wind=wind,
        sensor_noise=noise, dropout_probability=dropout, latency_steps=latency,
        fault_start=fault_start, fault_duration=fault_duration,
        initial_battery_wh=battery, max_time=300.0, dt=0.2, sensor_range=6.0,
    )


def obstacles_at_trial(scenario: Scenario, t: float) -> tuple[Box, ...]:
    """Bounded smooth shuttle with a known maximum excursion of 4.5 metres.

    Centre stays at x=18, y in [11.5,20.5], z=3.5; the complete fixture stays
    within the second room and never penetrates a wall, floor, or roof. The
    reference route is a static-feasibility certificate, not a time-dependent
    promise that the controller can pass this moving fixture.
    """
    if not math.isfinite(t):
        raise ValueError("time must be finite")
    phase = ((scenario.seed * 2654435761) & 0xffffffff) / 4294967296.0 * 2 * math.pi
    result = []
    amplitude = 4.5
    for box in scenario.obstacles:
        if box.velocity == (0.0, 0.0, 0.0):
            result.append(box)
            continue
        if box.id != "moving-inspection-fixture":
            raise ValueError(f"no motion contract for dynamic obstacle {box.id}")
        frequency = abs(box.velocity[1]) / amplitude
        theta = frequency * max(0.0, t) + phase
        displacement = amplitude * math.sin(theta)
        velocity = (0.0, amplitude * frequency * math.cos(theta), 0.0)
        result.append(Box(box.id,
                          (box.low[0], box.low[1] + displacement, box.low[2]),
                          (box.high[0], box.high[1] + displacement, box.high[2]), velocity))
    return tuple(result)


def _segment_box_distance(a: Vec3, b: Vec3, box: Box) -> float:
    """Independent exact segment-to-AABB Euclidean distance.

    Squared distance is piecewise quadratic in segment parameter u. Split at
    crossings of each box face, then evaluate the quadratic minimiser in each
    piece. This does not call simulator collision/planning geometry routines.
    """
    d = tuple(b[i] - a[i] for i in range(3))
    cuts = {0.0, 1.0}
    for i in range(3):
        if abs(d[i]) > 1e-15:
            for face in (box.low[i], box.high[i]):
                u = (face - a[i]) / d[i]
                if 0.0 < u < 1.0:
                    cuts.add(u)
    def squared(u: float) -> float:
        return sum(max(box.low[i] - (a[i] + d[i] * u), 0.0,
                       a[i] + d[i] * u - box.high[i]) ** 2 for i in range(3))
    ordered = sorted(cuts)
    best = min(squared(u) for u in ordered)
    for low, high in zip(ordered, ordered[1:]):
        mid = (low + high) / 2
        coefficient, linear = 0.0, 0.0
        for i in range(3):
            coordinate = a[i] + d[i] * mid
            face = box.low[i] if coordinate < box.low[i] else box.high[i] if coordinate > box.high[i] else None
            if face is not None:
                coefficient += d[i] ** 2
                linear += d[i] * (a[i] - face)
        if coefficient:
            u = min(high, max(low, -linear / coefficient))
            best = min(best, squared(u))
    return math.sqrt(best)


def reference_certificate(radius: float = 0.45) -> dict:
    """Check a complete ordered round trip independently of the controller.

    Besides sphere clearance, the certificate checks a enclosing sphere of
    radius sqrt(3)*radius. Passing this stronger condition implies clearance for
    the simulator's conservative axis-expanded-box collision convention too.
    """
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    segments = []
    enclosing_radius = math.sqrt(3.0) * radius
    for index, (start, end) in enumerate(zip(REFERENCE_ROUTE, REFERENCE_ROUTE[1:])):
        closest_distance, closest_id = min(
            (_segment_box_distance(start, end, box), box.id) for box in STATIC_OBSTACLES
        )
        boundary_clearance = min(min(p[i], BOUNDS[i] - p[i])
                                 for p in (start, end) for i in range(3))
        segments.append({
            "index": index, "start": start, "end": end,
            "length_m": math.dist(start, end), "nearest_obstacle": closest_id,
            "center_clearance_m": closest_distance,
            "body_clearance_m": closest_distance - radius,
            "conservative_cube_clearance_lower_bound_m": closest_distance - enclosing_radius,
            "boundary_body_clearance_m": boundary_clearance - radius,
            "clear": closest_distance > enclosing_radius and boundary_clearance > radius,
        })
    return {
        "course_id": COURSE_ID, "static_geometry_sha256": static_geometry_sha256(),
        "body_radius_m": radius, "all_segments_clear": all(s["clear"] for s in segments),
        "segment_count": len(segments), "route": REFERENCE_ROUTE,
        "length_m": sum(s["length_m"] for s in segments),
        "minimum_body_clearance_m": min(s["body_clearance_m"] for s in segments),
        "minimum_conservative_cube_clearance_lower_bound_m": min(s["conservative_cube_clearance_lower_bound_m"] for s in segments),
        "ideal_time_at_2_5_m_s": sum(s["length_m"] for s in segments) / 2.5,
        "ordered_inspection_targets": WAYPOINTS,
        "complete_round_trip": REFERENCE_ROUTE[0] == HOME == REFERENCE_ROUTE[-1]
                               and all(p in OUTBOUND_REFERENCE_ROUTE for p in WAYPOINTS),
        "segments": segments,
        "scope": "Static geometric feasibility only. The witness is not supplied to the controller and proves neither dynamic feasibility nor fault tolerance.",
    }


def validate_course() -> dict:
    """Small serialisable physical-validity report for benchmark manifests."""
    certificate = reference_certificate()
    return {
        "valid": certificate["all_segments_clear"] and certificate["complete_round_trip"],
        "static_geometry_sha256": static_geometry_sha256(),
        "reference_certificate": certificate,
        "partition_apertures_m": APERTURES,
        "partitions_span_bounds": True,
        "floor_roof_and_four_perimeter_walls_present": True,
        "dynamic_fixture_center_y_bounds_m": [11.5, 20.5],
        "dynamic_fixture_body_y_bounds_m": [10.5, 21.5],
    }


def course_manifest() -> dict:
    """Human/machine-readable course specification; contains no trial results."""
    return {
        "name": COURSE_ID, "course_id": COURSE_ID,
        "static_geometry_sha256": static_geometry_sha256(),
        "bounds": BOUNDS, "home": HOME, "waypoints": WAYPOINTS,
        "obstacles": [asdict(box) for box in STATIC_OBSTACLES],
        "apertures": APERTURES,
        "witness": reference_certificate(),
        "profiles": PROFILE_DEFINITIONS,
        "common_parameters": {
            "dt_seconds": 0.2, "deadline_seconds": 300.0,
            "normal_initial_battery_wh": 22.0,
            "fault_start_seconds": [15.0, 65.0],
            "fault_duration_seconds": [4.0, 12.0],
            "baseline_sensor_noise_rms_m": 0.02,
            "dynamic_detection_range_m": 6.0,
            "static_map_prior": True,
        },
        "randomised_conditions": ["bounded sensor error samples", "fault start and duration", "gust phase and direction", "fault magnitude", "moving fixture phase and speed", "low-battery initial charge"],
        "geometry_randomised": False,
        "reference_supplied_to_controller": False,
    }
