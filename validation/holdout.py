"""Seeded unseen enclosed halls; no controller state or route hints are used.

Only Scenario objects reach the controller. Witness routes are held separately
for independent geometric feasibility checks. The final 500 evaluation seeds
must not be flown until the controller and evaluation harness are frozen.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
import math
import random

from flybrain_sim.contracts import Box, Scenario, Vec3
from stress.course import PROFILES, PROFILE_DEFINITIONS, _segment_box_distance
from stress.course import make_trial as make_fault_trial

COURSE_ID = "unseen-enclosed-halls-v1"
SEED_START = 2200000
TRIALS = 500
SEED_END = SEED_START + TRIALS - 1
BODY_RADIUS_M = 0.45
PLANNING_MARGIN_M = 0.95
MOTION_AMPLITUDE_M = 4.5
PARAMETER_RANGES = {
    "room_lengths_m": [10.5, 13.5],
    "hall_width_m": [30.0, 38.0],
    "hall_height_m": [9.5, 12.0],
    "wall_thickness_m": 0.4,
    "partition_thickness_m": 0.8,
    "aperture_width_m": [4.2, 5.4],
    "aperture_height_m": [4.2, 4.8],
    "gate_lateral_inset_m": [6.2, 8.2],
    "gate_vertical_inset_m": [3.15, 3.65],
    "target_room_fraction": [0.46, 0.54],
    "target_lateral_jitter_m": [-0.75, 0.75],
    "target_vertical_jitter_m": [-0.25, 0.25],
    "horizontal_translation_m": [0.0, 3.0],
    "horizontal_orientations": 8,
    "witness_gate_approach_distance_m": 2.8,
}


def _sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _check_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")


def _transform_point(point: Vec3, layout: dict) -> Vec3:
    x, y, z = point
    if layout["reflect_x"]:
        x = layout["canonical_bounds"][0] - x
    if layout["reflect_y"]:
        y = layout["canonical_bounds"][1] - y
    if layout["swap_xy"]:
        x, y = y, x
    return (x + layout["translation"][0], y + layout["translation"][1], z)


def _transform_vector(vector: Vec3, layout: dict) -> Vec3:
    x, y, z = vector
    if layout["reflect_x"]:
        x = -x
    if layout["reflect_y"]:
        y = -y
    if layout["swap_xy"]:
        x, y = y, x
    return (x, y, z)


def _transform_box(box: Box, layout: dict) -> Box:
    a, b = _transform_point(box.low, layout), _transform_point(box.high, layout)
    return Box(box.id, tuple(min(a[i], b[i]) for i in range(3)),
               tuple(max(a[i], b[i]) for i in range(3)),
               _transform_vector(box.velocity, layout))


@lru_cache(maxsize=1024)
def _layout(seed: int) -> dict:
    _check_seed(seed)
    rng = random.Random(seed ^ 0x7D184AC9)
    room_lengths = tuple(rng.uniform(10.5, 13.5) for _ in range(4))
    length = sum(room_lengths)
    width, height = rng.uniform(30.0, 38.0), rng.uniform(9.5, 12.0)
    layout = {
        "seed": seed, "room_lengths": room_lengths,
        "canonical_bounds": (length, width, height),
        "swap_xy": bool(rng.getrandbits(1)),
        "reflect_x": bool(rng.getrandbits(1)),
        "reflect_y": bool(rng.getrandbits(1)),
        "translation": (rng.uniform(0.0, 3.0), rng.uniform(0.0, 3.0), 0.0),
    }
    gates = []
    for index in range(3):
        y_inset = rng.uniform(6.2, 8.2)
        z_inset = rng.uniform(3.15, 3.65)
        gates.append({"id": f"gate-{index + 1}",
                      "x": sum(room_lengths[:index + 1]),
                      "y": y_inset if index % 2 == 0 else width - y_inset,
                      "z": z_inset if index % 2 == 0 else height - z_inset,
                      "width": rng.uniform(4.2, 5.4),
                      "height": rng.uniform(4.2, 4.8)})
    home = (room_lengths[0] * rng.uniform(0.46, 0.54), gates[0]["y"], gates[0]["z"])
    targets = []
    for index in range(3):
        next_gate = gates[index + 1] if index < 2 else {
            "y": width - rng.uniform(6.2, 8.2), "z": height - rng.uniform(3.15, 3.65)}
        targets.append((gates[index]["x"] + room_lengths[index + 1] * rng.uniform(0.46, 0.54),
                        next_gate["y"] + rng.uniform(-0.75, 0.75),
                        next_gate["z"] + rng.uniform(-0.25, 0.25)))
    boxes = [
        Box("floor", (0.0, 0.0, 0.0), (length, width, 0.4)),
        Box("roof", (0.0, 0.0, height - 0.4), (length, width, height)),
        Box("west-wall", (0.0, 0.0, 0.0), (0.4, width, height)),
        Box("east-wall", (length - 0.4, 0.0, 0.0), (length, width, height)),
        Box("south-wall", (0.0, 0.0, 0.0), (length, 0.4, height)),
        Box("north-wall", (0.0, width - 0.4, 0.0), (length, width, height)),
    ]
    outbound = [home]
    for index, gate in enumerate(gates):
        x, y, z = (gate[key] for key in ("x", "y", "z"))
        y0, y1 = y - gate["width"] / 2, y + gate["width"] / 2
        z0, z1 = z - gate["height"] / 2, z + gate["height"] / 2
        boxes.extend((
            Box(f"{gate['id']}-sill", (x - 0.4, 0.0, 0.0), (x + 0.4, width, z0)),
            Box(f"{gate['id']}-lintel", (x - 0.4, 0.0, z1), (x + 0.4, width, height)),
            Box(f"{gate['id']}-south-jamb", (x - 0.4, 0.0, z0), (x + 0.4, y0, z1)),
            Box(f"{gate['id']}-north-jamb", (x - 0.4, y1, z0), (x + 0.4, width, z1)),
        ))
        outbound.extend(((x - 2.8, y, z), (x + 2.8, y, z), targets[index]))
    transformed_bounds = ((width, length, height) if layout["swap_xy"] else (length, width, height))
    # Translate the complete enclosed building inside a larger world box.
    layout.update({
        "bounds": tuple(transformed_bounds[i] + 2 * layout["translation"][i] for i in range(3)),
        "home": _transform_point(home, layout),
        "waypoints": tuple(_transform_point(p, layout) for p in targets),
        "obstacles": tuple(_transform_box(box, layout) for box in boxes),
        "canonical_apertures": tuple(gates),
        "apertures": tuple({**gate, "center": _transform_point((gate["x"], gate["y"], gate["z"]), layout),
                            "partition_axis": "y" if layout["swap_xy"] else "x"} for gate in gates),
        "outbound_witness": tuple(_transform_point(p, layout) for p in outbound),
    })
    layout["witness"] = layout["outbound_witness"] + tuple(reversed(layout["outbound_witness"][:-1]))
    return layout


def make_trial(seed: int, profile: str = "nominal") -> Scenario:
    """Return only the test scenario, never the independent witness route."""
    base = make_fault_trial(seed, profile)
    layout = _layout(seed)
    obstacles = layout["obstacles"]
    if profile in ("moving_obstacle", "compound"):
        original = next(box for box in base.obstacles if box.velocity != (0.0, 0.0, 0.0))
        x = sum(layout["room_lengths"][:1]) + layout["room_lengths"][1] / 2
        width, height = layout["canonical_bounds"][1:]
        fixture = Box("moving-inspection-fixture", (x - 1.0, width / 2 - 1.0, 0.4),
                      (x + 1.0, width / 2 + 1.0, min(6.6, height - 1.0)), original.velocity)
        obstacles += (_transform_box(fixture, layout),)
    return replace(base, bounds=layout["bounds"], home=layout["home"], waypoints=layout["waypoints"],
                   obstacles=obstacles, wind=_transform_vector(base.wind, layout), max_time=300.0)


def obstacles_at_trial(scenario: Scenario, t: float) -> tuple[Box, ...]:
    """Translate the fixture along its encoded unit velocity direction.

    For peak-speed vector v, u=v/|v|, phase=((seed*2654435761)&0xffffffff)/2**32*2pi,
    displacement = u*4.5*sin(|v|/4.5*max(t,0)+phase), and instantaneous velocity
    = v*cos(|v|/4.5*max(t,0)+phase). All other boxes are unchanged.
    """
    if not math.isfinite(t):
        raise ValueError("time must be finite")
    phase = ((scenario.seed * 2654435761) & 0xffffffff) / 4294967296.0 * 2 * math.pi
    result = []
    for box in scenario.obstacles:
        speed = math.sqrt(sum(v * v for v in box.velocity))
        if speed == 0.0:
            result.append(box)
            continue
        if box.id != "moving-inspection-fixture":
            raise ValueError(f"no motion contract for dynamic obstacle {box.id}")
        theta = speed / MOTION_AMPLITUDE_M * max(0.0, t) + phase
        shift = tuple(v / speed * MOTION_AMPLITUDE_M * math.sin(theta) for v in box.velocity)
        result.append(Box(box.id, tuple(box.low[i] + shift[i] for i in range(3)),
                          tuple(box.high[i] + shift[i] for i in range(3)),
                          tuple(v * math.cos(theta) for v in box.velocity)))
    return tuple(result)


def make_blocked_trial(seed: int = 2199999, profile: str = "nominal") -> Scenario:
    """Separate impossible negative control: the second partition is sealed."""
    scenario = make_trial(seed, profile)
    layout = _layout(seed)
    gate = layout["canonical_apertures"][1]
    _, width, height = layout["canonical_bounds"]
    barrier = _transform_box(Box("sealed-second-gate", (gate["x"] - 0.4, 0.0, 0.0),
                                (gate["x"] + 0.4, width, height)), layout)
    return replace(scenario, obstacles=scenario.obstacles + (barrier,))


def _expanded_box_intersects_segment(start: Vec3, end: Vec3, box: Box, margin: float) -> bool:
    """Independent slab intersection, including touching, with margin per axis."""
    lower, upper = 0.0, 1.0
    for axis in range(3):
        delta = end[axis] - start[axis]
        low, high = box.low[axis] - margin, box.high[axis] + margin
        if abs(delta) < 1e-15:
            if start[axis] < low or start[axis] > high:
                return False
        else:
            a, b = (low - start[axis]) / delta, (high - start[axis]) / delta
            lower, upper = max(lower, min(a, b)), min(upper, max(a, b))
            if lower > upper:
                return False
    return True


def geometry_sha256(seed: int) -> str:
    layout = _layout(seed)
    return _sha({"bounds": layout["bounds"], "home": layout["home"], "waypoints": layout["waypoints"],
                 "obstacles": [asdict(box) for box in layout["obstacles"]]})


def apertures_for_seed(seed: int) -> tuple[dict, ...]:
    """World-space aperture centers/axes for independent gate-crossing audit."""
    return tuple(dict(gate) for gate in _layout(seed)["apertures"])


def course_certificate(seed: int, *, blocked: bool = False) -> dict:
    layout = _layout(seed)
    scenario = make_blocked_trial(seed) if blocked else make_trial(seed)
    segments = []
    for index, (start, end) in enumerate(zip(layout["witness"], layout["witness"][1:])):
        closest_distance, closest_id = min((_segment_box_distance(start, end, box), box.id)
                                           for box in scenario.obstacles)
        boundary = min(min(p[i], scenario.bounds[i] - p[i]) for p in (start, end) for i in range(3))
        body_clear = (not any(_expanded_box_intersects_segment(start, end, box, BODY_RADIUS_M)
                              for box in scenario.obstacles) and boundary > BODY_RADIUS_M)
        planning_clear = (not any(_expanded_box_intersects_segment(start, end, box, PLANNING_MARGIN_M)
                                  for box in scenario.obstacles) and boundary > PLANNING_MARGIN_M)
        segments.append({"index": index, "start": start, "end": end, "length_m": math.dist(start, end),
                         "nearest_obstacle": closest_id, "center_clearance_m": closest_distance,
                         "body_clearance_m": closest_distance - BODY_RADIUS_M,
                         "body_expanded_box_clear": body_clear, "planning_expanded_box_clear": planning_clear})
    geometry_hash = (_sha({"bounds": scenario.bounds, "home": scenario.home,
                           "waypoints": scenario.waypoints, "obstacles": [asdict(box) for box in scenario.obstacles]})
                     if blocked else geometry_sha256(seed))
    return {
        "seed": seed, "course_id": COURSE_ID, "static_geometry_sha256": geometry_hash,
        "blocked_negative_control": blocked, "body_radius_m": BODY_RADIUS_M,
        "planning_margin_m": PLANNING_MARGIN_M, "route": layout["witness"],
        "all_segments_clear": all(segment["body_expanded_box_clear"] for segment in segments),
        "all_segments_planning_clear": all(segment["planning_expanded_box_clear"] for segment in segments),
        "complete_round_trip": layout["witness"][0] == layout["home"] == layout["witness"][-1]
                               and all(target in layout["outbound_witness"] for target in layout["waypoints"]),
        "minimum_body_clearance_m": min(segment["body_clearance_m"] for segment in segments),
        "length_m": sum(segment["length_m"] for segment in segments),
        "ideal_time_at_2_5_m_s": sum(segment["length_m"] for segment in segments) / 2.5,
        "segments": segments,
        "scope": "Static geometric feasibility only; no promise of fault tolerance, battery sufficiency, or controller success.",
    }


def course_manifest(seed: int) -> dict:
    layout = _layout(seed)
    return {"course_id": COURSE_ID, "seed": seed, "static_geometry_sha256": geometry_sha256(seed),
            "bounds": layout["bounds"], "home": layout["home"], "waypoints": layout["waypoints"],
            "obstacles": [asdict(box) for box in layout["obstacles"]], "apertures": layout["apertures"],
            "layout_parameters": {key: layout[key] for key in (
                "room_lengths", "canonical_bounds", "swap_xy", "reflect_x", "reflect_y", "translation", "canonical_apertures")},
            "witness": course_certificate(seed), "reference_supplied_to_controller": False}


@lru_cache(maxsize=4)
def validate_suite(seed_start: int = SEED_START, trials: int = TRIALS) -> dict:
    """Validate geometry only for every planned seed; never execute a controller."""
    if trials <= 0:
        raise ValueError("positive trial count required")
    certificates = [course_certificate(seed) for seed in range(seed_start, seed_start + trials)]
    hashes = [certificate["static_geometry_sha256"] for certificate in certificates]
    summaries = [{key: certificate[key] for key in (
        "seed", "static_geometry_sha256", "all_segments_clear", "all_segments_planning_clear",
        "complete_round_trip", "minimum_body_clearance_m", "length_m", "ideal_time_at_2_5_m_s")}
        for certificate in certificates]
    return {
        "valid": all(item["all_segments_clear"] and item["all_segments_planning_clear"]
                     and item["complete_round_trip"] for item in certificates),
        "course_id": COURSE_ID, "courses": trials, "unique_static_geometries": len(set(hashes)),
        "static_geometry_sha256": _sha(hashes), "all_courses_body_clear": all(item["all_segments_clear"] for item in certificates),
        "all_courses_planning_clear": all(item["all_segments_planning_clear"] for item in certificates),
        "minimum_body_clearance_m": min(item["minimum_body_clearance_m"] for item in certificates),
        "witness_length_range_m": [min(item["length_m"] for item in certificates), max(item["length_m"] for item in certificates)],
        "reference_certificate": {"all_segments_clear": all(item["all_segments_clear"] for item in certificates),
                                  "complete_round_trip": all(item["complete_round_trip"] for item in certificates),
                                  "scope": f"Independent static witnesses for all {trials} seeds; never supplied to the controller."},
        "course_certificates": summaries, "controller_executed": False,
    }


def validation_manifest(seed_start: int = SEED_START, trials: int = TRIALS) -> dict:
    validation = validate_suite(seed_start, trials)
    return {
        "name": COURSE_ID, "course_id": COURSE_ID, "valid": validation["valid"],
        "static_geometry_sha256": validation["static_geometry_sha256"],
        "trials": trials, "seed_start": seed_start, "seed_end": seed_start + trials - 1,
        "trials_per_profile": trials // len(PROFILES) if trials % len(PROFILES) == 0 else None,
        "profiles": PROFILE_DEFINITIONS,
        "trial_assignments": [{"trial_id": i + 1, "seed": seed_start + i, "profile": PROFILES[i % len(PROFILES)]}
                              for i in range(trials)],
        "parameter_ranges": PARAMETER_RANGES, "geometry_randomised": True,
        "reference_supplied_to_controller": False,
        "common_parameters": {"dt_seconds": 0.2, "deadline_seconds": 300.0,
                              "normal_initial_battery_wh": 22.0, "dynamic_detection_range_m": 6.0,
                              "static_map_prior": True, "dynamic_motion_amplitude_m": MOTION_AMPLITUDE_M},
        "validation": validation, "negative_control": {"seed": 2199999, "sealed_partition": 2,
                                                         "included_in_500_trials": False},
        "preregistration": "Use all 500 trials only after controller and harness freeze; no outcome-based selection or tuning.",
        "scope": "New procedural layouts from the enclosed partition-hall family, not unseen environment families or real flight.",
    }
