"""Closed-loop execution and independent simulator-truth scoring.

Truth is used only for physics, collision checks, scoring and visualization.
The autonomy controller receives the declared geometric Observation interface.
"""
from __future__ import annotations

from dataclasses import asdict
import math
import time

from .autonomy import Autonomy
from .brain import make_brain
from .contracts import Box, EpisodeResult, Scenario, VehicleState
from .geometry import clearance, distance, norm, obstacles_at, swept_collision, within_bounds
from .physics import step
from .sensors import SensorModel

RADIUS = 0.45


class TruthInspectionTracker:
    """Elapsed-time dwell between qualifying samples; no substep guarantee."""

    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.started = [None] * len(waypoints)
        self.completed: set[int] = set()

    def update(self, position, velocity, now: float) -> None:
        for i, waypoint in enumerate(self.waypoints):
            if distance(position, waypoint) <= 1.0 and norm(velocity) <= 1.2:
                if self.started[i] is None:
                    self.started[i] = now
                if now - self.started[i] >= 0.6 - 1e-9:
                    self.completed.add(i)
            else:
                self.started[i] = None


def motion_envelopes(before: tuple[Box, ...], after: tuple[Box, ...]) -> tuple[Box, ...]:
    """Conservative swept AABBs. Extra 1cm covers substep sine-path curvature."""
    current = {b.id: b for b in after}
    result = []
    for b in before:
        c = current.get(b.id, b)
        moving = b.low != c.low or b.high != c.high
        extra = 0.01 if moving else 0.0
        result.append(Box(b.id,
                          tuple(min(x, y) - extra for x, y in zip(b.low, c.low)),
                          tuple(max(x, y) + extra for x, y in zip(b.high, c.high))))
    return tuple(result)


def run_episode(scenario: Scenario, variant: str = "baseline", record: bool = False) -> EpisodeResult:
    start_wall = time.perf_counter()
    if scenario.dt <= 0 or scenario.max_time <= 0:
        raise ValueError("Positive timestep and duration required")
    controller = Autonomy(scenario)
    brain = make_brain(variant)
    sensors = SensorModel(scenario)
    state = VehicleState(0.0, scenario.home, (0.0, 0.0, 0.0), scenario.initial_battery_wh)
    collided = violated = False
    min_clearance = math.inf
    travelled = 0.0
    trajectory: list[dict] = []
    events: list[dict] = []
    last_mode = ""
    last_fault = ""
    inspection = TruthInspectionTracker(scenario.waypoints)
    outcome = "timeout"

    for tick in range(math.ceil(scenario.max_time / scenario.dt)):
        observation = sensors.observe(state)
        neural = brain.update(observation, state.time)
        command = controller.update(observation, state.time, neural)
        if command.mode != last_mode:
            events.append({"t": round(state.time, 3), "type": command.mode, "reason": command.reason})
            last_mode = command.mode
        if observation.fault != last_fault:
            if observation.fault or last_fault:
                events.append({"t": round(state.time, 3), "type": "sensor_fault" if observation.fault else "sensor_restored",
                               "reason": observation.fault or "geometric observations restored"})
            last_fault = observation.fault

        current_obstacles = obstacles_at(scenario, state.time)
        body_clearance = clearance(state.position, current_obstacles) - RADIUS
        min_clearance = min(min_clearance, body_clearance)
        if record and (tick % 2 == 0 or command.mode in ("COMPLETE", "ABORTED_HOME")):
            trajectory.append({"t": round(state.time, 3), "p": list(state.position), "v": list(state.velocity),
                               "battery_wh": state.battery_wh, "mode": command.mode,
                               "brain_caution": neural.caution, "clearance_m": body_clearance,
                               "obstacles": [asdict(b) for b in current_obstacles]})

        # Independent actual-position dwell score; geometric stand-in for inspection.
        inspection.update(state.position, state.velocity, state.time)

        if controller.returned_home:
            actual_home = distance(state.position, scenario.home) <= 1.0 and norm(state.velocity) <= 0.6
            if not actual_home:
                outcome = "completion_verification_failed"
            elif controller.mission_complete and len(inspection.completed) == len(scenario.waypoints):
                outcome = "mission_complete"
            elif controller.mission_complete:
                outcome = "inspection_verification_failed"
            else:
                outcome = "aborted_returned_home"
            if outcome != "mission_complete":
                events.append({"t": round(state.time, 3), "type": outcome,
                               "reason": "independent simulator-truth terminal verification"})
            break

        next_state = step(state, command.velocity, scenario)
        next_obstacles = obstacles_at(scenario, next_state.time)
        envelopes = motion_envelopes(current_obstacles, next_obstacles)
        collided = swept_collision(state.position, next_state.position, envelopes, radius=RADIUS)
        violated = not within_bounds(next_state.position, scenario.bounds, radius=RADIUS)
        travelled += distance(state.position, next_state.position)
        state = next_state
        min_clearance = min(min_clearance, clearance(state.position, next_obstacles) - RADIUS)
        if collided or violated or state.battery_wh <= 0:
            outcome = "collision" if collided else "geofence_violation" if violated else "battery_depleted"
            events.append({"t": round(state.time, 3), "type": outcome, "reason": "independent simulator-truth check"})
            if record:
                trajectory.append({"t": round(state.time, 3), "p": list(state.position), "v": list(state.velocity),
                                   "battery_wh": state.battery_wh, "mode": outcome, "brain_caution": neural.caution,
                                   "clearance_m": min_clearance, "obstacles": [asdict(b) for b in next_obstacles]})
            break

    actual_home = distance(state.position, scenario.home) <= 1.0 and norm(state.velocity) <= 0.6
    return EpisodeResult(
        seed=scenario.seed, category=scenario.category, variant=variant, outcome=outcome,
        mission_complete=outcome == "mission_complete", returned_home=actual_home and not collided and not violated,
        collision=bool(collided), geofence_violation=bool(violated), waypoints_completed=len(inspection.completed),
        waypoints_total=len(scenario.waypoints), simulated_seconds=state.time, energy_wh=state.energy_used_wh,
        minimum_clearance_m=min_clearance if math.isfinite(min_clearance) else 0.0,
        distance_m=travelled, interventions=controller.interventions,
        stale_observations=controller.stale_observations, brain_rejections=controller.brain_rejections,
        wall_seconds=time.perf_counter() - start_wall, trajectory=trajectory, events=events)
