"""Observation-only civilian inspection mission for the reduced-order simulator."""
from __future__ import annotations

import math

from dataclasses import replace

from .contracts import Box, BrainOutput, Guidance, Observation, Scenario, Vec3
from .geometry import add, clamp_norm, distance, mul, norm, sub, unit
from .planning import VisibilityPlanner, projected_boxes
from .safety import SafetyGate


class ObstacleMemory:
    """Bounded history of accepted geometry, never a hidden-world map.

    A moving object leaving the sensor frame does not become absent. For fault
    recovery we retain its observed swept extent and grow it along axes with
    observed motion. This is a conservative *heuristic* for repeat motion, not
    a certified acceleration bound or permission to use rejected observations.
    """
    max_entries = 64
    retention_seconds = 60.0

    def __init__(self, static_ids: set[str]):
        self.static_ids = static_ids
        self.entries: dict[str, dict] = {}

    def accept(self, observation: Observation, now: float) -> None:
        self.prune(now)
        for box in observation.obstacles:
            if box.id in self.static_ids:
                continue
            previous = self.entries.get(box.id)
            low, high = box.low, box.high
            speeds = tuple(abs(v) for v in box.velocity)
            if previous is not None:
                low = tuple(min(low[i], previous['low'][i]) for i in range(3))
                high = tuple(max(high[i], previous['high'][i]) for i in range(3))
                speeds = tuple(max(speeds[i], previous['speeds'][i]) for i in range(3))
            self.entries[box.id] = {'box': box, 'capture': observation.capture_time,
                                    'accepted': now, 'low': low, 'high': high,
                                    'speeds': speeds}
        while len(self.entries) > self.max_entries:
            oldest = min(self.entries, key=lambda key: (self.entries[key]['accepted'], key))
            del self.entries[oldest]

    def prune(self, now: float) -> None:
        for key, entry in tuple(self.entries.items()):
            if now < entry['accepted'] or now - entry['accepted'] > self.retention_seconds:
                del self.entries[key]

    def recovery_envelopes(self, now: float) -> tuple[Box, ...]:
        self.prune(now)
        result = []
        for key, entry in sorted(self.entries.items()):
            age = max(0.0, now - entry['capture'])
            # Retain previous extent even after velocity reverses. Unknown
            # transverse acceleration remains outside this geometric model.
            padding = tuple(0.08 + speed * min(age + 2.4, 12.0)
                            for speed in entry['speeds'])
            result.append(Box(key, tuple(entry['low'][i] - padding[i] for i in range(3)),
                              tuple(entry['high'][i] + padding[i] for i in range(3))))
        return tuple(result)

    def fresh_boxes(self, now: float, freshness: float) -> tuple[Box, ...]:
        self.prune(now)
        boxes = []
        for entry in self.entries.values():
            age = now - entry['capture']
            if 0.0 <= age <= freshness:
                boxes.extend(projected_boxes((entry['box'],), age, horizon=0.0))
        return tuple(boxes)

    def snapshot(self, now: float) -> dict:
        return {'entries': len(self.entries),
                'ages_s': {key: max(0.0, now - entry['capture'])
                           for key, entry in sorted(self.entries.items())},
                'retention_seconds': self.retention_seconds,
                'max_entries': self.max_entries}


class ObservationTracker:
    """Small observer for this point-mass plant, driven only by accepted samples.

    This is not visual-inertial localization. During rejected samples it performs
    dead reckoning with a bounded disturbance estimate learned before the fault.
    The prediction can drift and cannot establish real localization integrity.
    """
    response = 0.55

    def __init__(self, noise: float):
        self.position: Vec3 | None = None
        self.velocity: Vec3 = (0.0, 0.0, 0.0)
        self.disturbance: Vec3 = (0.0, 0.0, 0.0)
        self.command: Vec3 = (0.0, 0.0, 0.0)
        self.now: float | None = None
        self.last_observation: Observation | None = None
        self.last_valid = -math.inf
        self.position_gain = 0.22 if noise > 0.1 else 0.75
        self.velocity_gain = 0.25 if noise > 0.1 else 0.75

    def update(self, observation: Observation | None, now: float) -> Observation | None:
        if not math.isfinite(now):
            return None
        dt = 0.0 if self.now is None else now - self.now
        if dt < 0.0 or dt > 1.0:
            self.position = None
            self.disturbance = (0.0, 0.0, 0.0)
        if self.position is not None and 0.0 < dt <= 1.0:
            acceleration = clamp_norm(add(mul(sub(self.command, self.velocity),
                                             1.0 / self.response), self.disturbance), 3.0)
            next_velocity = add(self.velocity, mul(acceleration, dt))
            self.position = add(self.position, mul(add(self.velocity, next_velocity), 0.5 * dt))
            self.velocity = next_velocity
        self.now = now
        if observation is not None:
            measured_position = add(observation.position,
                                    mul(observation.velocity, max(0.0, now - observation.capture_time)))
            if self.position is None or now - self.last_valid > 0.8:
                # A long open-loop prediction is not a trustworthy filter prior.
                # Reinitialize only from a gate-accepted observation.
                self.position, self.velocity = measured_position, observation.velocity
            else:
                residual = sub(observation.velocity, self.velocity)
                if 0.0 < dt <= 1.0 and now - self.last_valid <= 0.4:
                    self.disturbance = clamp_norm(add(self.disturbance,
                                                       mul(residual, 0.025 / dt)), 1.0)
                self.velocity = add(self.velocity, mul(residual, self.velocity_gain))
                self.position = add(self.position, mul(sub(measured_position, self.position),
                                                       self.position_gain))
            self.last_valid = now
            self.last_observation = observation
        if self.position is None or self.last_observation is None:
            return None
        # Predict geometry from its original capture too. Relabelling old boxes
        # with `now` without this projection would erase their known age.
        boxes_now = projected_boxes(self.last_observation.obstacles,
                                    max(0.0, now - self.last_observation.capture_time),
                                    horizon=0.0)
        return replace(self.last_observation, position=self.position, velocity=self.velocity,
                       obstacles=boxes_now, capture_time=now, receive_time=now,
                       valid=observation is not None,
                       fault="" if observation is not None else "dead_reckoning")

    @property
    def compensation(self) -> Vec3:
        return mul(self.disturbance, self.response)


class Autonomy:
    def __init__(self, scenario: Scenario) -> None:
        self.scenario = scenario
        # Per-axis error bound for expanded boxes, never hidden simulator state.
        # This is not the Euclidean norm bound of the full noise vector.
        self._uncertainty = math.sqrt(3.0) * scenario.sensor_noise
        self.planner = VisibilityPlanner(scenario, margin=0.5 + self._uncertainty)
        self.gate = SafetyGate(scenario.bounds, position_uncertainty=self._uncertainty)
        self.completed = 0
        self.returned_home = False
        self.mission_complete = False
        self.interventions = 0
        self.mode = "INSPECT"
        self.abort_reason = ""
        self.brain_rejections = 0
        self.stale_observations = 0
        self._route: list[Vec3] = []
        self._target: Vec3 | None = None
        self._last_plan = -math.inf
        self._dwell_started: float | None = None
        self._blocked_started: float | None = None
        self._previous_fault = ""
        self._returning = not scenario.waypoints
        self.tracker = ObservationTracker(scenario.sensor_noise)
        self._fault_hold: Vec3 | None = None
        self._reacquiring = False
        self._settle_target: Vec3 | None = None
        self._settle_started: float | None = None
        self.obstacle_memory = ObstacleMemory({box.id for box in self.planner.static})
        self._reserve_position: Vec3 | None = None
        self._reserve_length: float | None = None
        self._reserve_updated = -math.inf
        self._reserve_success_time: float | None = None

    def _return_distance(self, position: Vec3, now: float) -> float:
        """Cached map route length; a failed search never means zero reserve."""
        if now - self._reserve_updated >= 5.0 or self._reserve_position is None:
            route = self.planner.plan(position, self.scenario.home)
            self._reserve_updated = now
            if route is not None:
                points = [position] + list(route)
                self._reserve_length = sum(distance(a, b) for a, b in zip(points, points[1:]))
                self._reserve_position = position
                self._reserve_success_time = now
        if (self._reserve_length is None or self._reserve_position is None
                or self._reserve_success_time is None):
            return math.inf
        # A straight connector to the cached origin can cut through a wall.
        # Allow retracing motion at the declared point-mass plant's 5 m/s bound
        # since the last successful plan instead. Failed attempts do not reset
        # this age. The energy conversion remains a simulation reserve heuristic.
        return self._reserve_length + 5.0 * max(0.0, now - self._reserve_success_time)

    def _finish(self, guidance: Guidance) -> Guidance:
        self.tracker.command = guidance.velocity
        self.mode = guidance.mode
        self.brain_rejections = self.gate.brain_rejections
        self.stale_observations = self.gate.stale_observations
        fault = guidance.reason if guidance.mode in ("RECOVERY", "BLOCKED") else ""
        if fault and fault != self._previous_fault:
            self.interventions += 1
        self._previous_fault = fault
        return guidance

    def _begin_return(self, reason: str = "") -> None:
        self._returning = True
        self._route = []
        self._target = None
        self._dwell_started = None
        self._last_plan = -math.inf
        if reason and not self.abort_reason:
            self.abort_reason = reason
            self.interventions += 1

    def update(self, observation: Observation, now: float,
               brain_output: BrainOutput) -> Guidance:
        fault = self.gate.validate_observation(observation, now)
        if fault:
            self._dwell_started = None
            if self.tracker.last_observation is not None and now - self.tracker.last_valid > self.gate.freshness:
                self._reacquiring = True
            self._settle_target = None
            self._settle_started = None
            self._route = []
            self._last_plan = -math.inf
            predicted = self.tracker.update(None, now)
            if predicted is None:
                return self._finish(self.gate.brake(now, fault))
            self.gate.braking_velocity = mul(self.tracker.compensation, -1.0)
            remembered = self.obstacle_memory.recovery_envelopes(now)
            fresh = self.obstacle_memory.fresh_boxes(now, self.gate.freshness)
            recovery_boxes = self.planner.static + fresh
            retreat = self.gate.escape_memory(predicted, now, recovery_boxes, remembered)
            if retreat is not None:
                self._fault_hold = None
                return self._finish(retreat)
            if self._fault_hold is None:
                self._fault_hold = add(predicted.position, mul(predicted.velocity, 0.35))
            desired = sub(sub(mul(sub(self._fault_hold, predicted.position), 1.2),
                              mul(predicted.velocity, 0.3)), self.tracker.compensation)
            # No rejected observation enters the estimator or obstacle map.
            command = self.gate.authorize(desired, predicted, now, "RECOVERY", fault,
                                          recovery_boxes)
            return self._finish(command)
        if (self.tracker.last_observation is not None
                and now - self.tracker.last_valid > self.gate.freshness):
            self._reacquiring = True
        self.obstacle_memory.accept(observation, now)
        observation = self.tracker.update(observation, now)
        assert observation is not None
        self._fault_hold = None
        self.gate.braking_velocity = mul(self.tracker.compensation, -1.0)
        scale = self.gate.brain_scale(brain_output)
        if self.returned_home:
            mode = "COMPLETE" if self.mission_complete else "ABORTED_HOME"
            return self._finish(self.gate.zero(now, mode, "terminal_mission"))

        # Deliberately approximate simulation reserve, not a physical flight rule.
        return_seconds = self._return_distance(observation.position, now) / 1.75
        reserve_wh = 1.0 + return_seconds * (250.0 / 3600.0)
        if not self._returning and not math.isfinite(return_seconds):
            self._begin_return("return_route_unavailable")
        elif not self._returning and observation.battery_wh <= reserve_wh:
            self._begin_return("low_energy_reserve")
        if (not self._returning
                and self.scenario.max_time - now < return_seconds * 1.6 + 10.0):
            self._begin_return("mission_time_reserve")

        goal = self.scenario.home if self._returning else self.scenario.waypoints[self.completed]
        mode = "RETURN" if self._returning else "INSPECT"
        if goal != self._target:
            self._target, self._route = goal, []
            self._dwell_started = None
            self._last_plan = -math.inf

        known = {box.id: box for box in self.planner.static}
        known.update({box.id: box for box in observation.obstacles})
        boxes = tuple(known.values())
        predicted = projected_boxes(boxes, now - observation.capture_time, 0.8)

        if self._reacquiring:
            self._dwell_started = None
            self._route = []
            escape = (self.gate.escape_static(observation, now, boxes)
                      or self.gate.escape_moving(observation, now, boxes))
            if escape is not None:
                self._settle_target = None
                self._settle_started = None
                return self._finish(escape)
            if self._settle_target is None:
                self._settle_target = add(observation.position, mul(observation.velocity, 0.35))
                self._settle_started = now
            desired = sub(sub(mul(sub(self._settle_target, observation.position), 1.2),
                              mul(observation.velocity, 0.3)), self.tracker.compensation)
            command = self.gate.authorize(desired, observation, now, "RECOVERY",
                                          "localization_reacquisition", boxes)
            settled = (self._settle_started is not None and now - self._settle_started >= 1.0
                       and distance(observation.position, self._settle_target) <= 0.3
                       and norm(observation.velocity) <= 0.3)
            if command.mode == "BLOCKED":
                self._settle_started = now
            elif settled:
                self._reacquiring = False
                self._settle_target = None
                self._settle_started = None
                self._last_plan = -math.inf
            # No inspection or terminal progress is credited in reacquisition.
            return self._finish(command)

        # Arrival hysteresis prevents ordinary measurement jitter repeatedly
        # deleting otherwise continuous inspection dwell. Maximum tolerance is
        # explicit and belongs to this abstract point-inspection task.
        arrival_radius = 0.45
        if self._dwell_started is not None:
            arrival_radius += 0.10
        if distance(observation.position, goal) <= arrival_radius and norm(observation.velocity) < (0.20 if self._returning else 0.40):
            if self._dwell_started is None:
                self._dwell_started = now
            if now - self._dwell_started >= 0.8:
                self._dwell_started = None
                if self._returning:
                    self.returned_home = True
                    self.mission_complete = (self.completed == len(self.scenario.waypoints)
                                             and not self.abort_reason)
                    mode = "COMPLETE" if self.mission_complete else "ABORTED_HOME"
                    return self._finish(self.gate.zero(now, mode, "home_dwell_complete"))
                self.completed += 1
                self._target, self._route = None, []
                if self.completed == len(self.scenario.waypoints):
                    self._begin_return()
                return self._finish(self.gate.zero(now, "INSPECT", "inspection_dwell_complete"))
        else:
            self._dwell_started = None

        # Discard reached corners; finite-route visibility checks react to moving
        # geometry without reconstructing the graph on every simulation step.
        while len(self._route) > 1 and distance(observation.position, self._route[0]) < 0.4:
            self._route.pop(0)
        extra = tuple(box for box in predicted
                      if box.id not in {item.id for item in self.planner.static})
        if self._route and not self.planner.segment_clear(
                observation.position, self._route[0], extra,
                clearance=0.67 + self._uncertainty):
            self._route = []
        if not self._route and now - self._last_plan >= 0.6:
            self._last_plan = now
            self._route = self.planner.plan(observation.position, goal, predicted) or []
        if not self._route:
            escape = (self.gate.escape_static(observation, now, boxes)
                      or self.gate.escape_moving(observation, now, boxes))
            if escape is not None:
                self._dwell_started = None
                self._blocked_started = None
                return self._finish(escape)
            if self._blocked_started is None:
                self._blocked_started = now
            if now - self._blocked_started > 6.0 and not self._returning:
                self._begin_return("route_unavailable")
            return self._finish(self.gate.brake(now, "route_unavailable", "BLOCKED"))
        self._blocked_started = None

        direction = sub(self._route[0], observation.position)
        speed = min(self.gate.max_speed * scale, 1.2 * norm(direction))
        desired = sub(sub(mul(unit(direction), speed), mul(observation.velocity, 0.30)),
                      self.tracker.compensation)
        desired = clamp_norm(desired, self.gate.max_speed)
        guidance = self.gate.authorize(desired, observation, now, mode,
                                       "inspection_route" if mode == "INSPECT" else "home_route",
                                       boxes)
        if guidance.mode == "BLOCKED":
            self._route = []
            escape = (self.gate.escape_static(observation, now, boxes)
                      or self.gate.escape_moving(observation, now, boxes))
            if escape is not None:
                self._dwell_started = None
                self._blocked_started = None
                guidance = escape
        return self._finish(guidance)
