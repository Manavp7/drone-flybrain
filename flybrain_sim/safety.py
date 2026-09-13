"""One authority gate for simulated guidance, observation age and bounds.

Braking requests use the observer's bounded disturbance compensation when
available. This geometric gate is not a physical aircraft safety certificate.
"""
from __future__ import annotations

import math

from .contracts import Box, BrainOutput, Guidance, Observation, Vec3
from .geometry import (add, clamp_norm, mul, norm, point_box_distance,
                       segment_intersects_box, sub, unit, within_bounds)
from .planning import finite_vec, projected_boxes


class SafetyGate:
    def __init__(self, bounds: Vec3, radius: float = 0.45,
                 max_speed: float = 2.5, freshness: float = 0.8,
                 position_uncertainty: float = 0.0) -> None:
        self.bounds = bounds
        self.radius = radius
        self.max_speed = max_speed
        self.freshness = freshness
        self.guard_margin = 0.12 + max(0.0, position_uncertainty)
        self.previous_now: float | None = None
        self.previous_capture: float | None = None
        self.braking_velocity: Vec3 = (0.0, 0.0, 0.0)
        self.rejections = 0
        self.brain_rejections = 0
        self.stale_observations = 0

    def validate_observation(self, observation: Observation, now: float) -> str:
        if not math.isfinite(now):
            return "nonfinite_clock"
        prior_now = self.previous_now
        self.previous_now = now
        if prior_now is not None and now < prior_now - 1e-9:
            self.previous_capture = None
            return "clock_reset"
        if not observation.valid:
            return "invalid_localization"
        if not finite_vec(observation.position) or not finite_vec(observation.velocity):
            return "nonfinite_localization"
        if not math.isfinite(observation.battery_wh) or observation.battery_wh < 0.0:
            return "invalid_battery"
        if not (math.isfinite(observation.capture_time)
                and math.isfinite(observation.receive_time)):
            return "nonfinite_timestamp"
        if observation.capture_time > now + 0.05 or observation.receive_time > now + 0.05:
            return "future_timestamp"
        if observation.capture_time > observation.receive_time + 0.05:
            return "timestamp_order"
        if now - observation.capture_time > self.freshness:
            self.stale_observations += 1
            return "stale_localization"
        if now - observation.receive_time > self.freshness:
            self.stale_observations += 1
            return "stale_delivery"
        if (self.previous_capture is not None
                and observation.capture_time < self.previous_capture - 0.05):
            self.previous_capture = observation.capture_time
            return "capture_clock_reset"
        self.previous_capture = observation.capture_time
        if not within_bounds(observation.position, self.bounds, self.radius):
            return "localization_outside_bounds"
        for box in observation.obstacles:
            if (not finite_vec(box.low) or not finite_vec(box.high)
                    or not finite_vec(box.velocity)
                    or any(box.low[i] > box.high[i] for i in range(3))):
                return "invalid_obstacle_geometry"
        return ""

    def brain_scale(self, output: BrainOutput) -> float:
        try:
            valid = (output.healthy and math.isfinite(output.speed_scale)
                     and math.isfinite(output.caution)
                     and all(math.isfinite(v) for v in output.diagnostics.values()))
        except (AttributeError, TypeError, ValueError):
            valid = False
        if not valid:
            self.brain_rejections += 1
            return 1.0
        # An experimental module may only reduce bounded route speed.
        return max(0.55, min(1.0, output.speed_scale)) * (1.0 - 0.15 *
                         max(0.0, min(1.0, output.caution)))

    def brake(self, now: float, reason: str, mode: str = "RECOVERY") -> Guidance:
        self.rejections += 1
        return self.zero(now, mode, reason)

    def zero(self, now: float, mode: str, reason: str) -> Guidance:
        """Issue compensated braking through the same authority.

        The historical name is retained for callers; with a learned disturbance
        this command need not be the numeric zero vector.
        """
        expiry = now + 0.3 if math.isfinite(now) else 0.0
        return Guidance(clamp_norm(self.braking_velocity, 0.55), mode, reason, expiry)

    def _motion_clear(self, command: Vec3, observation: Observation, now: float,
                      obstacles: tuple[Box, ...], horizon: float = 2.4,
                      recover_static: bool = False) -> tuple[bool, Vec3]:
        """Finite point-mass prediction against time-indexed observed geometry.

        Obstacles use constant observed velocity, not a hidden future trajectory.
        Relative segment checks permit moving away from a forecast envelope that
        contains the current point even though the obstacle is not there yet.
        """
        position, velocity = observation.position, observation.velocity
        age = max(0.0, now - observation.capture_time)
        disturbance = mul(self.braking_velocity, -1.0 / 0.55)
        # Recovery may leave a violated discretionary margin, but never waive
        # the body collision check. Choose one initial outward face per box and
        # require nondecreasing separation on every predicted segment.
        recovery_faces: dict[str, tuple[int, float, float]] = {}
        if recover_static:
            for box in obstacles:
                if box.velocity == (0.0, 0.0, 0.0) and segment_intersects_box(
                        position, position, box, self.radius + self.guard_margin):
                    faces = [(position[i] - box.high[i], i, 1.0, box.high[i]) for i in range(3)]
                    faces += [(box.low[i] - position[i], i, -1.0, box.low[i]) for i in range(3)]
                    _, axis, sign, face = max(faces)
                    recovery_faces[box.id] = axis, sign, face
        elapsed = 0.0
        while elapsed < horizon - 1e-9:
            dt = min(0.2, horizon - elapsed)
            acceleration = clamp_norm(add(mul(sub(command, velocity), 1.0 / 0.55),
                                          disturbance), 3.0)
            next_velocity = clamp_norm(add(velocity, mul(acceleration, dt)), 5.0)
            next_position = add(position, mul(add(velocity, next_velocity), 0.5 * dt))
            if not within_bounds(next_position, self.bounds, self.radius + self.guard_margin):
                return False, next_position
            for box in obstacles:
                # Transform both vehicle endpoints into this moving box's frame.
                relative_start = sub(position, mul(box.velocity, age + elapsed))
                relative_end = sub(next_position, mul(box.velocity, age + elapsed + dt))
                radius = self.radius + self.guard_margin
                if box.id in recovery_faces:
                    axis, sign, face = recovery_faces[box.id]
                    before = sign * (relative_start[axis] - face)
                    after = sign * (relative_end[axis] - face)
                    if after < before - 1e-9:
                        return False, next_position
                    radius = self.radius
                if segment_intersects_box(relative_start, relative_end, box, radius):
                    return False, next_position
            position, velocity = next_position, next_velocity
            elapsed += dt
        return True, position

    def escape_static(self, observation: Observation, now: float,
                      obstacles: tuple[Box, ...]) -> Guidance | None:
        """Leave a violated static guard margin without accepting penetration.

        The accepted estimated state must be outside every physical body-expanded
        box. This is a bounded recovery search, not proof of the unknown true
        position or a substitute for a localization uncertainty bound.
        """
        trapped = tuple(box for box in obstacles
                        if box.velocity == (0.0, 0.0, 0.0)
                        and segment_intersects_box(observation.position, observation.position,
                                                   box, self.radius + self.guard_margin))
        if not trapped:
            return None
        best: tuple[float, Vec3] | None = None
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                for z in (-1.0, 0.0, 1.0):
                    if x == y == z == 0.0:
                        continue
                    command = clamp_norm(add(mul(unit((x, y, z)), 1.2),
                                              self.braking_velocity), self.max_speed)
                    clear, endpoint = self._motion_clear(command, observation, now, obstacles,
                                                        horizon=1.8, recover_static=True)
                    if not clear:
                        continue
                    margin = self.radius + self.guard_margin
                    if any(segment_intersects_box(endpoint, endpoint, box, margin) for box in trapped):
                        continue
                    score = min(point_box_distance(endpoint, box) for box in trapped)
                    if best is None or score > best[0]:
                        best = score, command
        if best is None:
            return None
        self.rejections += 1
        return Guidance(best[1], "RECOVERY", "static_margin_escape", now + 0.3)

    def escape_moving(self, observation: Observation, now: float,
                      obstacles: tuple[Box, ...]) -> Guidance | None:
        """Search bounded retreat commands when braking would be hit by motion.

        Called only with accepted localization when normal route guidance blocks.
        Failure to find a candidate is explicit; the search makes no completeness
        or safety guarantee under unobserved motion or model mismatch.
        """
        moving = tuple(box for box in obstacles if norm(box.velocity) > 1e-6)
        if not moving:
            return None
        safe_to_brake, _ = self._motion_clear(self.braking_velocity, observation,
                                              now, moving)
        if safe_to_brake:
            return None
        best: tuple[float, Vec3] | None = None
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                for z in (-1.0, 0.0, 1.0):
                    if x == y == z == 0.0:
                        continue
                    command = clamp_norm(add(mul(unit((x, y, z)), self.max_speed),
                                              self.braking_velocity), self.max_speed)
                    clear, endpoint = self._motion_clear(command, observation, now, obstacles)
                    if not clear:
                        continue
                    future = projected_boxes(obstacles,
                                             max(0.0, now - observation.capture_time) + 2.4,
                                             horizon=0.0)
                    clearance = min((point_box_distance(endpoint, box) for box in future),
                                    default=0.0)
                    # Prefer separation without assuming that shortest mission
                    # progress is the correct response to an approaching object.
                    score = clearance - 0.05 * norm(sub(command, observation.velocity))
                    if best is None or score > best[0]:
                        best = score, command
        if best is None:
            return None
        self.rejections += 1
        return Guidance(best[1], "RECOVERY", "moving_obstacle_escape", now + 0.3)

    def escape_memory(self, observation: Observation, now: float,
                      obstacles: tuple[Box, ...],
                      hazards: tuple[Box, ...]) -> Guidance | None:
        """Leave a remembered motion corridor before choosing a blind hold.

        `hazards` are uncertain historical envelopes, not claimed current body
        positions. A candidate must increase separation along an outward face
        and pass the normal momentum-aware static/fresh-geometry check. We do
        not pretend to prove separation from an unseen accelerating obstacle.
        """
        stop = add(observation.position, mul(observation.velocity, 0.55))
        margin = self.radius + self.guard_margin + 0.35
        nearby = tuple(box for box in hazards
                       if segment_intersects_box(observation.position, stop, box, margin))
        if not nearby:
            return None
        faces = []
        for box in nearby:
            options = [(observation.position[i] - box.high[i], i, 1.0, box.high[i])
                       for i in range(3)]
            options += [(box.low[i] - observation.position[i], i, -1.0, box.low[i])
                        for i in range(3)]
            # The nearest exit face provides a consistent retreat direction;
            # choosing by Euclidean distance alone is flat inside an envelope.
            before, axis, sign, face = max(options)
            faces.append((before, axis, sign, face))
        best: tuple[float, Vec3] | None = None
        for x in (-1.0, 0.0, 1.0):
            for y in (-1.0, 0.0, 1.0):
                for z in (-1.0, 0.0, 1.0):
                    if x == y == z == 0.0:
                        continue
                    command = clamp_norm(add(mul(unit((x, y, z)), 1.2),
                                              self.braking_velocity), self.max_speed)
                    clear, endpoint = self._motion_clear(command, observation, now,
                                                        obstacles, horizon=2.4)
                    if not clear:
                        continue
                    gains = [sign * (endpoint[axis] - face) - before
                             for before, axis, sign, face in faces]
                    if min(gains) <= 0.15:
                        continue
                    score = min(gains) - 0.05 * norm(sub(command, observation.velocity))
                    if best is None or score > best[0]:
                        best = score, command
        if best is None:
            return None
        self.rejections += 1
        return Guidance(best[1], "RECOVERY", "remembered_obstacle_retreat", now + 0.3)

    def authorize(self, velocity: Vec3, observation: Observation, now: float,
                  mode: str, reason: str, obstacles=()) -> Guidance:
        if not finite_vec(velocity):
            return self.brake(now, "nonfinite_guidance")
        velocity = clamp_norm(velocity, self.max_speed)
        # Near-horizon guard includes measured momentum before commanded motion.
        projected = add(observation.position,
                        add(mul(observation.velocity, 0.30), mul(velocity, 0.65)))
        if not within_bounds(projected, self.bounds, self.radius + self.guard_margin):
            return self.brake(now, "predicted_boundary", "BLOCKED")
        boxes = projected_boxes(tuple(obstacles), now - observation.capture_time, 0.95)
        if any(segment_intersects_box(observation.position, projected,
                                      box, self.radius + self.guard_margin) for box in boxes):
            return self.brake(now, "predicted_obstacle", "BLOCKED")
        return Guidance(velocity, mode, reason, now + 0.3)
