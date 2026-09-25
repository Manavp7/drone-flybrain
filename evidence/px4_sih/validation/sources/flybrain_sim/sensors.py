"""Geometric observations with explicit synthetic timing and noise faults.

Static fixtures are a declared prior map. Moving fixtures are observed only
inside sensor_range. No camera renderer or real state estimator is implied.
"""
from __future__ import annotations

from collections import deque
import math
from .contracts import Observation, Scenario, VehicleState
from .geometry import obstacles_at, point_box_distance


def _samples(seed: int, tick: int) -> tuple[float, ...]:
    """Stateless per-tick samples: skipped calls cannot change fault timing."""
    value = ((seed & 0xffffffff) ^ ((tick + 1) * 0x9e3779b9)) & 0xffffffff
    value ^= value >> 16
    value = (value * 0x85ebca6b) & 0xffffffff
    value ^= value >> 13
    values = []
    for _ in range(7):
        value = (1664525 * value + 1013904223) & 0xffffffff
        values.append(value / 4294967296.0)
    return tuple(values)


class SensorModel:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self._history: deque[Observation] = deque(maxlen=max(2, scenario.latency_steps + 1))
        self._clock_reset_emitted = False

    def observe(self, state: VehicleState) -> Observation:
        scenario = self.scenario
        tick = round(state.time / scenario.dt)
        samples = _samples(scenario.seed, tick)
        in_fault = (scenario.fault_start <= state.time <
                    scenario.fault_start + scenario.fault_duration)
        sigma = scenario.sensor_noise
        # Uniform bounded error has RMS sigma; this is not calibrated sensor noise.
        amplitude = sigma * math.sqrt(3.0)
        position = tuple(state.position[i] + amplitude * (2.0 * samples[i] - 1.0)
                         for i in range(3))
        velocity = tuple(state.velocity[i] + 0.5 * amplitude * (2.0 * samples[i + 3] - 1.0)
                         for i in range(3))
        visible = tuple(box for box in obstacles_at(scenario, state.time)
                        if box.id != "moving-inspection-fixture" or
                        point_box_distance(state.position, box) <= scenario.sensor_range)
        current = Observation(state.time, state.time, position, velocity, visible,
                              state.battery_wh)
        self._history.append(current)
        selected = self._history[0] if in_fault and scenario.latency_steps else current
        valid = True
        fault = "latency" if selected.capture_time < state.time - 1e-9 else ""
        capture = selected.capture_time
        if in_fault and scenario.category in ("clock_reset", "compound"):
            capture -= 12.0
            fault = "clock_reset" if not self._clock_reset_emitted else "clock_offset"
            self._clock_reset_emitted = True
        if in_fault and samples[6] < scenario.dropout_probability:
            valid, fault = False, "sensor_dropout"
        if in_fault and scenario.category == "compute_stall":
            valid, fault = False, "compute_stall"
        return Observation(capture, state.time, selected.position, selected.velocity,
                           selected.obstacles, selected.battery_wh, valid, fault)
