"""Deterministic geometric sensing for the hard-course evaluation.

This is a sensor fault/visibility model, not a camera or visual localization.
Static boxes remain a declared prior. Dynamic boxes require range and a clear
centerline to their center at capture time. Visibility is computed before delay
selection, so delayed observations cannot reveal newly visible geometry.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict
import math
from collections.abc import Mapping

from flybrain_sim.contracts import Observation, Scenario, VehicleState
from flybrain_sim.geometry import obstacles_at, point_box_distance, segment_intersects_box
from flybrain_sim.sensors import _samples


def profile_config(profile) -> dict:
    if isinstance(profile, Mapping):
        return dict(profile)
    if hasattr(profile, "__dataclass_fields__"):
        return asdict(profile)
    return {"name": str(profile)}


class StressSensorModel:
    def __init__(self, scenario: Scenario, profile=None, obstacle_function=None):
        self.scenario = scenario
        self.config = profile_config(profile or {})
        self.obstacle_function = obstacle_function or obstacles_at
        self.base_compatible = self.config.get("sensor_mode") == "base"
        self.occlusion = bool(self.config.get("dynamic_occlusion", True)) and not self.base_compatible
        self.sensor_range = float(self.config.get("sensor_range", 6.0))
        self._history: deque[Observation] = deque(maxlen=max(2, scenario.latency_steps + 1))
        self._clock_reset_emitted = False
        self.last_metadata: dict = {}
        self._metadata_history: deque[dict] = deque(maxlen=self._history.maxlen)
        self.dynamic_ids = {b.id for b in scenario.obstacles if b.velocity != (0.0, 0.0, 0.0)}

    def _visible(self, state: VehicleState):
        boxes = self.obstacle_function(self.scenario, state.time)
        if self.base_compatible:
            visible = tuple(box for box in boxes
                            if box.id != "moving-inspection-fixture" or
                            point_box_distance(state.position, box) <= self.scenario.sensor_range)
            return visible, {"capture_truth_time": state.time,
                             "visible_dynamic_ids": [b.id for b in visible if b.id in self.dynamic_ids],
                             "hidden_dynamic": []}
        static = tuple(b for b in boxes if b.id not in self.dynamic_ids)
        result, hidden = list(static), []
        for box in boxes:
            if box.id not in self.dynamic_ids:
                continue
            if point_box_distance(state.position, box) > self.sensor_range:
                hidden.append({"id": box.id, "reason": "range"})
                continue
            center = tuple((a + b) * 0.5 for a, b in zip(box.low, box.high))
            blocker = next((b.id for b in static if self.occlusion and
                            segment_intersects_box(state.position, center, b, radius=0.0)), None)
            if blocker is not None:
                hidden.append({"id": box.id, "reason": "static_occlusion", "blocker": blocker})
                continue
            result.append(box)
        return tuple(result), {"capture_truth_time": state.time,
                               "visible_dynamic_ids": [b.id for b in result if b.id in self.dynamic_ids],
                               "hidden_dynamic": hidden}

    def observe(self, state: VehicleState) -> Observation:
        scenario = self.scenario
        tick = round(state.time / scenario.dt)
        samples = _samples(scenario.seed, tick)
        in_fault = (scenario.fault_start <= state.time < scenario.fault_start + scenario.fault_duration)
        amplitude = scenario.sensor_noise * math.sqrt(3.0)
        position = tuple(state.position[i] + amplitude * (2.0 * samples[i] - 1.0) for i in range(3))
        velocity = tuple(state.velocity[i] + 0.5 * amplitude * (2.0 * samples[i + 3] - 1.0) for i in range(3))
        visible, metadata = self._visible(state)
        current = Observation(state.time, state.time, position, velocity, visible, state.battery_wh)
        self._history.append(current)
        self._metadata_history.append(metadata)
        delayed = in_fault and scenario.latency_steps
        selected = self._history[0] if delayed else current
        selected_metadata = self._metadata_history[0] if delayed else metadata
        valid = True
        fault = "latency" if selected.capture_time < state.time - 1e-9 else ""
        active_faults = [fault] if fault else []
        capture = selected.capture_time
        clock_reset = self.config.get("clock_reset", scenario.category in ("clock_reset", "compound"))
        compute_stall = self.config.get("compute_stall", scenario.category == "compute_stall")
        if in_fault and clock_reset:
            capture -= 12.0
            fault = "clock_reset" if not self._clock_reset_emitted else "clock_offset"
            active_faults.append(fault)
            self._clock_reset_emitted = True
        if in_fault and samples[6] < scenario.dropout_probability:
            valid, fault = False, "sensor_dropout"
            active_faults.append(fault)
        if in_fault and compute_stall:
            valid, fault = False, "compute_stall"
            active_faults.append(fault)
        self.last_metadata = {**selected_metadata, "in_fault_window": in_fault,
                              "active_faults": active_faults,
                              "actual_sample_age_s": state.time - selected.capture_time}
        return Observation(capture, state.time, selected.position, selected.velocity,
                           selected.obstacles, selected.battery_wh, valid, fault)
