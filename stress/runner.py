"""Frozen-controller hard-course harness with every integration sample retained.

Scoring uses the original run_episode criteria. Instrumentation does not enter
controller inputs. All trace floats are rounded to six decimal places only after
scoring; scalar episode metrics retain their original floating-point values.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
import time

from flybrain_sim.autonomy import Autonomy
from flybrain_sim.brain import make_brain
from flybrain_sim.contracts import EpisodeResult, VehicleState
from flybrain_sim.geometry import (clearance, distance, norm, obstacles_at,
                                  point_box_distance, segment_intersects_box,
                                  within_bounds)
from flybrain_sim.physics import step
from flybrain_sim.runner import RADIUS, TruthInspectionTracker, motion_envelopes
from .sensors import StressSensorModel, profile_config


def _logged(value):
    if isinstance(value, float):
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _logged(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_logged(item) for item in value]
    return value


def _nearest(position, boxes):
    if not boxes:
        return None, None
    box = min(boxes, key=lambda item: point_box_distance(position, item))
    return box.id, point_box_distance(position, box) - RADIUS


def _percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    idx = fraction * (len(ordered) - 1)
    low, high = math.floor(idx), math.ceil(idx)
    return ordered[low] + (ordered[high] - ordered[low]) * (idx - low)


def _entry_fraction(start, end, box):
    enter, leave = 0.0, 1.0
    for i in range(3):
        delta = end[i] - start[i]
        low, high = box.low[i] - RADIUS, box.high[i] + RADIUS
        if abs(delta) < 1e-12:
            if start[i] < low or start[i] > high:
                return None
        else:
            first, second = sorted(((low - start[i]) / delta, (high - start[i]) / delta))
            enter, leave = max(enter, first), min(leave, second)
            if enter > leave:
                return None
    return enter


def run_trial(scenario, profile, record=True, *, obstacle_function=None):
    """Return exact metrics, all-tick trace, events, contacts and diagnostics.

    `obstacle_function` and `sensor_mode=base` exist for scorer-parity tests.
    Normal stress runs use the course's explicitly bounded motion and the hard
    sensor visibility contract. No controller or core module is modified.
    """
    start_wall = time.perf_counter()
    if scenario.dt <= 0 or scenario.max_time <= 0:
        raise ValueError("Positive timestep and duration required")
    config = profile_config(profile)
    if obstacle_function is None:
        from .course import obstacles_at_trial
        obstacle_function = obstacles_at_trial
    variant = config.get("variant", "baseline")
    controller = Autonomy(scenario)
    brain = make_brain(variant)
    sensors = StressSensorModel(scenario, config, obstacle_function)
    state = VehicleState(0.0, scenario.home, (0.0, 0.0, 0.0), scenario.initial_battery_wh)
    inspection = TruthInspectionTracker(scenario.waypoints)
    dynamic_ids = {box.id for box in scenario.obstacles if box.velocity != (0.0, 0.0, 0.0)}
    trace, events, contacts = [], [], []
    collided = violated = False
    min_clearance, min_static, min_dynamic = math.inf, math.inf, math.inf
    travelled = 0.0
    last_guidance, last_faults, last_accepted, last_visible = None, None, None, None
    update_timings, peak_speed, peak_acceleration = [], 0.0, 0.0
    peak_error, peak_raw_error, max_age, rejected_seconds = 0.0, 0.0, 0.0, 0.0
    fault_seconds, mode_seconds, reason_seconds = Counter(), Counter(), Counter()
    inspection_times: dict[int, float] = {}
    command_count, step_count = 0, 0
    outcome = "timeout"
    latest_command = None
    last_sample_time = None

    def world_sample(sample_state, sample_kind, command, observation=None,
                     accepted=None, observation_metadata=None, update_wall_s=None):
        nonlocal min_clearance, min_static, min_dynamic, peak_error, peak_raw_error, max_age
        nonlocal last_sample_time
        boxes = obstacle_function(scenario, sample_state.time)
        static = tuple(box for box in boxes if box.id not in dynamic_ids)
        dynamic = tuple(box for box in boxes if box.id in dynamic_ids)
        near_id, body_clearance = _nearest(sample_state.position, boxes)
        static_id, static_clearance = _nearest(sample_state.position, static)
        dynamic_id, dynamic_clearance = _nearest(sample_state.position, dynamic)
        min_clearance = min(min_clearance, body_clearance if body_clearance is not None else math.inf)
        min_static = min(min_static, static_clearance if static_clearance is not None else math.inf)
        min_dynamic = min(min_dynamic, dynamic_clearance if dynamic_clearance is not None else math.inf)
        estimated_error = (distance(controller.tracker.position, sample_state.position)
                           if controller.tracker.position is not None else None)
        # A post-integration terminal sample has no estimator update. Do not treat
        # its intentionally earlier estimate as a current-time estimation error.
        if sample_kind == "control":
            peak_error = max(peak_error, estimated_error or 0.0)
        raw_error = distance(observation.position, sample_state.position) if observation is not None else None
        if raw_error is not None:
            peak_raw_error = max(peak_raw_error, raw_error)
            max_age = max(max_age, sample_state.time - observation.capture_time)
        last_sample_time = sample_state.time
        if not record:
            return
        tracker = controller.tracker
        item = {"t": sample_state.time, "sample_kind": sample_kind,
                "p": sample_state.position, "v": sample_state.velocity,
                "battery_wh": sample_state.battery_wh, "energy_used_wh": sample_state.energy_used_wh,
                "clearance_m": body_clearance, "nearest_obstacle_id": near_id,
                "nearest_static_id": static_id, "static_clearance_m": static_clearance,
                "nearest_dynamic_id": dynamic_id, "dynamic_clearance_m": dynamic_clearance,
                "dynamic_obstacles": [asdict(box) for box in dynamic],
                "inside_bounds": within_bounds(sample_state.position, scenario.bounds, RADIUS),
                "mode": command.mode if command is not None else outcome,
                "command": asdict(command) if command is not None else None,
                "observation": None,
                "tracker": {"position": tracker.position, "velocity": tracker.velocity,
                            "disturbance": tracker.disturbance, "last_valid": tracker.last_valid,
                            "now": tracker.now, "position_error_m": estimated_error},
                "truth_inspected_waypoints": sorted(inspection.completed),
                "truth_dwell_started": inspection.started,
                "controller_completed_waypoints": controller.completed,
                "controller_target": controller._target,
                "controller_route": controller._route,
                "controller_abort_reason": controller.abort_reason,
                "controller_returning": controller._returning,
                "controller_recovery_memory": (controller.obstacle_memory.snapshot(sample_state.time)
                                               if hasattr(controller, "obstacle_memory") else None),
                "update_wall_seconds": update_wall_s,
                "terminal_outcome": None if sample_kind == "control" else outcome}
        if observation is not None:
            item["observation"] = {
                "capture_time": observation.capture_time, "receive_time": observation.receive_time,
                "position": observation.position, "velocity": observation.velocity,
                "battery_wh": observation.battery_wh, "valid": observation.valid,
                "fault": observation.fault, "accepted": accepted,
                "reported_age_s": sample_state.time - observation.capture_time,
                "position_difference_from_current_truth_m": raw_error,
                "observed_static_ids": [b.id for b in observation.obstacles if b.id not in dynamic_ids],
                "observed_dynamic_obstacles": [asdict(b) for b in observation.obstacles if b.id in dynamic_ids],
                "visibility_and_fault_metadata": observation_metadata}
        trace.append(_logged(item))

    for tick in range(math.ceil(scenario.max_time / scenario.dt)):
        observation = sensors.observe(state)
        start_update = time.perf_counter()
        neural = brain.update(observation, state.time)
        command = controller.update(observation, state.time, neural)
        update_wall = time.perf_counter() - start_update
        update_timings.append(update_wall)
        command_count += 1
        latest_command = command
        accepted = controller.tracker.last_observation is observation
        current_guidance = (command.mode, command.reason)
        if current_guidance != last_guidance:
            events.append({"t": state.time, "type": "guidance_change", "mode": command.mode,
                           "reason": command.reason, "previous": last_guidance})
            last_guidance = current_guidance
        faults = tuple(sensors.last_metadata["active_faults"])
        if faults != last_faults:
            events.append({"t": state.time, "type": "fault_change", "active_faults": faults,
                           "reported_fault": observation.fault, "previous": last_faults})
            last_faults = faults
        if accepted != last_accepted:
            events.append({"t": state.time, "type": "observation_acceptance_change", "accepted": accepted,
                           "reason": command.reason, "fault": observation.fault})
            last_accepted = accepted
        visible = tuple(sensors.last_metadata["visible_dynamic_ids"])
        if visible != last_visible:
            events.append({"t": state.time, "type": "dynamic_visibility_change", "visible_ids": visible,
                           "capture_truth_time": sensors.last_metadata["capture_truth_time"]})
            last_visible = visible

        previous_inspected = set(inspection.completed)
        inspection.update(state.position, state.velocity, state.time)
        for index in sorted(inspection.completed - previous_inspected):
            inspection_times[index] = state.time
            events.append({"t": state.time, "type": "truth_waypoint_completed", "waypoint": index})
        world_sample(state, "control", command, observation, accepted, sensors.last_metadata, update_wall)
        peak_speed = max(peak_speed, norm(state.velocity))

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
            break

        current_obstacles = obstacle_function(scenario, state.time)
        next_state = step(state, command.velocity, scenario)
        next_obstacles = obstacle_function(scenario, next_state.time)
        envelopes = motion_envelopes(current_obstacles, next_obstacles)
        hit_boxes = [box for box in envelopes if segment_intersects_box(
                     state.position, next_state.position, box, radius=RADIUS)]
        collided = bool(hit_boxes)
        violated = not within_bounds(next_state.position, scenario.bounds, radius=RADIUS)
        for box in hit_boxes:
            fraction = _entry_fraction(state.position, next_state.position, box)
            contacts.append({"t_start": state.time, "t_end": next_state.time,
                             "obstacle_id": box.id, "dynamic": box.id in dynamic_ids,
                             "segment_start": state.position, "segment_end": next_state.position,
                             "conservative_envelope": asdict(box), "entry_fraction": fraction,
                             "envelope_contact_time": state.time + scenario.dt * fraction,
                             "method": "closed segment against radius-expanded motion AABB"})
        travelled += distance(state.position, next_state.position)
        peak_speed = max(peak_speed, norm(next_state.velocity))
        peak_acceleration = max(peak_acceleration, distance(state.velocity, next_state.velocity) / scenario.dt)
        for fault in faults:
            fault_seconds[fault] += scenario.dt
        mode_seconds[command.mode] += scenario.dt
        reason_seconds[command.reason] += scenario.dt
        if not accepted:
            rejected_seconds += scenario.dt
        step_count += 1
        state = next_state
        min_clearance = min(min_clearance, clearance(state.position, next_obstacles) - RADIUS)
        if collided or violated or state.battery_wh <= 0:
            outcome = "collision" if collided else "geofence_violation" if violated else "battery_depleted"
            break

    # Every endpoint is retained. A terminal decision made at an already-recorded
    # control time annotates that sample; a physics/timeout endpoint gets its own
    # sample with no invented new sensor reading or controller update.
    if last_sample_time is None or abs(last_sample_time - state.time) > 1e-9:
        world_sample(state, "terminal", latest_command)
    elif record:
        trace[-1]["terminal_outcome"] = outcome
    events.append({"t": state.time, "type": "terminal", "outcome": outcome,
                   "reason": controller.abort_reason or "independent simulator-truth terminal check",
                   "failure_stage": "return" if controller._returning else f"inspection_waypoint_{controller.completed}"})
    actual_home = distance(state.position, scenario.home) <= 1.0 and norm(state.velocity) <= 0.6
    result = EpisodeResult(
        seed=scenario.seed, category=scenario.category, variant=variant, outcome=outcome,
        mission_complete=outcome == "mission_complete", returned_home=actual_home and not collided and not violated,
        collision=bool(collided), geofence_violation=bool(violated), waypoints_completed=len(inspection.completed),
        waypoints_total=len(scenario.waypoints), simulated_seconds=state.time, energy_wh=state.energy_used_wh,
        minimum_clearance_m=min_clearance if math.isfinite(min_clearance) else 0.0,
        distance_m=travelled, interventions=controller.interventions,
        stale_observations=controller.stale_observations, brain_rejections=controller.brain_rejections,
        wall_seconds=time.perf_counter() - start_wall)
    result_dict = asdict(result)
    result_dict.pop("trajectory")
    result_dict.pop("events")
    diagnostics = {
        "profile": config.get("name", scenario.category),
        "command_updates": command_count, "integration_steps": step_count,
        "trace_samples": len(trace), "trace_float_decimal_places": 6,
        "final_position": state.position, "final_velocity": state.velocity,
        "remaining_battery_wh": state.battery_wh,
        "distance_to_home_m": distance(state.position, scenario.home),
        "peak_speed_m_s": peak_speed, "peak_acceleration_m_s2": peak_acceleration,
        "peak_tracker_position_error_m": peak_error,
        "peak_raw_observation_difference_from_current_truth_m": peak_raw_error,
        "maximum_reported_observation_age_s": max_age,
        "rejected_observation_seconds": rejected_seconds,
        "fault_seconds": dict(fault_seconds), "mode_seconds": dict(mode_seconds),
        "reason_seconds": dict(reason_seconds),
        "minimum_static_clearance_m": min_static if math.isfinite(min_static) else None,
        "minimum_dynamic_clearance_m": min_dynamic if math.isfinite(min_dynamic) else None,
        "truth_waypoint_completion_times_s": inspection_times,
        "truth_waypoints_completed_in_order": list(inspection_times) == sorted(inspection_times),
        "controller_completed_waypoints": controller.completed,
        "controller_abort_reason": controller.abort_reason,
        "terminal_stage": "return" if controller._returning else f"inspection_waypoint_{controller.completed}",
        "contact_obstacle_ids": [item["obstacle_id"] for item in contacts],
        "contact_dynamic": any(item["dynamic"] for item in contacts),
        "update_wall_seconds": {"mean": sum(update_timings) / len(update_timings),
                                "p50": _percentile(update_timings, 0.5),
                                "p95": _percentile(update_timings, 0.95),
                                "p99": _percentile(update_timings, 0.99),
                                "maximum": max(update_timings),
                                "count_over_simulation_dt": sum(t > scenario.dt for t in update_timings)},
        "timing_note": "Host wall timing includes interpreter and scheduler effects; not onboard or real-time evidence.",
        "scoring_note": "Original unordered independent waypoint dwell score preserved; ordered completion reported separately.",
        "sensor_contract": {"dynamic_range_m": scenario.sensor_range if sensors.base_compatible else sensors.sensor_range,
                            "static_map_prior": True, "dynamic_centerline_occlusion": sensors.occlusion},
    }
    return {"result": result_dict, "trace": trace, "events": _logged(events),
            "contacts": _logged(contacts), "diagnostics": diagnostics}
