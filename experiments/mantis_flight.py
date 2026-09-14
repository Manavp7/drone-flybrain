"""Mantis: actual YOLO/neural inference, animated actors and motor-only flight.

The three closed-loop runs use the same full perception pipeline and safety
limits, selecting a different bearing for guidance. They naturally see different
camera trajectories. A separate recorded-input benchmark compares estimators on
exactly the same observations. Neither is real-time or physical flight.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.flight_autopilot import Autopilot
from experiments.flight_contracts import (ROOT, PHYSICS_DT, DEPTH_PERIOD_S, ALTITUDE_M,
                                         COMMAND_MAX_CAPTURE_AGE_S, MAX_OBSERVATION_AGE_S)
from experiments.flight_demo import save_json, sha, write_rows, RUNTIME_SOURCES
from experiments.flight_guidance import physical_completion, release, active_request
from experiments.flight_safety import DepthGuardian
from experiments.flight_report import score_records
from experiments.flight_vision import MANIFEST, TINY_SHA
from experiments.mantis_actor import ASSET_SHA256
from experiments.mantis_comparison import (METHODS, CONFIG, BearingInput, AlphaBeta,
                                          direct_bearing, run_comparison, score_comparison)
from experiments.mantis_world import MantisWorld
from experiments.mantis_vision import MantisVision, MANTIS_READOUT_FOLDER, MANTIS_TRACK_MEMORY_S
from experiments.mantis_acceptance import CAUSAL_DEPTH_LIMITS, depth_causality

SPECS = [
    dict(name='walk', kind='normal', duration_s=10., trajectory='diagonal'),
    dict(name='turn', kind='normal', duration_s=10., trajectory='turn'),
    dict(name='brief_loss', kind='recovery', duration_s=10., trajectory='diagonal',
         event_s=3.5, recovery_s=4.),
    dict(name='obstacle', kind='stop', duration_s=8., trajectory='forward', event_s=3.5,
         obstacle_position=[1.75, .40, .7], causal_depth_test=True),
    dict(name='inference_pause', kind='latency_recovery', duration_s=12., trajectory='diagonal',
         event_s=6., injected_delay_s=1.6),
    dict(name='detector_pause', kind='latency_recovery', duration_s=10., trajectory='diagonal',
         event_s=4., injected_delay_s=1.6, pause_stage='detector'),
]
LIMITS = dict(maximum_speed_m_s=.65, maximum_altitude_error_m=.12,
              minimum_normal_translation_m=.4, minimum_normal_target_fraction=.75,
              minimum_normal_iou_fraction=.70, iou_threshold=.5,
              maximum_tail_center_error=.15, minimum_tail_target_fraction=.8,
              maximum_tail_tracking_gap_s=.65, maximum_settled_fault_speed_m_s=.08,
              fault_settle_s=2., minimum_prefault_speed_m_s=.08,
              recovery_deadline_s=2., minimum_depth_override_ticks=1)
BENCHMARK_PLAN = [('clean', 27101), ('noise', 27101), ('noise', 27102),
                  ('noise', 27103), ('gaps', 27101)]
VALIDATION_BENCHMARK_PLAN = [('clean', 38101), ('noise', 38101), ('noise', 38102),
                            ('noise', 38103), ('gaps', 38101)]
SOURCES = sorted(set(RUNTIME_SOURCES + [
    'experiments/mantis_actor.py', 'experiments/mantis_world.py',
    'experiments/mantis_flight.py', 'experiments/mantis_comparison.py',
    'experiments/mantis_depth.py', 'experiments/mantis_vision.py',
    'experiments/mantis_pipeline.py',
    'experiments/mantis_report.py', 'experiments/mantis_acceptance.py',
    'experiments/mantis_calibration.py', 'experiments/flight_report.py', 'assets/mantis_lab.html',
    'assets/mantis_calibration/readout.json', 'assets/mantis_calibration/selection_frozen.json']))


def fixture_pose(spec, time_s):
    """Scenario-only trajectory. This function is never called by guidance."""
    if spec['trajectory'] == 'diagonal':
        point = [4.8+.08*time_s, -.55+.11*time_s, 0.]
        yaw = np.arctan2(.11, .08)
    elif spec['trajectory'] == 'turn':
        phase = .65*time_s+.2
        point = [4.8+.05*time_s, .45*np.sin(phase), 0.]
        yaw = np.arctan2(.45*.65*np.cos(phase), .05)
    elif spec['trajectory'] == 'forward':
        point = [4.8+.06*time_s, .1+.025*time_s, 0.]
        yaw = np.arctan2(.025, .06)
    else:
        raise ValueError('Unknown predeclared actor trajectory')
    hidden = (spec['kind'] == 'recovery' and
              spec['event_s']-1e-9 <= time_s < spec['recovery_s']-1e-9)
    # Development trajectories are deliberately shifted from the frozen cases.
    if spec.get('development_fixture'):
        point = (np.asarray(point)+[.2, -.15, 0.]).tolist()
    if 'trajectory_offset' in spec:
        offset = np.asarray(spec['trajectory_offset'], dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all() or offset[2] != 0:
            raise ValueError('Trajectory offset requires a finite planar displacement')
        point = (np.asarray(point)+offset).tolist()
    return point, float(yaw), hidden


def update_fixture(world, spec):
    point, yaw, hidden = fixture_pose(spec, float(world.data.time))
    world.set_actor(point, yaw, hidden)
    world.set_obstacle(spec.get('obstacle_position', [1.85, 0., .7]), enabled=spec['kind'] == 'stop' and
                       float(world.data.time) >= spec['event_s']-1e-9)


def capture_depth_pair(world, spec):
    """Actual safety input plus an evaluator-only paired counterfactual.

    Only the barrier is hidden for the second synchronized capture. Restore it
    before any motor step. The counterfactual never reaches control authority.
    """
    actual = world.capture(safety=True)
    counterfactual = None
    if spec.get('causal_depth_test') and float(world.data.time) >= spec['event_s']-1e-9:
        position = world._obstacle_position.copy()
        enabled = world._obstacle_enabled
        world.set_obstacle(position, enabled=False)
        try:
            counterfactual = world.capture(safety=True)
        finally:
            world.set_obstacle(position, enabled=enabled)
    return actual, counterfactual


def pause_delay(spec, capture_time_s, already_applied):
    """Inject one predeclared stall; never derive its timing from task truth."""
    if spec['kind'] != 'latency_recovery' or already_applied or capture_time_s < spec['event_s']-1e-9:
        return 0.
    delay = spec.get('injected_delay_s')
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not np.isfinite(delay) or delay <= COMMAND_MAX_CAPTURE_AGE_S:
        raise ValueError('Latency recovery requires a finite delay beyond command expiry')
    return float(delay)


def perceive_with_pause(vision, frame, detector_pause_s=0.):
    """Exercise actual detector deadlines; this delay is included in wall time.

    The detector still executes exactly once. Its stale return must be rejected
    by the real pipeline. This test fixture never supplies boxes or identities.
    """
    if (isinstance(detector_pause_s, bool) or not isinstance(detector_pause_s, (int, float))
            or not np.isfinite(detector_pause_s) or detector_pause_s < 0):
        raise ValueError('Detector test pause must be finite and nonnegative')
    if detector_pause_s == 0:
        return vision.process(frame)
    original = vision.pipeline.detector

    class DelayedDetector:
        def detect(self, image_rgb):
            time.sleep(detector_pause_s)
            return original.detect(image_rgb)

    vision.pipeline.detector = DelayedDetector()
    try:
        return vision.process(frame)
    finally:
        vision.pipeline.detector = original


def bearing_input(frame, bundle, detector_wall_s=0.):
    observation = bundle['observation']
    return BearingInput(bundle['sequence'], float(frame.capture_time_s), frame.rgb.shape[:2],
                        observation['bbox_xyxy'] if observation['valid'] else None,
                        frame.intrinsics, frame.rotation_world_camera, detector_wall_s)


def selected_candidate(method, bundle, frame, angular_filter):
    """Choose bearing from sensor data only; every method needs valid depth.

    The neural candidate is the existing frozen model path. Geometric methods
    use the identical cue envelope, selected ID and registered surface limits.
    They do not require neural success, which would bias the comparison.
    """
    if method not in METHODS:
        raise ValueError('Unknown Mantis controller')
    frame.validate()
    sensor = bearing_input(frame, bundle)
    direct = direct_bearing(sensor)
    filtered = angular_filter.update(frame.capture_time_s, direct)
    heading = direct if method == 'direct_yolo' else filtered
    result = dict(valid=False, reason='target_or_cue_unavailable',
                  heading_world_rad=None, surface_optical_z_m=None, decoded=None,
                  cue_input=bundle['candidate'].get('cue_input'),
                  capture_time_s=float(frame.capture_time_s),
                  track_id=bundle['observation'].get('track_id'))
    if heading is None:
        return result
    surface = bundle['surface']
    if not frame.registration_verified or not isinstance(surface, dict):
        return dict(result, reason='target_depth_unavailable')
    z, fraction, spread = (surface.get('surface_optical_z_m'),
                          surface.get('valid_depth_fraction', 0.),
                          surface.get('depth_spread_p90_p10_m', float('inf')))
    try:
        supported = (not isinstance(z, bool) and np.isfinite([z, fraction, spread]).all()
                     and .3 <= z <= 20 and fraction >= .8 and spread <= .35)
    except (TypeError, ValueError):
        supported = False
    if not supported:
        return dict(result, reason='target_depth_unsupported')
    if method == 'mantis_neural':
        candidate = bundle['candidate']
        if (candidate.get('capture_time_s') != frame.capture_time_s or
                candidate.get('track_id') != result['track_id'] or
                candidate.get('valid') and candidate.get('surface_optical_z_m') != z):
            return dict(result, reason='neural_candidate_sensor_mismatch')
        return dict(candidate)
    return dict(result, valid=True, reason=method+'_bearing_and_registered_depth',
                heading_world_rad=float(heading), surface_optical_z_m=float(z))


def _iou(a, b):
    if a is None or b is None:
        return 0.
    a, b = np.asarray(a), np.asarray(b)
    overlap = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap
    return float(overlap/union) if union > 0 else 0.


def score_episode(spec, ticks, observations, episode):
    """Operational gates; report failed research outcomes without changing them."""
    # Reuse the proven V1 structural/authority checks without borrowing its
    # photograph-specific scores, thresholds, or interpretation of case kinds.
    structural_names = (
        'valid_duration', 'complete_tick_count', 'finite_valid_states',
        'complete_tick_clock', 'state_continuity', 'complete_contact_truth',
        'no_aircraft_contacts', 'bounded_altitude', 'bounded_tilt', 'bounded_speed',
        'bounded_motor_forces', 'causal_observation_clock', 'capture_anchored_candidates',
        'no_future_or_expired_command_application', 'depth_gate_never_increases_forward_request',
        'fixed_target_identity', 'episode_receipt_matches_trace')
    structural = score_records(dict(spec, kind='normal'), ticks, observations, episode)
    structural_gates = {name: structural['gates'][name] for name in structural_names}
    if not all(structural_gates.values()):
        return dict(case=spec['name'], method=spec['method'], passed=False,
                    gates=structural_gates, metrics=structural['metrics'])
    states = [r['state_after'] for r in ticks]
    speeds = np.array([np.linalg.norm(s['velocity']) for s in states])
    altitude = np.array([abs(s['position'][2]-ALTITUDE_M) for s in states])
    truth_rows = [r for r in observations if r['truth_heading_world_rad'] is not None]
    matches = [r for r in truth_rows if r['observation']['valid'] and
               _iou(r['observation']['bbox_xyxy'], r['target_truth_bbox']) >= LIMITS['iou_threshold']]
    tracked = [r for r in observations if r['observation']['valid']]
    translation = float(np.linalg.norm(np.asarray(states[-1]['position'])[:2]-
                       np.asarray(ticks[0]['state_before']['position'])[:2])) if states else 0.
    tail = [r for r in observations if r['capture_time_s'] >= spec['duration_s']-2.]
    valid_tail = [r for r in tail if r['observation']['valid']]
    tail_fraction = len(valid_tail)/len(tail) if tail else 0.
    tail_gaps = np.diff([max(0., spec['duration_s']-2.)]+
                       [r['capture_time_s'] for r in valid_tail]+[spec['duration_s']])
    tail_complete = (len(tail) >= 3 and tail_fraction >= LIMITS['minimum_tail_target_fraction']
                     and max(tail_gaps) <= LIMITS['maximum_tail_tracking_gap_s'])
    tail_center = float(np.mean([abs(r['observation']['center_normalized'][0]) for r in valid_tail])) if tail_complete else None
    contacts = sum(r['truth_after']['contact_count'] for r in ticks)
    gates = dict(structural_gates, completed=episode['status'] == 'completed',
                 full_physics_coverage=len(ticks) == round(spec['duration_s']/PHYSICS_DT),
                 actual_perception=len(observations) > 0 and episode['actual_yolo_calls'] == len(observations)
                    and episode['actual_flyvis_observations'] == len(observations),
                 no_aircraft_contacts=contacts == 0,
                 bounded_speed=bool(len(speeds) and max(speeds) <= LIMITS['maximum_speed_m_s']),
                 bounded_altitude=bool(len(altitude) and max(altitude) <= LIMITS['maximum_altitude_error_m']))
    metrics = dict(translation_m=translation, observations=len(observations), target_observations=len(tracked),
                   geometric_truth_observations=len(truth_rows), overlap_matches=len(matches),
                   target_fraction=len(tracked)/max(1, len(observations)),
                   geometric_overlap_fraction=len(matches)/max(1, len(truth_rows)),
                   tail_mean_center_error=tail_center, aircraft_contact_ticks=contacts,
                   tail_target_fraction=tail_fraction, maximum_tail_tracking_gap_s=float(max(tail_gaps)),
                   max_speed_m_s=float(max(speeds)) if len(speeds) else None,
                   max_altitude_error_m=float(max(altitude)) if len(altitude) else None)
    if spec['kind'] in ('normal', 'recovery', 'latency_recovery'):
        expected_truth = [r for r in observations if not fixture_pose(spec, r['capture_time_s'])[2]]
        gates.update(translated=translation >= LIMITS['minimum_normal_translation_m'],
                     full_geometric_truth=len(truth_rows) == len(expected_truth) and bool(expected_truth),
                     tracked=metrics['target_fraction'] >= LIMITS['minimum_normal_target_fraction'],
                     geometric_overlap=metrics['geometric_overlap_fraction'] >= LIMITS['minimum_normal_iou_fraction'],
                     complete_tail=tail_complete,
                     tail_centered=tail_center is not None and tail_center <= LIMITS['maximum_tail_center_error'])
    if spec['kind'] == 'recovery':
        hidden = [r for r in observations if spec['event_s'] <= r['capture_time_s'] < spec['recovery_s']]
        recovered = [r for r in observations if spec['recovery_s'] <= r['capture_time_s'] <=
                     spec['recovery_s']+LIMITS['recovery_deadline_s'] and r['observation']['valid']]
        initial_id = tracked[0]['observation']['track_id'] if tracked else None
        gates.update(hidden_actor_not_reported=bool(hidden) and all(not r['observation']['valid'] for r in hidden),
                     selected_id_recovered=bool(recovered) and all(r['observation']['track_id'] == initial_id for r in recovered))
        metrics['first_recovery_s'] = recovered[0]['capture_time_s'] if recovered else None
    if spec['kind'] == 'latency_recovery':
        paused = [r for r in observations if r.get('injected_delay_s', 0.) > 0 or
                  r.get('injected_detector_delay_s', 0.) > 0]
        recovery = []
        hold_ticks = []
        stale_rejected = False
        paused_speed = None
        due = [r for r in observations if r['capture_time_s'] >= spec['event_s']-1e-9]
        scheduled = len(paused) == 1 and bool(due) and paused[0]['sequence'] == due[0]['sequence']
        if len(paused) == 1:
            row = paused[0]
            command = row.get('candidate') or {}
            stale_rejected = (command.get('valid') is False and
                              command.get('reason') == 'stale_perception_result' and
                              abs(row.get('injected_delay_s', 0.)+row.get('injected_detector_delay_s', 0.)-
                                  spec['injected_delay_s']) < 1e-9)
            hold_ticks = [r for r in ticks if row['capture_time_s']+COMMAND_MAX_CAPTURE_AGE_S <= r['time_s'] < row['completed_time_s']]
            if hold_ticks:
                paused_speed = float(np.linalg.norm(hold_ticks[-1]['state_after']['velocity']))
            recovery = [r for r in observations if row['completed_time_s'] <= r['capture_time_s'] and
                        r['completed_time_s'] <= row['completed_time_s']+LIMITS['recovery_deadline_s'] and
                        r['observation']['valid'] and (r.get('candidate') or {}).get('valid') is True and
                        r['completed_time_s']-r['capture_time_s'] <= MAX_OBSERVATION_AGE_S+1e-9 and
                        r['candidate'].get('issued_at_s') == r['completed_time_s'] and
                        not r.get('discarded_after_episode', False)]
        initial_id = tracked[0]['observation']['track_id'] if tracked else None
        gates.update(pause_injected_once=scheduled,
                     paused_result_rejected=stale_rejected,
                     expired_pause_commands_hold=bool(hold_ticks) and all(
                         r['request']['forward_speed'] == 0 and r['guardian']['forward_speed'] == 0
                         for r in hold_ticks),
                     paused_vehicle_stopped=paused_speed is not None and paused_speed <= LIMITS['maximum_settled_fault_speed_m_s'],
                     fresh_same_id_after_pause=bool(recovery) and all(
                         r['observation']['track_id'] == initial_id for r in recovery))
        metrics.update(pause_hold_ticks=len(hold_ticks), pause_end_speed_m_s=paused_speed,
                       first_fresh_recovery_s=recovery[0]['completed_time_s'] if recovery else None)
        if spec.get('pause_stage') == 'detector':
            detector_record = paused[0]['detections'] if len(paused) == 1 else {}
            memory = detector_record.get('tracking_memory', {})
            gates['detector_deadline_memory_preserved'] = (len(paused) == 1 and
                paused[0].get('injected_delay_s', 0.) == 0 and
                paused[0].get('injected_detector_delay_s') == spec['injected_delay_s'] and
                paused[0]['inference_wall_s'] >= spec['injected_delay_s'] and
                detector_record.get('reason') == 'inference_deadline_missed' and
                detector_record.get('status') == 'rejected' and
                detector_record.get('inference_executed') is True and
                detector_record.get('control_authority') is False and
                detector_record.get('detections') == [] and
                detector_record.get('processing_ms', -1) >= spec['injected_delay_s']*1000 and
                paused[0]['observation'].get('valid') is False and
                memory.get('preserved_after_deadline') is True and
                type(memory.get('restored_track_count')) is int and memory['restored_track_count'] > 0 and
                memory.get('observation_timestamps_renewed') is False and
                memory.get('control_authority') is False)
    if spec['kind'] == 'stop':
        prefault = [np.linalg.norm(r['state_after']['velocity']) for r in ticks if r['time_s'] < spec['event_s']]
        settled = [np.linalg.norm(r['state_after']['velocity']) for r in ticks if
                   r['time_s'] >= spec['event_s']+LIMITS['fault_settle_s']]
        overrides = sum(r['time_s'] >= spec['event_s'] and r['request']['forward_speed'] > 0 and r['guardian']['forward_speed'] == 0 and
                        r['guardian']['reason'] == 'blocked_stopping_distance' for r in ticks)
        metrics.update(depth_override_ticks=overrides,
                       max_settled_speed_m_s=float(max(settled)) if settled else None)
        post_ticks = [r for r in ticks if r['time_s'] >= spec['event_s']]
        clearance = structural['metrics']['minimum_planar_obstacle_surface_clearance_m']
        metrics['minimum_planar_obstacle_clearance_m'] = clearance
        gates.update(moving_before_obstacle=bool(prefault) and max(prefault) >= LIMITS['minimum_prefault_speed_m_s'],
                     declared_obstacle_present=bool(post_ticks) and all(r['truth_after'].get('obstacle_enabled') is True
                                                                      for r in post_ticks),
                     positive_obstacle_clearance=clearance is not None and clearance > 0.,
                     stopped_after_obstacle=bool(settled) and max(settled) <= LIMITS['maximum_settled_fault_speed_m_s'],
                     no_settled_forward_authority=all(r['guardian']['forward_speed'] == 0 for r in ticks if
                                                    r['time_s'] >= spec['event_s']+LIMITS['fault_settle_s']),
                     depth_independently_braked=overrides >= LIMITS['minimum_depth_override_ticks'])
        if spec.get('causal_depth_test'):
            causal = depth_causality(spec, ticks, observations)
            gates.update(causal['gates'])
            metrics.update(causal['metrics'])
    # NumPy comparisons can yield np.bool_; the public score is also printed
    # directly by the CLI, so keep its contract JSON-native at this boundary.
    gates = {name: bool(value) for name, value in gates.items()}
    return dict(case=spec['name'], method=spec['method'], passed=all(gates.values()), gates=gates, metrics=metrics)


def run_episode(spec, folder, vision):
    folder.mkdir()
    (folder/'frames').mkdir()
    save_json(folder/'spec.json', spec)
    world = MantisWorld()
    pilot, guardian, angular_filter = Autopilot(), DepthGuardian(), AlphaBeta()
    ticks, observations = [], []
    current_command = latest_depth = latest_counterfactual = None
    pause_applied = False
    tick = 0
    total_ticks = round(spec['duration_s']/PHYSICS_DT)
    failure = None
    started = time.perf_counter()
    vision.reset()
    np.savez_compressed(folder/'neural_binding.npz', baseline=vision.brain.baseline_activity,
                        cell_indices=vision.brain.cell_indices, centers_rc=vision.brain.centers_rc)
    update_fixture(world, spec)
    (folder/'world.xml').write_text(world.mjcf)

    def advance(completion):
        nonlocal tick, latest_depth, latest_counterfactual
        goal = min(total_ticks, round(completion/PHYSICS_DT))
        while tick < goal:
            # 20 Hz actor/depth updates; exact actor pose is refreshed for each
            # primary capture below. Aircraft physics continues at 200 Hz.
            if tick % round(DEPTH_PERIOD_S/PHYSICS_DT) == 0:
                update_fixture(world, spec)
                latest_depth, latest_counterfactual = capture_depth_pair(world, spec)
            state = world.state()
            request = active_request(current_command, state)
            safety = guardian.check(latest_depth, state, request['forward_speed'], state.time_s)
            # Evidence only: the actual safety decision above exclusively
            # determines motors. Never merge the unobstructed result into it.
            counterfactual = (guardian.check(latest_counterfactual, state, request['forward_speed'], state.time_s)
                              if latest_counterfactual is not None else None)
            motors = pilot.command(state, safety['forward_speed'], request['yaw_target'], ALTITUDE_M)
            after = world.step(motors)
            ticks.append(dict(index=tick, time_s=state.time_s, state_before=asdict(state),
                              state_after=asdict(after), request=request, guardian=safety,
                              motor_targets=motors.tolist(), applied_command_sequence=request['sequence'],
                              truth_after=world.truth(), counterfactual_guardian=counterfactual,
                              actual_depth_capture_time_s=float(latest_depth.capture_time_s),
                              counterfactual_depth_capture_time_s=(float(latest_counterfactual.capture_time_s)
                                                                  if latest_counterfactual is not None else None)))
            tick += 1

    try:
        while tick < total_ticks:
            update_fixture(world, spec)
            # Overview is evaluator-only and captured outside inference timing.
            overview = world.overview()
            started_inference = time.perf_counter()
            frame = world.capture()
            injected = pause_delay(spec, frame.capture_time_s, pause_applied)
            detector_pause = injected if spec.get('pause_stage') == 'detector' else 0.
            bundle = perceive_with_pause(vision, frame, detector_pause)
            candidate = selected_candidate(spec['method'], bundle, frame, angular_filter)
            elapsed = time.perf_counter()-started_inference
            pause_applied = pause_applied or injected > 0
            completion_delay = injected if not detector_pause else 0.
            completion = physical_completion(frame.capture_time_s, elapsed, completion_delay)
            truth = world.evaluation_projection(frame)
            before_tick = tick
            advance(completion)
            discarded = completion > spec['duration_s']+1e-8
            command = None if discarded else release(candidate, world.state(), completion, bundle['sequence'])
            if command is not None:
                current_command = command
            neural = bundle['neural']
            row = dict(sequence=bundle['sequence'], capture_time_s=float(frame.capture_time_s),
                       overview_capture_time_s=float(frame.capture_time_s), completed_time_s=completion,
                       inference_wall_s=elapsed, injected_delay_s=completion_delay,
                       injected_detector_delay_s=detector_pause,
                       delay_physics_tick_start=before_tick, delay_physics_tick_end=tick,
                       discarded_after_episode=discarded, observation=bundle['observation'],
                       detections=bundle['detections'], surface=bundle['surface'], estimate=candidate,
                       ego_motion=bundle.get('ego_motion', []),
                       candidate=command, neural_candidate=bundle['candidate'],
                       target_truth_bbox=truth['bbox_xyxy'], truth_heading_world_rad=truth['heading_world_rad'],
                       truth_annotation=truth, image_hw=list(frame.rgb.shape[:2]),
                       camera_intrinsics=frame.intrinsics, camera_rotation=frame.rotation_world_camera,
                       camera_position=frame.position_world_camera, neural_features=neural['features'],
                       neural_valid=neural['valid'], neural_elapsed_wall_s=neural['elapsed_wall_s'],
                       neural_stimulus_time_s=neural['stimulus_time_s'],
                       neural_response_time_s=neural['response_time_s'],
                       frame_file='frames/%06d.npz'%bundle['sequence'])
            observations.append(row)
            np.savez_compressed(folder/row['frame_file'], rgb=frame.rgb, depth_m=frame.depth_m,
                                overview=overview, features=neural['features'], retina=neural['retina'],
                                activity=neural['activity'])
            if len(observations) % 25 == 0:
                print(json.dumps(dict(case=spec['name'], method=spec['method'],
                      sim_s=round(float(world.data.time), 3), captures=len(observations),
                      tracked=bundle['observation']['valid'], guidance=candidate['reason'])), flush=True)
    except Exception as exc:
        failure = dict(type=type(exc).__name__, reason=str(exc))
        raise
    finally:
        write_rows(folder/'ticks.jsonl', ticks)
        write_rows(folder/'observations.jsonl', observations)
        episode = dict(status='failed' if failure else 'completed', failure=failure,
                       whole_wall_s=time.perf_counter()-started, physics_ticks=len(ticks),
                       observations=len(observations), final_state=asdict(world.state()),
                       actual_yolo_calls=sum(r['detections']['inference_executed'] for r in observations),
                       actual_flyvis_observations=len(observations), neural_steps=5*len(observations),
                       method=spec['method'], real_time=False, physical_flight=False,
                       rotor_forces_are_the_only_aircraft_actuation=True,
                       perception='Actual YOLOX-tiny and frozen Flyvis in every controller run',
                       timing='Measured shared perception delay; independent closed-loop paths; logging excluded from inference')
        save_json(folder/'episode.json', episode)
        world.close()
    score = score_episode(spec, ticks, observations, episode)
    save_json(folder/'checks.json', score)
    return score


def run_benchmarks(output, vision, specs, plan):
    comparisons = []
    for spec in specs:
        if spec['kind'] != 'normal':
            continue
        source = output/(spec['name']+'__mantis_neural')/'observations.jsonl'
        if not source.is_file():
            raise ValueError('Requested benchmark is missing its Mantis Neural camera recording')
        records = [json.loads(line) for line in source.read_text().splitlines()]
        inputs = [BearingInput(r['sequence'], r['capture_time_s'], tuple(r['image_hw']),
                  r['observation']['bbox_xyxy'] if r['observation']['valid'] else None,
                  tuple(r['camera_intrinsics']), r['camera_rotation'],
                  # Detector receipt already measures the actual detector call.
                  float(r['detections']['processing_ms'])/1000.) for r in records]
        truth = [dict(sequence=r['sequence'], capture_time_s=r['capture_time_s'],
                      heading_world_rad=r['truth_heading_world_rad']) for r in records]
        for variant, seed in plan:
            result = run_comparison(inputs, vision.brain, vision.readout, variant=variant, seed=seed)
            score = score_comparison(result['rows'], truth)
            entry = dict(case=spec['name'], source_method='mantis_neural', variant=variant, seed=seed,
                         input_sha256=sha(source), result=result, score=score)
            comparisons.append(entry)
            save_json(output/'comparison.json', comparisons)
            print(json.dumps(dict(benchmark=spec['name'], variant=variant, seed=seed,
                                  matched=score['matched_count'])), flush=True)
    return comparisons


def run(output, specs, methods=METHODS, development=False, benchmark=True):
    if not specs or not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('Distinct known methods and at least one scenario are required')
    if len({s['name'] for s in specs}) != len(specs):
        raise ValueError('Scenario names must be distinct')
    if benchmark and any(s['kind'] == 'normal' for s in specs) and 'mantis_neural' not in methods:
        raise ValueError('Matched benchmark needs --method mantis_neural or all methods; use --skip-benchmark otherwise')
    benchmark_plan = (VALIDATION_BENCHMARK_PLAN if any(s.get('validation_fixture') for s in specs)
                      else BENCHMARK_PLAN)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {p: sha(ROOT/p) for p in SOURCES}
    (output/'sources').mkdir()
    for path in source_hashes:
        shutil.copy2(ROOT/path, output/'sources'/path.replace('/', '__'))
    save_json(output/'definition.json', dict(name='Mantis', version='2.1',
        frozen_utc=datetime.now(timezone.utc).isoformat(), specs=specs, methods=list(methods),
        development=development, source_hashes=source_hashes, criteria=LIMITS,
        causal_depth_criteria=CAUSAL_DEPTH_LIMITS,
        tracking_memory_s=MANTIS_TRACK_MEMORY_S,
        comparison_config=CONFIG, benchmark_plan=benchmark_plan if benchmark else [],
        physics_dt_s=PHYSICS_DT, actor_update_period_s=DEPTH_PERIOD_S,
        detector_sha256=TINY_SHA, actor_sha256=ASSET_SHA256,
        neural_manifest_sha256=sha(MANIFEST), readout_sha256=sha(MANTIS_READOUT_FOLDER/'readout.json'),
        readout_selection_sha256=sha(MANTIS_READOUT_FOLDER/'selection_frozen.json'),
        readout_source=str(MANTIS_READOUT_FOLDER.relative_to(ROOT)),
        description='Stylized animated 3D person; ideal camera/depth/state; approximate capsule contact',
        source_representation='Animated Cesium Man (adapted clothing), true skinned 3D mesh; ideal depth/state; approximate capsule contacts',
        physical_control_authority=False, real_time=False,
        upstream_neural_model='Flyvis; project display name Mantis',
        benchmark_scope='Capture-time bearing on same boxes from Mantis Neural camera paths; not superiority evidence'))
    scores = []
    try:
        vision = MantisVision()
        save_json(output/'runtime.json', dict(packages=vision.brain.adapter.runtime,
                  device=str(vision.brain.adapter.device), mujoco='3.2.7', readout_frozen=vision.frozen))
        for spec in specs:
            for method in methods:
                scored = run_episode(dict(spec, method=method), output/(spec['name']+'__'+method), vision)
                scores.append(scored)
                save_json(output/'checks.json', dict(complete=False, cases=scores))
                print(json.dumps(scored), flush=True)
        if benchmark:
            comparisons = run_benchmarks(output, vision, specs, benchmark_plan)
            if len(comparisons) != sum(s['kind'] == 'normal' for s in specs)*len(benchmark_plan):
                raise RuntimeError('Incomplete benchmark execution')
        if {(s['case'], s['method']) for s in scores} != {(s['name'], m) for s in specs for m in methods}:
            raise RuntimeError('Incomplete scenario/controller execution')
        if any(sha(ROOT/p) != h for p, h in source_hashes.items()):
            raise RuntimeError('Sources changed during the frozen experiment')
        save_json(output/'checks.json', dict(complete=True, passed=all(s['passed'] for s in scores),
                                            development=development, cases=scores))
        save_json(output/'execution_complete.json', dict(complete=True, development=development,
                  finished_utc=datetime.now(timezone.utc).isoformat(), cases=len(scores)))
    except Exception as exc:
        save_json(output/'failure.json', dict(type=type(exc).__name__, reason=str(exc)))
        raise
    finally:
        save_json(output/'artifact_hashes.json', {str(p.relative_to(output)): sha(p)
            for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'artifact_hashes.json'})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', choices=[s['name'] for s in SPECS])
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--development', action='store_true')
    parser.add_argument('--validation-fixture', action='store_true',
                        help='Predeclared fresh path offset (-0.10,+0.12,0); separate from development fixtures')
    parser.add_argument('--duration', type=float)
    parser.add_argument('--skip-benchmark', action='store_true')
    args = parser.parse_args()
    if args.development and args.validation_fixture:
        parser.error('--development and --validation-fixture are distinct fixtures')
    if args.duration is not None and (not args.development or not .2 <= args.duration <= 30):
        parser.error('--duration needs --development and a value in [.2,30]')
    specs = [dict(s) for s in SPECS if args.case is None or s['name'] == args.case]
    if args.development:
        for spec in specs:
            spec['development_fixture'] = True
    if args.validation_fixture:
        for spec in specs:
            spec['trajectory_offset'] = [-.10, .12, 0.]
            spec['validation_fixture'] = True
    if args.duration is not None:
        for spec in specs:
            spec['duration_s'] = round(args.duration/PHYSICS_DT)*PHYSICS_DT
    methods = METHODS if args.method is None else (args.method,)
    run(args.output, specs, methods, args.development, not args.skip_benchmark)


if __name__ == '__main__':
    main()
