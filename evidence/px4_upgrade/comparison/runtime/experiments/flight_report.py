"""Fail-closed scoring and causal previews of saved motor-flight experiments.

No model inference or simulator steps occur here. The photograph's projected
foreground is approximate fixture ground truth, not a person-tracking benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from experiments.flight_contracts import PHYSICS_DT, COMMAND_MAX_CAPTURE_AGE_S

FPS = 10
CASE_ORDER = ('approach_left', 'approach_right', 'moving_target', 'target_loss',
              'depth_loss', 'stale_depth', 'delayed_inference', 'obstacle_stop',
              'zero_neural_features')
LIMITS = dict(altitude_m=1.1, altitude_tolerance_m=.12, maximum_tilt_rad=.30,
              maximum_speed_m_s=.65, maximum_motor_force_n=5.,
              normal_minimum_translation_m=.6, normal_target_fraction=.85,
              active_iou_threshold=.5, active_iou_fraction=.8,
              tail_mean_horizontal_center_error=.1, tail_standoff_error_m=.6,
              desired_surface_optical_z_m=3.5, minimum_prefault_speed_m_s=.08,
              minimum_prefault_target_fraction=.8, settled_speed_m_s=.08,
              event_settle_s=2., ablation_maximum_translation_m=.04)


def _read_json(path):
    return json.loads(Path(path).read_text())


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _array(value, shape):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return result if result.shape == shape and np.isfinite(result).all() else None


def _iou(a, b):
    a, b = _array(a, (4,)), _array(b, (4,))
    if a is None or b is None or np.any(a[2:] <= a[:2]) or np.any(b[2:] <= b[:2]):
        return 0.
    intersection = np.maximum(0., np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-intersection
    return float(intersection/union)


def _state_valid(state):
    if not isinstance(state, dict) or not _number(state.get('time_s')):
        return False
    for key, shape in [('position', (3,)), ('velocity', (3,)), ('rotation', (3, 3)),
                       ('angular_velocity', (3,)), ('motor_forces', (4,))]:
        if _array(state.get(key), shape) is None:
            return False
    rotation = np.asarray(state['rotation'])
    return bool(np.allclose(rotation.T@rotation, np.eye(3), atol=1e-6)
                and abs(np.linalg.det(rotation)-1) < 1e-6)


def _same_state(a, b):
    return (abs(a['time_s']-b['time_s']) < 1e-7 and all(
        np.allclose(a[key], b[key], atol=1e-7, rtol=0.)
        for key in ('position', 'velocity', 'rotation', 'angular_velocity', 'motor_forces')))


def score_records(spec, ticks, observations, episode=None):
    """Score every supplied row; loss, missing coverage and malformed data fail.

    Tracking is capture-aligned, including an explicitly discarded final result.
    Control gates consider only commands released before the episode finishes.
    """
    failures, gates = [], {}

    def gate(name, passed, detail=None):
        gates[name] = bool(passed)
        if not passed:
            failures.append(name if detail is None else f'{name}: {detail}')

    duration = spec.get('duration_s')
    valid_duration = _number(duration) and duration > 0 and abs(duration/PHYSICS_DT-round(duration/PHYSICS_DT)) < 1e-7
    gate('valid_duration', valid_duration)
    if not valid_duration:
        duration = 0.
    expected = round(duration/PHYSICS_DT)
    gate('complete_tick_count', expected > 0 and len(ticks) == expected,
         f'{len(ticks)} of {expected} ticks')
    finite_states = bool(ticks) and all(_state_valid(t.get('state_before')) and _state_valid(t.get('state_after')) for t in ticks)
    gate('finite_valid_states', finite_states)
    tick_clock = bool(ticks) and all(
        t.get('index') == i and _number(t.get('time_s'))
        and abs(t['time_s']-i*PHYSICS_DT) < 1e-7
        and _number(t.get('state_before', {}).get('time_s'))
        and _number(t.get('state_after', {}).get('time_s'))
        and abs(t['state_before']['time_s']-i*PHYSICS_DT) < 1e-7
        and abs(t['state_after']['time_s']-(i+1)*PHYSICS_DT) < 1e-7
        for i, t in enumerate(ticks))
    gate('complete_tick_clock', tick_clock)
    gate('state_continuity', finite_states and all(
        _same_state(a['state_after'], b['state_before']) for a, b in zip(ticks, ticks[1:])))
    contacts_valid = bool(ticks) and all(
        isinstance(t.get('truth_after'), dict)
        and _number(t['truth_after'].get('time_s'))
        and abs(t['truth_after']['time_s']-t.get('state_after', {}).get('time_s', -1)) < 1e-7
        and isinstance(t['truth_after'].get('contacts'), list)
        and t['truth_after'].get('contact_count') == len(t['truth_after']['contacts']) for t in ticks)
    gate('complete_contact_truth', contacts_valid)
    contact_count = sum(len(t.get('truth_after', {}).get('contacts', [])) for t in ticks)
    gate('no_aircraft_contacts', contacts_valid and contact_count == 0)
    states = [ticks[0]['state_before']]+[t['state_after'] for t in ticks] if finite_states else []
    positions = np.asarray([s['position'] for s in states]) if states else np.empty((0, 3))
    speeds = np.asarray([np.linalg.norm(s['velocity']) for s in states])
    times = np.asarray([s['time_s'] for s in states])
    tilts = np.asarray([math.acos(float(np.clip(s['rotation'][2][2], -1., 1.))) for s in states])
    motors = np.asarray([s['motor_forces'] for s in states])
    targets = [_array(t.get('motor_targets'), (4,)) for t in ticks]
    gate('bounded_altitude', bool(states) and bool(np.all(abs(positions[:, 2]-1.1) <= .12+1e-9)))
    gate('bounded_tilt', bool(states) and bool(np.all(tilts < .30)))
    gate('bounded_speed', bool(states) and bool(np.all(speeds < .65)))
    gate('bounded_motor_forces', bool(states) and bool(np.all((motors >= 0) & (motors <= 5.)))
         and bool(targets) and all(a is not None and bool(np.all((a >= 0) & (a <= 5.))) for a in targets))
    translation = float(np.linalg.norm(positions[-1, :2]-positions[0, :2])) if states else None
    path_length = float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()) if states else None

    observation_clock, candidate_integrity = bool(observations), bool(observations)
    commands = {}
    for i, row in enumerate(observations):
        capture, complete = row.get('capture_time_s'), row.get('completed_time_s')
        wall, delay = row.get('inference_wall_s'), row.get('injected_delay_s', 0.)
        clock_ok = (row.get('sequence') == i and all(_number(v) for v in (capture, complete, wall, delay))
                    and 0 <= capture < duration+1e-7 and min(wall, delay) >= 0 and complete > capture)
        if clock_ok:
            expected_complete = math.ceil((capture+max(.1, wall+delay)-1e-10)/PHYSICS_DT)*PHYSICS_DT
            clock_ok &= abs(complete-expected_complete) < 1e-7
            if i == 0:
                clock_ok &= abs(capture) < 1e-7
            else:
                previous = observations[i-1].get('completed_time_s')
                clock_ok &= _number(previous) and capture >= previous-1e-7
        observation_clock &= bool(clock_ok)
        candidate = row.get('candidate')
        discarded = row.get('discarded_after_episode', False)
        if not clock_ok:
            candidate_integrity = False
            continue
        if complete > duration+1e-7:
            candidate_integrity &= discarded is True and candidate is None and i == len(observations)-1
            continue
        if discarded or not isinstance(candidate, dict):
            candidate_integrity = False
            continue
        issued, expires = candidate.get('issued_at_s'), candidate.get('valid_until_s')
        c_capture = candidate.get('capture_time_s')
        integrity = (candidate.get('sequence') == row['sequence'] and all(_number(v) for v in (issued, expires, c_capture))
                     and abs(issued-complete) < 1e-7 and abs(c_capture-capture) < 1e-7
                     and expires <= capture+COMMAND_MAX_CAPTURE_AGE_S+1e-7
                     and _number(candidate.get('forward_speed')) and 0 <= candidate['forward_speed'] <= .45+1e-9
                     and type(candidate.get('valid')) is bool)
        candidate_integrity &= integrity
        if integrity:
            commands[row['sequence']] = candidate
    gate('causal_observation_clock', observation_clock)
    gate('capture_anchored_candidates', candidate_integrity)
    applied_causal = bool(ticks)
    forward_bounded = bool(ticks)
    for tick in ticks:
        request, guardian = tick.get('request', {}), tick.get('guardian', {})
        seq = tick.get('applied_command_sequence')
        speed, authorized = request.get('forward_speed'), guardian.get('forward_speed')
        forward_bounded &= (all(_number(v) for v in (speed, authorized))
                            and 0 <= authorized <= speed+1e-9 and speed <= .45+1e-9)
        if seq != request.get('sequence'):
            applied_causal = False
        if seq is None:
            applied_causal &= _number(speed) and abs(speed) <= 1e-12
        else:
            cmd = commands.get(seq)
            now = tick.get('time_s')
            applied_causal &= bool(cmd and cmd.get('valid') is True and _number(now)
                                   and now >= cmd['issued_at_s']-1e-7
                                   and now < cmd['valid_until_s']-1e-9
                                   and _number(speed) and speed <= cmd['forward_speed']+1e-9)
    gate('no_future_or_expired_command_application', applied_causal)
    gate('depth_gate_never_increases_forward_request', forward_bounded)

    valid_rows = [r for r in observations if r.get('observation', {}).get('valid') is True]
    selected_ids = {r['observation'].get('track_id') for r in valid_rows}
    fixed_id = len(selected_ids) == 1 and None not in selected_ids
    gate('fixed_target_identity', fixed_id)
    target_fraction = len(valid_rows)/len(observations) if observations else 0.
    overlaps = [_iou(r.get('observation', {}).get('bbox_xyxy'), r.get('target_truth_bbox'))
                if r.get('observation', {}).get('valid') is True and fixed_id else 0. for r in observations]
    hits = sum(value >= .5 for value in overlaps)
    hit_fraction = hits/len(observations) if observations else 0.
    tail = [r for r in observations if _number(r.get('capture_time_s')) and r['capture_time_s'] >= duration-2.-1e-7]
    tail_coverage = (len(tail) >= 2 and observations and _number(observations[-1].get('capture_time_s'))
                     and observations[-1]['capture_time_s'] >= duration-.65-1e-7)
    tail_centers, tail_depth_errors = [], []
    for row in tail:
        obs = row.get('observation', {})
        center = _array(obs.get('center_normalized'), (2,))
        z = row.get('target_surface_optical_z_m')
        tail_centers.append(abs(float(center[0])) if obs.get('valid') is True and center is not None else None)
        tail_depth_errors.append(abs(z-3.5) if obs.get('valid') is True and _number(z) and z > 0 else None)
    center_error = float(np.mean(tail_centers)) if tail_centers and all(v is not None for v in tail_centers) else None
    depth_error = float(np.mean(tail_depth_errors)) if tail_depth_errors and all(v is not None for v in tail_depth_errors) else None
    kind = spec.get('kind', 'normal')
    event = spec.get('event_s')
    metrics = dict(tick_count=len(ticks), expected_tick_count=expected, observation_count=len(observations),
        contact_count=contact_count, horizontal_translation_m=translation, horizontal_path_length_m=path_length,
        maximum_speed_m_s=float(speeds.max()) if states else None,
        maximum_tilt_rad=float(tilts.max()) if states else None,
        maximum_altitude_error_m=float(abs(positions[:, 2]-1.1).max()) if states else None,
        valid_target_observations=len(valid_rows), valid_target_fraction=target_fraction,
        selected_track_ids=sorted(selected_ids, key=str), active_overlap_hits=hits,
        active_overlap_fraction=hit_fraction, mean_active_iou=float(np.mean(overlaps)) if overlaps else None,
        tail_observation_count=len(tail), tail_mean_abs_horizontal_center=center_error,
        tail_mean_surface_standoff_error_m=depth_error,
        discarded_after_episode_count=sum(r.get('discarded_after_episode') is True for r in observations),
        guardian_reasons=dict(Counter(t.get('guardian', {}).get('reason', 'missing') for t in ticks)),
        candidate_reasons=dict(Counter((r.get('candidate') or {}).get('reason', 'discarded') for r in observations)),
        maximum_inference_wall_s=max((r['inference_wall_s'] for r in observations if _number(r.get('inference_wall_s'))), default=None))
    if episode is not None:
        gate('episode_receipt_matches_trace', episode.get('status') == 'completed'
             and episode.get('physics_ticks') == len(ticks) and episode.get('observations') == len(observations)
             and bool(states) and _state_valid(episode.get('final_state'))
             and _same_state(states[-1], episode['final_state']))
        metrics['whole_wall_s'] = episode.get('whole_wall_s')
    if kind == 'normal':
        gate('normal_genuine_translation', translation is not None and translation > .6)
        gate('normal_target_retention', target_fraction >= .85)
        gate('normal_approximate_photo_overlap', hit_fraction >= .8)
        gate('normal_complete_tail_observations', tail_coverage)
        gate('normal_tail_centered', center_error is not None and center_error < .1)
        gate('normal_tail_surface_standoff', depth_error is not None and depth_error < .6)
    elif kind == 'ablation' or spec.get('ablation') is True:
        gate('ablation_no_authorized_pursuit', bool(ticks) and all(
            _number(t.get('guardian', {}).get('forward_speed')) and abs(t['guardian']['forward_speed']) <= 1e-12 for t in ticks))
        gate('ablation_no_translation', translation is not None and translation < .04)
        gate('ablation_complete_capture_coverage', tail_coverage)
    elif kind in ('target_loss', 'depth_loss', 'stale_depth', 'delayed_inference', 'obstacle_stop'):
        valid_event = _number(event) and 0 < event < duration-2.
        gate('valid_event_interval', valid_event)
        if valid_event:
            pre = [r for r in observations if _number(r.get('capture_time_s')) and r['capture_time_s'] < event-1e-7]
            post = [r for r in observations if _number(r.get('capture_time_s')) and r['capture_time_s'] >= event-1e-7]
            pre_fraction = sum(r.get('observation', {}).get('valid') is True for r in pre)/len(pre) if pre else 0.
            pre_speeds = speeds[times < event] if states else np.array([])
            settled = speeds[times >= event+2.-1e-7] if states else np.array([])
            settled_ticks = [t for t in ticks if _number(t.get('time_s')) and t['time_s'] >= event+2.-1e-7]
            gate('event_capture_coverage', bool(pre) and bool(post))
            gate('event_prefault_target_retention', pre_fraction >= .8)
            gate('event_prefault_genuine_motion', len(pre_speeds) > 0 and float(pre_speeds.max()) > .08)
            gate('event_settled_speed', len(settled) > 0 and bool(np.all(settled < .08)))
            gate('event_tail_no_authorized_forward', bool(settled_ticks) and all(
                _number(t.get('guardian', {}).get('forward_speed')) and abs(t['guardian']['forward_speed']) <= 1e-12 for t in settled_ticks))
            after_ticks = [t for t in ticks if _number(t.get('time_s')) and t['time_s'] >= event-1e-7]
            reason_counts = Counter(t.get('guardian', {}).get('reason', 'missing') for t in after_ticks)
            if kind == 'target_loss':
                fault_seen = any(r.get('observation', {}).get('valid') is False and r.get('target_truth_bbox') is None for r in post)
                fault_seen &= any((r.get('candidate') or {}).get('reason') == 'target_lost' for r in post)
            elif kind == 'depth_loss':
                fault_seen = reason_counts['missing_depth'] > 0
            elif kind == 'stale_depth':
                fault_seen = reason_counts['stale_depth'] > 0
            elif kind == 'delayed_inference':
                fault_seen = any(_number(r.get('injected_delay_s')) and r['injected_delay_s'] > 0
                    and (r.get('candidate') or {}).get('reason') == 'stale_perception_result' for r in post)
            else:
                fault_seen = reason_counts['blocked_stopping_distance'] > 0
            gate('declared_fault_observed', fault_seen)
            metrics.update(prefault_maximum_speed_m_s=float(pre_speeds.max()) if len(pre_speeds) else None,
                           prefault_target_fraction=pre_fraction,
                           settled_maximum_speed_m_s=float(settled.max()) if len(settled) else None)
    else:
        gate('known_case_kind', False, str(kind))
    clearances = []
    for tick in ticks:
        truth = tick.get('truth_after', {})
        if truth.get('obstacle_enabled') is True:
            position = _array(tick.get('state_after', {}).get('position'), (3,))
            obstacle = _array(truth.get('obstacle_position'), (3,))
            if position is not None and obstacle is not None:
                delta = np.maximum(abs(position[:2]-obstacle[:2])-np.array([.10, .25]), 0.)
                clearances.append(float(np.linalg.norm(delta)-.32))
    metrics['minimum_planar_obstacle_surface_clearance_m'] = min(clearances) if clearances else None
    if kind == 'obstacle_stop':
        gate('obstacle_truth_coverage', len(clearances) > 0)
        gate('conservative_planar_obstacle_clearance', bool(clearances) and min(clearances) > 0.)
    return dict(name=spec.get('name', 'unnamed'), kind=kind, duration_s=duration, passed=all(gates.values()),
        outcome_scope='expected absence of pursuit with zero neural features' if kind == 'ablation' else 'controlled motor-flight fixture',
        gates=gates, failures=failures, metrics=metrics, thresholds=LIMITS,
        tracking_scope='Capture-aligned overlap against approximate projected photograph foreground; not real-person benchmark accuracy.',
        timing_scope='Measured-delay offline simulation. Discarded final inference is scored only as capture-time tracking, never as applied guidance.',
        obstacle_clearance_scope='Evaluation-only horizontal AABB surface distance minus 0.32 m vehicle radius; controller receives no truth.')


def score_episode(folder):
    folder = Path(folder)
    return score_records(_read_json(folder/'spec.json'), _read_rows(folder/'ticks.jsonl'),
                         _read_rows(folder/'observations.jsonl'), _read_json(folder/'episode.json'))


def suite_coverage(definition, execution, specs):
    """Compare saved cases with the frozen plan, including exact case specs."""
    planned = definition.get('specs') if isinstance(definition, dict) else None
    registered = isinstance(planned, list) and bool(planned) and all(isinstance(s, dict) for s in planned)
    names = [s.get('name') for s in planned] if registered else []
    valid_names = all(isinstance(name, str) and bool(name) and Path(name).name == name and name not in ('.', '..') for name in names)
    unique = valid_names and len(set(names)) == len(names)
    saved_names = [s.get('name') for s in specs]
    saved_unique = all(isinstance(n, str) for n in saved_names) and len(set(saved_names)) == len(saved_names)
    exact = (registered and unique and saved_unique and set(names) == set(saved_names)
             and all(next(saved for saved in specs if saved['name'] == spec['name']) == spec for spec in planned))
    receipt = (isinstance(execution, dict) and execution.get('complete') is True
               and execution.get('cases') == names
               and execution.get('development') == definition.get('development')) if registered else False
    return dict(passed=bool(exact and receipt), planned_case_names=names, saved_case_names=saved_names,
        missing_cases=[n for n in names if n not in saved_names],
        unexpected_cases=[n for n in saved_names if n not in names],
        exact_frozen_specs=bool(exact), execution_receipt_complete=bool(receipt),
        development=definition.get('development') if isinstance(definition, dict) else None)


def latest_completed(observations, time_s):
    """Return only a saved result that physically existed by preview time."""
    eligible = [r for r in observations if _number(r.get('completed_time_s'))
                and r['completed_time_s'] <= time_s+1e-9 and not r.get('discarded_after_episode', False)]
    return max(eligible, key=lambda row: row['completed_time_s']) if eligible else None


def _text(canvas, text, xy, scale=.60, color=(223, 231, 238), thickness=1):
    import cv2
    cv2.putText(canvas, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _fit_rgb(canvas, rgb, x, y, width, height):
    import cv2
    h, w = rgb.shape[:2]
    scale = min(width/w, height/h)
    output = cv2.resize(rgb[:, :, ::-1], (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    dx, dy = (width-output.shape[1])//2, (height-output.shape[0])//2
    canvas[y+dy:y+dy+output.shape[0], x+dx:x+dx+output.shape[1]] = output
    return scale, x+dx, y+dy


def preview_frame(folder, score, ticks, observations, time_s, cache=None):
    """Render saved views, with capture, completion and current state distinct."""
    import cv2
    folder = Path(folder)
    cache = {} if cache is None else cache
    row = latest_completed(observations, time_s)
    key = None if row is None else row['sequence']
    if cache.get('key', object()) != key:
        path = folder/'initial.npz' if row is None else folder/'observations'/f'{key:06d}.npz'
        with np.load(path, allow_pickle=False) as saved:
            cache.clear()
            cache.update(key=key, arrays={name: saved[name].copy() for name in saved.files})
    arrays = cache['arrays']
    canvas = np.full((720, 1280, 3), (28, 23, 18), np.uint8)
    label = score['name'].replace('_', ' ').upper()
    result = 'PASS' if score['passed'] else 'FAIL'
    result_color = (108, 223, 135) if score['passed'] else (93, 108, 247)
    _text(canvas, 'YOLO + FLYVIS  /  MOTOR-DRIVEN FLIGHT SIMULATION', (20, 32), .79, thickness=2)
    _text(canvas, f'{label}   {time_s:04.1f} / {score["duration_s"]:g} s', (20, 60), .60)
    _text(canvas, f'CASE {result}', (1100, 58), .65, result_color, 2)
    overview = arrays.get('overviewRGB', arrays.get('overview'))
    if overview is None:
        raise ValueError('Saved observation is missing completion-time overviewRGB')
    _fit_rgb(canvas, overview, 20, 85, 640, 480)
    overview_time = 0. if row is None else row['completed_time_s']
    _text(canvas, f'Overview held from t={overview_time:.3f} s', (20, 590), .55)
    if row is None:
        _text(canvas, 'First perception result is still computing', (730, 240), .63)
        _text(canvas, 'Hover under conventional stabilization', (730, 277), .55)
    else:
        rgb = arrays['rgb']
        scale, x, y = _fit_rgb(canvas, rgb, 790, 100, 391, 391)
        obs = row.get('observation', {})
        _text(canvas, f'YOLO target #{obs.get("track_id")}: {"TRACKED" if obs.get("valid") is True else "LOST"}',
              (750, 84), .58, (110, 238, 127) if obs.get('valid') is True else (93, 108, 247))
        box = _array(obs.get('bbox_xyxy'), (4,))
        if obs.get('valid') is True and box is not None:
            x0, y0, x1, y1 = (box*scale+np.array([x, y, x, y])).round().astype(int)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (110, 238, 127), 2)
            _text(canvas, f'person #{obs.get("track_id")}', (x0, max(96, y0-7)), .50, (110, 238, 127))
        _text(canvas, f'Camera captured t={row["capture_time_s"]:.3f} s', (750, 520), .55)
        _text(canvas, f'Result released t={row["completed_time_s"]:.3f} s', (750, 547), .55)
        neural = 'present' if row.get('neural_valid', row.get('neuralvalid', False)) else 'absent'
        _text(canvas, f'Flyvis target evidence: {neural} / 45,669 neurons', (730, 579), .50)
    tick_times = np.asarray([t['time_s'] for t in ticks])
    index = int(np.searchsorted(tick_times, time_s+1e-9, side='right')-1)
    tick = ticks[max(0, min(index, len(ticks)-1))]
    state = tick['state_before']
    speed = float(np.linalg.norm(state['velocity']))
    altitude = state['position'][2]
    guardian = tick['guardian']
    _text(canvas, f'Current state t={state["time_s"]:.3f} s  |  speed {speed:.3f} m/s  |  altitude {altitude:.3f} m', (20, 625), .61)
    _text(canvas, f'Guidance: {tick["request"]["reason"]}  |  authorized forward {guardian["forward_speed"]:.3f} m/s', (20, 651), .55)
    _text(canvas, f'Depth guardian: {guardian["reason"]}', (20, 675), .53)
    _text(canvas, 'Camera > YOLO target > Flyvis visual network > neural readout > guidance > depth stop > autopilot > 4 motors', (20, 698), .49, (163, 188, 199))
    _text(canvas, 'Offline physics fixture / photograph target / ideal state + depth / no physical aircraft', (20, 717), .43, (151, 159, 170))
    return canvas


def export_suite(root_output):
    """Finalize a saved run once; never overwrite an existing report or preview."""
    import cv2
    root = Path(root_output)
    output, summary_path = root/'flight_preview.mp4', root/'summary.json'
    raw = root/'flight_preview.raw.mp4'
    if any(path.exists() for path in (output, summary_path, raw)):
        raise FileExistsError('Preview/report already exists; use a new output run or preserve and remove the incomplete finalization explicitly.')
    folders = [p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
               and (p/'spec.json').is_file() and (p/'ticks.jsonl').is_file()]
    if not folders:
        raise ValueError('No saved flight cases found')
    order = {name: i for i, name in enumerate(CASE_ORDER)}
    folders.sort(key=lambda path: (order.get(path.name, len(order)), path.name))
    definition = _read_json(root/'definition.json') if (root/'definition.json').is_file() else {}
    execution = _read_json(root/'execution_complete.json') if (root/'execution_complete.json').is_file() else {}
    coverage = suite_coverage(definition, execution, [_read_json(folder/'spec.json') for folder in folders])
    scores = [score_episode(folder) for folder in folders]
    expected_frames = sum(round(score['duration_s']*FPS) for score in scores)
    cv2.setNumThreads(2)
    writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError('Unable to open flight preview writer')
    try:
        for folder, score in zip(folders, scores):
            ticks, rows = _read_rows(folder/'ticks.jsonl'), _read_rows(folder/'observations.jsonl')
            if not ticks:
                raise ValueError('Cannot render an empty flight episode')
            cache = {}
            for index in range(round(score['duration_s']*FPS)):
                writer.write(preview_frame(folder, score, ticks, rows, index/FPS, cache))
    finally:
        writer.release()
    subprocess.run(['ffmpeg', '-v', 'error', '-nostdin', '-n', '-threads', '2', '-i', str(raw),
                    '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '21',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)],
                   check=True, env=dict(os.environ), timeout=240)
    capture = cv2.VideoCapture(str(output))
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape != (720, 1280, 3):
                raise ValueError('Unexpected encoded preview dimensions')
            frames += 1
    finally:
        capture.release()
    if frames != expected_frames:
        raise ValueError(f'Incomplete encoded preview: {frames} of {expected_frames} frames')
    summary = dict(passed=coverage['passed'] and all(score['passed'] for score in scores),
        suite_coverage=coverage, case_count=len(scores),
        passing_cases=sum(score['passed'] for score in scores), cases=scores,
        preview=dict(path=output.name, width=1280, height=720, fps=FPS,
                     decoded_frames=frames, duration_s=frames/FPS),
        scope='Offline six-DOF motor-flight fixture with photograph target, ideal state sensors and rendered depth. No physical flight or achieved realtime claim.')
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    raw.unlink()
    return dict(video=str(output.resolve()), summary=str(summary_path.resolve()), report=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    result = export_suite(args.run)
    print(json.dumps({key: value for key, value in result.items() if key != 'report'}, indent=2))
    return 0 if result['report']['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
