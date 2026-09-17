"""One interactive, motor-driven simulation; never connects to an aircraft."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np

from experiments.flight_contracts import (ROOT, PHYSICS_DT, ALTITUDE_M,
    MAX_SPEED_M_S, COMMAND_MAX_CAPTURE_AGE_S, MAX_OBSERVATION_AGE_S, VEHICLE_RADIUS_M)
from experiments.mantis_studio_config import validate_config


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def atomic_json(path, value):
    data = json.dumps(plain(value), allow_nan=False, separators=(',', ':')).encode()
    if len(data) > 128*1024:
        raise ValueError('Interactive metadata exceeds its bounded size')
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def motion_brake_scale(result, now_s):
    """Explicit experimental speed reduction only; not a collision detector.

    Frozen transfer-unit threshold, not fitted to these flights. Image motion
    includes ego-motion and can cause unnecessary slowing. Unknown data holds.
    """
    if not result or result.get('valid') is not True:
        return 0.
    times = [now_s, result.get('capture_time_s'), result.get('response_time_s'), result.get('available_time_s')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in times):
        return 0.
    if now_s < max(times[2:]):
        return 0.
    age = now_s-times[1]
    magnitude = result.get('neural', {}).get('rms_decoder_magnitude')
    if age < 0 or age > MAX_OBSERVATION_AGE_S or isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)) or not np.isfinite(magnitude) or magnitude < 0:
        return 0.
    return float(np.clip(1.-magnitude/.10, .2, 1.))


def selectable_preview_is_fresh(selection, now_s):
    capture = selection.get('capture_time_s')
    return (isinstance(capture, (int, float)) and not isinstance(capture, bool)
            and np.isfinite([capture, now_s]).all() and 0 <= now_s-capture <= MAX_OBSERVATION_AGE_S
            and any(person.get('selectable') for person in selection.get('people', [])))


def run_session(folder, config):
    # Heavy imports/load occur in this child process, with native GL on its main
    # thread. The HTTP server stays responsive while models initialize.
    folder = Path(folder).resolve()
    config = validate_config(config)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    import cv2
    from experiments.flight_autopilot import Autopilot
    from experiments.flight_guidance import release, active_request, physical_completion
    from experiments.flight_safety import DepthGuardian
    from experiments.mantis_vision import MantisVision
    from experiments.mantis_studio_vision import StudioPersonRetryDetector
    from experiments.mantis_arena import ArenaWorld, evaluate_selection
    from experiments.mantis_selection import SelectionGuard
    from experiments.mantis_navigation import DetourNavigator
    from experiments.mantis_comparison import AlphaBeta
    from experiments.mantis_flight import selected_candidate
    from experiments.mantis_recording import SessionRecorder
    cv2.setNumThreads(2)
    state_path = folder/'state.json'
    atomic_json(state_path, dict(phase='loading', message='Loading YOLO and Flyvis', config=config))
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *args: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *args: stop.__setitem__(0, True))
    world = recorder = None
    statistics = dict(observations=0, wrong_person_observations=0, evaluated_selected_observations=0,
                      unscorable_selected_observations=0, holds=0, motion_speed_reductions=0,
                      contacts=0, actual_yolo_calls=0, actual_flyvis_observations=0,
                      detector_retry_frames=0, detector_retry_person_frames=0,
                      selected_actor_reference=None, detours_completed=0,
                      detour_completion_events=[], detour_events_omitted=0, positive_forward_ticks=0,
                      minimum_obstacle_hull_clearance_m=None,
                      motion_observations=0, motion_fresh_observations=0,
                      motion_gap_resets=0, motion_deadline_misses=0,
                      motion_valid_brake_ticks=0, motion_valid_speed_reductions=0,
                      motion_unavailable_hold_ticks=0, motion_depth_approved_reductions=0,
                      motion_brake_pairs=[])
    started = time.monotonic()
    phase, failure = 'loading', None
    last_telemetry, motion_result = {}, None
    seen_control = -1
    seen_selection = -1
    pending_neural_reset = False
    ever_selected = False
    command = None
    latest_depth = None
    final_safety = dict(forward_speed=0., reason='initializing')
    navigation = dict(phase='holding', reason='selection_required', forward_speed=0., yaw_target=0.)
    intended_actor = None
    latest_projections = []
    tick = 0
    stopping_until = None
    operation = 'run'
    message = None
    try:
        vision = MantisVision()
        vision.detector = StudioPersonRetryDetector(vision.detector)
        vision.pipeline.detector = vision.detector
        guard = SelectionGuard()
        vision.bridge = guard
        motion = None
        if config['motion_mode'] != 'off':
            from experiments.mantis_motion import RawMotionExperiment
            motion = RawMotionExperiment(ROOT/'models/flyvis_0000_000.manifest.json')
        world = ArenaWorld()
        pilot, safety, planner, angular_filter = Autopilot(), DepthGuardian(), DetourNavigator(), AlphaBeta()
        model_hash = hashlib.sha256((ROOT/'models/flyvis_0000_000.manifest.json').read_bytes()).hexdigest()
        source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT/'experiments').glob('mantis_*.py'))}
        recorder = SessionRecorder(folder/'capture', profile=config['recording'],
            max_bytes=config['max_recording_mb']*1024**2, provenance=dict(
                name='Mantis Studio', version='3.0-experimental', config=config,
                sources=source_hashes, model_manifest_sha256=model_hash,
                physical_control_authority=False, real_time=False,
                selection='Camera detections and fixed clothing anchor; truth used only for scoring',
                detection_retry='One same-frame horizontal-flip pass only after zero primary persons; unchanged thresholds and total deadline',
                motion_brake='Optional uncalibrated image-motion speed reduction; no steering; depth gate remains final'))

        def publish(extra=None):
            state = world.state()
            payload = dict(phase=phase, message=message, config=config, sim_s=float(state.time_s),
                wall_s=time.monotonic()-started, selection=guard.status(), statistics=statistics,
                navigation=navigation, safety=final_safety,
                vehicle=dict(position=state.position.tolist(), speed_m_s=float(np.linalg.norm(state.velocity)),
                             yaw_rad=state.yaw, motor_forces=state.motor_forces.tolist()),
                recording=dict(profile=recorder.profile, bytes=recorder.bytes_written,
                               max_bytes=recorder.budget.limit), motion=motion_result,
                frame=last_telemetry.get('frame'), neural_features=last_telemetry.get('neural_features'),
                guidance=last_telemetry.get('guidance'), control_revision=seen_control,
                selection_revision=seen_selection, simulation_only=True)
            payload.update(extra or {})
            atomic_json(state_path, payload)
            return plain(payload)

        def controls():
            nonlocal config, seen_control, seen_selection, operation, pending_neural_reset
            nonlocal command, intended_actor, ever_selected, message, angular_filter
            if not (folder/'control.json').is_file():
                return
            current = json.loads((folder/'control.json').read_text())
            revision = current.get('revision', 0)
            if revision <= seen_control:
                return
            config = validate_config(current.get('settings', {}), base=config, live=True)
            operation = current.get('operation', 'run')
            if operation not in ('run', 'pause', 'stop'):
                raise ValueError('Invalid operation')
            if operation != 'run':
                command = None
            selection = current.get('selection')
            if selection and selection['revision'] > seen_selection:
                seen_selection = selection['revision']
                try:
                    if selection.get('clear'):
                        guard.clear()
                        intended_actor = None
                    else:
                        guard.select(selection['track_id'], selection['sequence'], now_s=world.data.time)
                        person = next(p for p in guard.status()['people'] if p['track_id'] == selection['track_id'])
                        evaluation = evaluate_selection(dict(person, valid=True), latest_projections)
                        intended_actor = evaluation['actor_id']
                        statistics['selected_actor_reference'] = intended_actor
                        ever_selected = True
                    command = None
                    pending_neural_reset = True
                    angular_filter = AlphaBeta()
                    planner.reset()
                    message = None
                except ValueError as exc:
                    message = str(exc)
            seen_control = revision

        def fixture():
            world.update_scene(float(world.data.time), trajectory=config['scenario'],
                               target_speed=config['target_speed'], obstacle=config['scenario'] == 'detour')

        def advance(completion, braking=False):
            nonlocal tick, latest_depth, navigation, final_safety
            end_tick = round(completion/PHYSICS_DT)
            while tick < end_tick:
                state = world.state()
                motion_pair_request = None
                request = active_request(None if braking else command, state)
                if request['sequence'] is not None:
                    z = command['estimate']['surface_optical_z_m']
                    request['range_speed'] = min(MAX_SPEED_M_S, max(0., .6*(z-config['follow_distance_m'])))
                    request['forward_speed'] = min(request['forward_speed'], request['range_speed'])
                if tick % 10 == 0 or latest_depth is None:
                    fixture()
                    latest_depth = world.capture(safety=True)
                    navigation = planner.update(latest_depth, state, request, enabled=config['detours'])
                    if planner.completed > statistics['detours_completed']:
                        if len(statistics['detour_completion_events']) < 16:
                            statistics['detour_completion_events'].append(dict(
                                sim_s=float(state.time_s), position=state.position.tolist(),
                                selected_track=guard.status().get('track_id'),
                                command_sequence=request['sequence'],
                                source_capture_time_s=command['capture_time_s'] if command else None))
                        else:
                            statistics['detour_events_omitted'] += 1
                        statistics['detours_completed'] = planner.completed
                if request['sequence'] is None:
                    forward, yaw = 0., state.yaw
                else:
                    forward, yaw = navigation['forward_speed'], navigation['yaw_target']
                    if config['motion_mode'] == 'brake':
                        scale = motion_brake_scale(motion_result, state.time_s)
                        if scale > 0:
                            statistics['motion_valid_brake_ticks'] += 1
                            if forward > 0 and scale < .999:
                                statistics['motion_valid_speed_reductions'] += 1
                        elif forward > 0:
                            statistics['motion_unavailable_hold_ticks'] += 1
                        if forward > 0 and scale < .999:
                            statistics['motion_speed_reductions'] += 1
                            if scale > 0 and tick % 10 == 0:
                                motion_pair_request = float(forward)
                        forward *= scale
                final_safety = safety.check(latest_depth, state, forward, state.time_s)
                if motion_pair_request is not None:
                    # Evaluator-only paired depth query. Neither this result nor
                    # the comparison can alter the actual motor command below.
                    reference = safety.check(latest_depth, state, motion_pair_request, state.time_s)
                    if reference['forward_speed'] > final_safety['forward_speed']+1e-6:
                        statistics['motion_depth_approved_reductions'] += 1
                        if len(statistics['motion_brake_pairs']) < 32:
                            statistics['motion_brake_pairs'].append(dict(
                                sim_s=float(state.time_s), motion_capture_time_s=motion_result['capture_time_s'],
                                motion_available_time_s=motion_result['available_time_s'],
                                command_sequence=request['sequence'], command_capture_time_s=command['capture_time_s'],
                                depth_capture_time_s=float(latest_depth.capture_time_s),
                                depth_approved_unscaled_speed=reference['forward_speed'],
                                actual_forward_speed=final_safety['forward_speed'],
                                scale=scale, evaluator_only=True))
                statistics['positive_forward_ticks'] += final_safety['forward_speed'] > 0
                world.step(pilot.command(state, final_safety['forward_speed'], yaw, ALTITUDE_M))
                truth = world.truth()
                statistics['contacts'] += bool(truth['contacts'])
                # Evaluation only, after motor authority has been applied. This
                # known fixture geometry never feeds perception or navigation.
                if truth['obstacle_enabled']:
                    delta = np.maximum(np.abs(world.state().position[:2]
                        - np.asarray(truth['obstacle_position'][:2])) - [.10, .25], 0.)
                    clearance = float(np.linalg.norm(delta)-VEHICLE_RADIUS_M)
                    previous = statistics['minimum_obstacle_hull_clearance_m']
                    statistics['minimum_obstacle_hull_clearance_m'] = clearance if previous is None else min(previous, clearance)
                tick += 1
            statistics['detours_completed'] = planner.completed

        while True:
            controls()
            if stopping_until is not None or stop[0] or operation == 'stop':
                stopping_until = stopping_until or float(world.data.time)+2.
                phase = 'stopping'
                command = None
                if world.data.time >= stopping_until-1e-8:
                    break
            elif world.data.time >= config['duration_s']-1e-8:
                break
            elif operation == 'pause' or (statistics['observations'] and not ever_selected
                    and selectable_preview_is_fresh(guard.status(), float(world.data.time))):
                phase = 'paused' if operation == 'pause' else 'awaiting-selection'
                publish()
                time.sleep(.08)
                continue
            else:
                phase = 'running'
            fixture()
            frame, overview = world.capture(), world.overview()
            latest_projections = world.evaluation_people(frame)
            guard.observe_frame(frame)
            begin = time.perf_counter()
            if pending_neural_reset:
                vision.brain.reset()
                pending_neural_reset = False
            vision.detector.last_receipt = None
            bundle = vision.process(frame)
            detection_receipt = vision.detector.last_receipt
            # Obtain the explicit selection preview before optional research
            # work. On a slow CPU that work can exceed a camera deadline; it
            # must not consume the entire run before the user can select.
            pending_motion = motion.step(frame.rgb, float(frame.capture_time_s)) if motion is not None and ever_selected else None
            candidate = selected_candidate(config['method'], bundle, frame, angular_filter)
            elapsed = time.perf_counter()-begin
            completion = physical_completion(frame.capture_time_s, elapsed)
            advance(completion, braking=phase == 'stopping')
            if pending_motion is not None:
                motion_result = dict(pending_motion, available_time_s=completion)
                statistics['motion_observations'] += 1
                statistics['motion_gap_resets'] += bool(motion_result['gap_reset'])
                fresh_motion = motion_brake_scale(motion_result, completion) > 0
                statistics['motion_fresh_observations'] += fresh_motion
                statistics['motion_deadline_misses'] += completion-frame.capture_time_s > MAX_OBSERVATION_AGE_S
            command = release(candidate, world.state(), completion, bundle['sequence'])
            if command['valid']:
                # Reuse the measured depth; interactive standoff never alters
                # bearing inference, capture expiry or the final depth gate.
                command['forward_speed'] = min(MAX_SPEED_M_S, max(0., .6*(candidate['surface_optical_z_m']-config['follow_distance_m'])))
            if phase == 'stopping':
                command = None
            observation = bundle['observation']
            evaluation = evaluate_selection(observation, latest_projections, intended_actor)
            if observation['valid']:
                if evaluation['wrong_person'] is None:
                    statistics['unscorable_selected_observations'] += 1
                else:
                    statistics['evaluated_selected_observations'] += 1
                    statistics['wrong_person_observations'] += bool(evaluation['wrong_person'])
            else:
                statistics['holds'] += 1
            statistics['observations'] += 1
            statistics['actual_yolo_calls'] += detection_receipt['backend_calls'] if detection_receipt else 0
            statistics['detector_retry_frames'] += bool(detection_receipt and detection_receipt['retry_attempted'])
            statistics['detector_retry_person_frames'] += bool(detection_receipt and detection_receipt['used_retry'])
            statistics['actual_flyvis_observations'] += 1
            last_telemetry = dict(frame=dict(sequence=bundle['sequence'], capture_time_s=float(frame.capture_time_s),
                width=frame.rgb.shape[1], height=frame.rgb.shape[0]),
                selection=guard.status(), guidance=plain(candidate), navigation=plain(navigation),
                safety=plain(final_safety), motor_forces=world.motor_forces.tolist(),
                position=world.state().position.tolist(), speed_m_s=float(np.linalg.norm(world.state().velocity)),
                evaluation=evaluation, neural_features=plain(bundle['neural']['features']),
                motion=motion_result, inference_wall_s=elapsed, completed_time_s=completion,
                detector_status=bundle['detections'].get('status'),
                detector_reason=bundle['detections'].get('reason'), detector_receipt=detection_receipt)
            for name, image in [('camera', frame.rgb), ('overview', overview)]:
                okay, jpeg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not okay:
                    raise RuntimeError('Cannot encode live camera')
                filename = name+'-'+str(bundle['sequence'])+'.jpg'
                temporary = folder/(filename+'.tmp')
                temporary.write_bytes(jpeg.tobytes())
                temporary.replace(folder/filename)
            raw = dict(depth_m=np.where(np.isfinite(frame.depth_m), frame.depth_m, 0.),
                       depth_valid=np.isfinite(frame.depth_m), activity=bundle['neural']['activity'],
                       retina=bundle['neural']['retina'], features=bundle['neural']['features'])
            if not recorder.append(completion, frame.rgb, overview, plain(last_telemetry), raw=raw):
                phase = 'budget-exhausted'
                command = None
                advance(float(world.data.time)+2., braking=True)
                break
            publish()
            for name in ('camera', 'overview'):
                for old in folder.glob(name+'-*.jpg'):
                    index = old.stem.rsplit('-', 1)[-1]
                    if index.isdigit() and int(index) < bundle['sequence']-1:
                        old.unlink()
        phase = 'interrupted' if stopping_until is not None else 'completed' if phase != 'budget-exhausted' else phase
    except Exception as exc:
        failure = dict(type=type(exc).__name__, reason=str(exc))
        phase = 'error'
        raise
    finally:
        receipt = None
        final_vehicle = None
        if world is not None:
            state = world.state()
            final_vehicle = dict(position=state.position.tolist(),
                speed_m_s=float(np.linalg.norm(state.velocity)), yaw_rad=state.yaw,
                motor_forces=state.motor_forces.tolist())
        if recorder is not None:
            summary = dict(status=phase,
                duration_s=float(world.data.time), statistics=statistics, failure=failure,
                wall_s=time.monotonic()-started, physical_flight=False, real_time=False,
                final_vehicle=final_vehicle, final_safety=plain(final_safety))
            if failure:
                recorder.error = str(failure)
            receipt = recorder.finish(summary)
            if receipt['status'] == 'error':
                phase = 'error'
                failure = failure or dict(type='RecordingError', reason=receipt['error'])
            elif receipt['status'] == 'budget-exhausted':
                phase = 'budget-exhausted'
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        atomic_json(state_path, dict(previous, phase=phase, failure=failure, receipt=receipt,
            sim_s=float(world.data.time) if world is not None else 0., statistics=statistics,
            vehicle=final_vehicle, safety=plain(final_safety)))
        if world is not None:
            world.close()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    run_session(folder, json.loads((folder/'config.json').read_text()))


if __name__ == '__main__':
    main()
