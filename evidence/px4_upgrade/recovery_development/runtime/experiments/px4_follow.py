"""Actual local PX4 takeoff / selected YOLO-Flyvis follow / stale hold / land.

Run --smoke first for autopilot-only transport verification. All evidence is
local and bounded. SIH has no collisions with the separately rendered room.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import time
import uuid

import numpy as np

from experiments.flight_guidance import active_request, release
from experiments.flight_safety import DepthGuardian
from experiments.px4_camera import SihCamera, VisionWorker, estimated_state
from experiments.px4_recording import AsyncRecorder
from flybrain_sim.px4_sih import ALTITUDE, ROOT, OwnedSih, SihLink, sha, ESTIMATOR_PROFILES
from flybrain_sim.px4_sitl import SitlError


class ResultGate:
    def __init__(self, session, selected_track):
        self.session, self.selected_track = session, selected_track
        self.sequence, self.capture = -1, -math.inf

    def accept(self, packet, state, now):
        candidate = packet['candidate']
        sequence, capture = packet['sequence'], packet['capture_time_s']
        if (packet['session'] != self.session or type(sequence) is not int
                or sequence <= self.sequence or not math.isfinite(capture)
                or capture <= self.capture or not capture <= packet['completed_s'] <= now
                or candidate['capture_time_s'] != capture):
            raise SitlError('Invalid perception session, order or timestamps')
        self.sequence, self.capture = sequence, capture
        selection, observation = packet['selection'], packet['observation']
        if (selection['track_id'] != self.selected_track or selection['held']
                or not observation['valid'] or observation['track_id'] != self.selected_track):
            candidate = dict(candidate, valid=False, reason='selected_person_unavailable')
        # Includes model execution, output serialization/queue and parent work.
        return release(candidate, replace(state, time_s=now), now, sequence)


class StudioStop(Exception):
    """Cooperative cancellation; the owned aircraft still goes through LAND."""


def following_metrics(rows, observations):
    follow = [r for r in rows if r['phase'] == 'follow']
    advancing = [r for r in follow if r['sent_speed'] > .05]
    valid = [o for o in observations if o['command']['valid']]
    displacement = (float(np.linalg.norm(np.array(follow[-1]['position'][:2])
                         - np.array(follow[0]['position'][:2]))) if len(follow) > 1 else 0.)
    depths = [o['packet']['candidate']['surface_optical_z_m'] for o in valid]
    depth_reduction = depths[0]-depths[-1] if len(depths) >= 2 else 0.
    return dict(fresh_neural_commands=len(valid), advancing_ticks=len(advancing),
                follow_displacement_m=displacement, selected_depth_reduction_m=depth_reduction)


def following_checks(metrics):
    return dict(fresh_neural_commands=metrics['fresh_neural_commands'] >= 5,
        depth_approved_forward_commands=metrics['advancing_ticks'] >= 10,
        translated_at_least_30cm=metrics['follow_displacement_m'] >= .3,
        selected_depth_reduced_20cm=metrics['selected_depth_reduction_m'] >= .2)


def score(rows, observations, events, smoke=False):
    took_off = any(e['event'] == 'stable_takeoff' for e in events)
    landed = any(e['event'] == 'landed_disarmed' for e in events)
    metrics = following_metrics(rows, observations)
    hold = [r for r in rows if r['phase'] == 'stale_hold' and r['stall_elapsed_s'] >= 1.]
    late_hold = hold[-20:]
    stalls = [e for e in events if e['event'] == 'vision_stall_started']
    expired = [r for r in rows if stalls and r.get('sent_at_s', -math.inf) >= stalls[0]['expiry_s']]
    checks = dict(stable_takeoff=took_off, landed_disarmed=landed)
    if not smoke:
        checks.update(**following_checks(metrics),
            advancing_at_stall=bool(stalls) and stalls[0].get('sent_speed', 0.) > .1,
            actual_worker_stall=any(e['event']=='worker_stall_injected' for e in events),
            delayed_output_rejected=any(o['command']['reason']=='stale_perception_result'
                and o['packet']['completed_s']-o['packet']['capture_time_s'] >= 4 for o in observations),
            stale_command_expired=bool(expired) and all(r['sent_speed'] == 0 for r in expired),
            stale_hold_slowdown=len(late_hold) >= 10
                and max(r['horizontal_speed'] for r in late_hold) < .1)
    return dict(passed=all(checks.values()), checks=checks, **metrics, hold_ticks=len(hold))


def run(folder, smoke=False, selected_track=1, low_noise_sensors=False, *,
        estimator_profile='stock', hover_seconds=3., method='neural',
        trajectory='stationary', target_speed=.1, duration_s=60.,
        mission=False, faults=False, recovery=False, studio=None):
    if studio is not None:
        mission = True
        duration_s = studio.config['duration_s']
        target_speed = studio.config['target_speed']
        selected_track = None
    if (method not in ('neural', 'direct', 'filtered') or trajectory not in
            ('stationary', 'walk', 'crossing', 'occlusion') or not 2 <= duration_s <= 120
            or not 0 <= target_speed <= .5 or recovery and trajectory != 'stationary'):
        raise ValueError('Invalid mission configuration')
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    session = str(uuid.uuid4())
    source_files = [p for directory in ('experiments', 'perception', 'flybrain_sim')
                    for p in (ROOT/directory).glob('*.py')]
    source_files += [ROOT/'scripts/build_px4_sih.py', ROOT/'integrations/px4/sih.px4board',
                    ROOT/'models/flyvis_0000_000.manifest.json']
    for source in source_files:
        destination = folder/'runtime'/source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    sensor_profile = '1% stock GPS/baro/mag/IMU noise' if low_noise_sensors else 'stock sensor noise'
    provenance = dict(session=session, px4_sensor_profile=sensor_profile,
        estimator_profile=estimator_profile, estimator_parameters=ESTIMATOR_PROFILES[estimator_profile],
        mission_config=dict(method=method, trajectory=trajectory, target_speed=target_speed,
            duration_s=duration_s, faults=faults, recovery=recovery,
            scenario_phase='starts at stable takeoff using SIH source time',
            noise_seed='upstream SIH srand(1234); asynchronous sensor scheduling may vary'),
        source_sha256={str(p.relative_to(ROOT)): sha(p)
        for p in sorted(source_files)}, selection=dict(track_id=selected_track,
        policy='fixed observed track selected once; no truth or automatic replacement'),
        acceptance='takeoff; >=5 fresh neural results; >=10 advancing ticks; >=.3m movement; '
        '>=.2m selected-depth reduction; reach all following milestones before active >.1m/s stall '
        'between 4s and 18s in the 24s mission; '
        '4s worker delay rejected; zero speed after capture expiry; late speed <.1m/s; landed/disarmed')
    if mission:
        provenance['acceptance'] = ('Separate safe-flight and tracking qualification: stable takeoff, '
            'complete declared mission, fresh depth-approved sends, land/disarm; >=80% of declared '
            'window with fresh selected observations and uniquely confirmed original identity; '
            'zero wrong-person observations. Recovery requires every independent proof subcheck.')
    (folder/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    events, rows, observations, takeoff_samples = [], [], [], []
    lease = link = worker = camera = recorder = None
    failure = None
    armed_requested = False
    epoch = time.monotonic()
    follow_start = mission_end = None
    pending_selection = None
    selection_revision = -1
    latest_preview = None
    latest_selection_action = None
    last_studio_publish = -math.inf
    command = None
    def control():
        nonlocal pending_selection, selection_revision, command
        if studio is not None:
            request = studio.control()
            if request['operation'] == 'stop':
                raise StudioStop('Stop requested from Studio')
            selection = request.get('selection')
            if selection is not None and selection['revision'] > selection_revision:
                selection_revision = selection['revision']
                pending_selection = ({'clear': True} if selection.get('clear') else
                    {key: selection[key] for key in ('track_id', 'sequence')})
                command = None
    def publish(phase, message, state=None, *, packet=None, images=None, depth=None):
        nonlocal last_studio_publish, latest_preview, latest_selection_action
        if studio is None:
            return
        now = time.monotonic()
        if packet is None and now-last_studio_publish < .25:
            return
        if packet is not None:
            latest_preview = packet
            if packet.get('selection_action') is not None:
                latest_selection_action = packet['selection_action']
        snapshot = dict(phase=phase, message=message, wall_s=now-epoch,
            sim_s=0. if follow_start is None else now-follow_start,
            vehicle=None if state is None else dict(position=state.position.tolist(),
                speed_m_s=float(np.linalg.norm(state.velocity)), yaw_rad=state.yaw),
            safety=depth or {}, navigation=dict(mode='hold' if command is None else command['reason']),
            neural_features=[] if latest_preview is None else latest_preview['neural_features'],
            selection_request=dict(revision=selection_revision, pending=pending_selection is not None),
            selection_action=latest_selection_action,
            selection=dict(people=[],track_id=None,sequence=None,reason='selection_required')
                if latest_preview is None else latest_preview['selection'],
            frame=None if latest_preview is None else dict(sequence=latest_preview['sequence'],
                capture_time_s=latest_preview['capture_time_s']))
        if latest_preview is not None:
            snapshot['frame_selection'] = (latest_preview['selection']
                if latest_preview['selection'].get('sequence') == latest_preview['sequence'] else
                dict(sequence=latest_preview['sequence'], capture_time_s=latest_preview['capture_time_s'],
                     people=[], track_id=None, reason='unavailable'))
        studio.publish(snapshot, *(images or (None, None)))
        last_studio_publish = now
    def event(name, **values):
        value = dict(event=name, elapsed_s=time.monotonic()-epoch, **values)
        events.append(value)
        try:
            print(json.dumps(value), flush=True)
            publish(name, name.replace('_', ' '))
        except Exception as exc:
            # Reporting must never interrupt LAND or resource cleanup.
            events.append(dict(event='studio_status_unavailable', error=str(exc)))
    try:
        if not smoke:
            worker = VisionWorker(session, selected_track, method=method, trajectory=trajectory)
            deadline = time.monotonic()+180
            while True:
                control()
                publish('loading', 'Loading detector and guidance before arming')
                result = worker.receive()
                if result is not None:
                    if result['kind'] != 'ready':
                        raise SitlError('Unexpected vision startup receipt')
                    event('vision_ready', model=result['model'])
                    break
                if time.monotonic() > deadline:
                    raise SitlError('Vision startup timed out before arming')
                time.sleep(.05)
        lease = OwnedSih(folder / 'px4', low_noise_sensors=low_noise_sensors,
                         estimator_profile=estimator_profile)
        link = SihLink(lease)
        link.verify_simulator()
        for name, value in ESTIMATOR_PROFILES[estimator_profile].items():
            if name == 'EKF2_REQ_GPS_H':
                # Setting a compiled default before rcS's set-default loses it.
                # This readiness duration is runtime-updatable, before arming.
                link.set_parameter(name, value, integer=False)
            if abs(link.get_parameter(name)-value) > 1e-4:
                raise SitlError('Estimator startup override did not apply: '+name)
        event('owned_px4_verified', pid=lease.process.pid, build=lease.receipt,
              px4_sensor_profile=sensor_profile)
        for message_id, hz in ((31, 40), (32, 40), (230, 10), (1, 2), (245, 5)):
            link.command(511, (message_id, 1e6/hz))
        for name, value, integer in [('COM_OBL_RC_ACT', 4, True), ('COM_OF_LOSS_T', .5, False),
                                      ('COM_RC_IN_MODE', 4, True), ('SDLOG_MODE', -1, True)]:
            link.set_parameter(name, value, integer=integer)
        started = time.monotonic()
        next_report = started
        healthy_since = None
        while True:
            control()
            link.poll()
            now = time.monotonic()
            error = link.health_error(now)
            healthy_since = (healthy_since if healthy_since is not None else now) if not error else None
            if not error and link.truth is not None:
                try:
                    link.clock.capture_bound(link.truth_source, now, .1)
                    if not estimator_profile.endswith('_settled') or now-healthy_since >= 10:
                        break
                except SitlError:
                    pass
            if now >= next_report:
                event('waiting_for_estimator', reason=error, status=list(link.last_status))
                next_report = now+10
            if now-started > 90:
                raise SitlError('PX4 estimator/truth not ready: '+error)
            time.sleep(.025)
        origin = np.asarray(link.telemetry.position_ned)
        truth_origin = (link.truth.lat*1e-7, link.truth.lon*1e-7, link.truth.alt/1000)
        home = origin + [0., 0., -ALTITUDE]
        if not smoke:
            camera = SihCamera(link.truth, epoch, trajectory=trajectory, target_speed=target_speed)
            # Allocate native graphics contexts before arming. These warmup
            # images are not submitted as observations or used as depth.
            camera.world.capture(safety=True)
            camera.world.capture()
            camera.world.overview()
            from experiments.mantis_recording import SessionRecorder
            recorder = AsyncRecorder(SessionRecorder(folder/'recording' if studio is None else
                studio.recording_folder, fps=10, max_bytes=(32 if studio is None else
                studio.config['max_recording_mb'])*1024**2,
                provenance=dict(backend='patched PX4 SIH / MuJoCo render only',
                    build=lease.receipt, session=session, selected_track=selected_track,
                    neural_clock='sampled target cue; .1 model seconds per observation',
                    collision_physics=False, px4_sensor_profile=sensor_profile,
                    sensor_model='ideal RGB/depth; estimated camera pose')))
            link.wait_for(lambda: not link.health_error(time.monotonic()), 10)

        def prime():
            control()
            error = link.health_error(time.monotonic())
            if error:
                raise SitlError(error)
            link.setpoint(home, position=True, yaw_ned=0.)
        prime_start = time.monotonic()
        link.wait_for(lambda: time.monotonic()-prime_start >= 1.2, 3, prime)
        link.command(176, (1, 6, 0), prime)
        armed_requested = True
        link.command(400, (1,), prime)
        link.wait_for(lambda: link.telemetry.armed and link.telemetry.offboard, 4, prime)
        event('armed_offboard')
        started, stable_since = time.monotonic(), None
        last_takeoff_sample = last_takeoff_report = -math.inf
        while True:
            link.poll()
            prime()
            now = time.monotonic()
            if not link.telemetry.armed or not link.telemetry.offboard:
                raise SitlError('Flight mode or armed state lost during takeoff')
            stable = (np.linalg.norm(np.asarray(link.telemetry.position_ned)-home) < .12
                      and np.linalg.norm(link.telemetry.velocity_ned) < .08)
            stable_since = (stable_since if stable_since is not None else now) if stable else None
            if now-last_takeoff_sample >= .1:
                sample = dict(elapsed_s=now-started, position_ned=list(link.telemetry.position_ned),
                    target_ned=home.tolist(), velocity_ned=list(link.telemetry.velocity_ned),
                    truth_altitude_m=link.truth.alt/1000 if link.truth else None,
                    truth_velocity_ned=None if link.truth is None else
                        [link.truth.vx/100, link.truth.vy/100, link.truth.vz/100],
                    truth_lat_lon=None if link.truth is None else
                        [link.truth.lat*1e-7, link.truth.lon*1e-7],
                    source_time_s=link.truth_source, stable=bool(stable))
                takeoff_samples.append(sample)
                last_takeoff_sample = now
                if now-last_takeoff_report >= 5:
                    event('takeoff_progress', sample=sample)
                    last_takeoff_report = now
            if stable_since is not None and now-stable_since >= 1:
                break
            if now-started > 40:
                raise SitlError('Takeoff did not stabilize')
            time.sleep(.025)
        event('stable_takeoff', position_ned=list(link.telemetry.position_ned))

        guardian, gate = DepthGuardian(), ResultGate(session, selected_track)
        command = None
        follow_start = time.monotonic()
        if camera:
            camera.start_scenario(link.truth_source)
        stall_start = None
        last_rgb = None
        last_record = -math.inf
        last_submitted_capture = -math.inf
        pending_frames = {}
        observed_capture_yaws = {}
        depth_dropped = delay_injected = False
        last_result = None
        recovery_turn = None
        while (time.monotonic()-follow_start < (hover_seconds if smoke else duration_s if mission else 24)
               and (stall_start is None or time.monotonic()-stall_start < 6)):
            control()
            tick = time.monotonic()
            link.poll()
            now = time.monotonic()
            error = link.health_error(now)
            if error or not link.telemetry.offboard or not link.telemetry.armed:
                raise SitlError(error or 'offboard_lost')
            state = estimated_state(link, origin, now)
            if np.linalg.norm(state.position[:2]) > 4 or not .7 <= state.position[2] <= 1.5:
                raise SitlError('Controlled fixture flight envelope exceeded')
            phase = 'smoke_hold' if smoke else 'follow'
            captured_overview = None
            receive_seconds = publish_seconds = 0.
            progress = following_metrics(rows, observations)
            if (not smoke and not mission and 4 <= now-follow_start <= 18 and stall_start is None and rows
                    and rows[-1]['sent_speed'] > .1 and now-follow_start-rows[-1]['elapsed_s'] < .1
                    and command and command['valid'] and now < command['valid_until_s']
                    and all(following_checks(progress).values())):
                stall_start = now
                event('vision_stall_started', duration_s=4., sent_speed=rows[-1]['sent_speed'],
                    command_sequence=command['sequence'], capture_time_s=command['capture_time_s'],
                    expiry_s=command['valid_until_s'], following_metrics=progress)
            if not smoke and not mission and stall_start is None and now-follow_start > 18:
                raise SitlError('Following milestones not reached before stall-window deadline')
            if stall_start is not None:
                phase = 'stale_hold'
            if worker:
                receive_started = time.monotonic()
                result = worker.receive()
                receive_seconds = time.monotonic()-receive_started
                if result is not None:
                    completed = time.monotonic()
                    captured = pending_frames.pop(result['sequence'], None)
                    if captured is None:
                        raise SitlError('Perception result has no matching captured frame')
                    if studio is not None and result.get('selection_action') is not None:
                        action = result['selection_action']
                        action['revision'] = captured[2]['selection_revision']
                        gate.selected_track = (action['request'].get('track_id')
                            if action['accepted'] and captured[2]['selection_revision'] == selection_revision
                            else None)
                    accepted = gate.accept(result, state, completed)
                    if stall_start is None and pending_selection is None:
                        command = accepted
                    # The test withholds post-stall data. An old in-flight result
                    # still expires at its original capture+.9 deadline.
                    observations.append(dict(packet=result, command=accepted, evaluation=captured[2]))
                    last_result = result
                    publish_started = time.monotonic()
                    publish('running', 'Following the selected observed person' if accepted['valid']
                        else accepted['reason'], state, packet=result, images=captured[:2])
                    publish_seconds = time.monotonic()-publish_started
                    event('vision_result', valid=accepted['valid'], reason=accepted['reason'],
                          capture_age_s=completed-result['capture_time_s'])
            request = active_request(command, replace(state, time_s=now))
            if (recovery and recovery_turn is None and now-follow_start >= 3
                    and command is not None and command['valid']):
                recovery_turn = dict(start_s=now, initial_yaw=state.yaw,
                                     target_yaw=state.yaw+.12)
                event('recovery_turn_started', **recovery_turn,
                      policy='scripted .12rad PX4 yaw test; forward remains zero until depth drop')
            depth = dict(forward_speed=0., reason='smoke_hold')
            safety_capture = None
            if camera:
                try:
                    safety = camera.capture(link, state, safety=True)
                    safety_capture = safety.capture_time_s
                    depth = guardian.check(safety, state, request['forward_speed'], time.monotonic())
                    if not worker.busy and (stall_start is None or worker.sequence == 0):
                        # Fault admission uses only observed anchors and measured attitude.
                        # Evaluation truth is never an input to selection or motion.
                        anchors = [] if last_result is None else last_result.get('association_anchors', [])
                        selected_anchor = next((a for a in anchors if a['track_id'] == gate.selected_track), None)
                        anchor_yaw = (None if selected_anchor is None else
                            observed_capture_yaws.get(selected_anchor['capture_time_s']))
                        can_drop = (recovery and not depth_dropped and now-follow_start >= 2
                            and last_result is not None and last_result['observation']['valid']
                            and recovery_turn is not None and anchor_yaw is not None
                            and abs(math.atan2(math.sin(state.yaw-anchor_yaw),
                                              math.cos(state.yaw-anchor_yaw))) >= .025)
                        frame = camera.capture(link, state, drop_depth=bool(can_drop))
                        if frame.capture_time_s > last_submitted_capture:
                            delay = 1. if faults and not delay_injected and now-follow_start >= duration_s/2 else 0.
                            sequence = worker.sequence
                            overview = camera.world.overview()
                            captured_overview = overview
                            if worker.submit(frame, delay=delay, selection=pending_selection):
                                observed_capture_yaws[frame.capture_time_s] = state.yaw
                                observed_capture_yaws = dict(list(observed_capture_yaws.items())[-16:])
                                pending_frames[sequence] = (frame.rgb, overview,
                                    dict(camera.last_evaluation, sequence=sequence, pose_yaw=state.yaw,
                                         selection_revision=selection_revision))
                                pending_selection = None
                                last_submitted_capture, last_rgb = frame.capture_time_s, frame.rgb
                                if can_drop:
                                    depth_dropped = True
                                    event('front_depth_dropped', sequence=sequence, yaw=state.yaw)
                                if delay:
                                    delay_injected = True
                                    event('mission_delay_injected', sequence=sequence, delay_s=delay)
                    elif stall_start is not None and not worker.busy and not any(
                            e['event'] == 'worker_stall_injected' for e in events):
                        frame = camera.capture(link, state)
                        if frame.capture_time_s > last_submitted_capture:
                            sequence = worker.sequence
                            worker.submit(frame, delay=4.)
                            pending_frames[sequence] = (frame.rgb, camera.world.overview(),
                                dict(camera.last_evaluation, sequence=sequence, pose_yaw=state.yaw))
                            last_submitted_capture = frame.capture_time_s
                            event('worker_stall_injected')
                except SitlError as exc:
                    depth = dict(forward_speed=0., reason=str(exc))
            record_overview = None
            overview_started = time.monotonic()
            if recorder and last_rgb is not None and now-last_record >= .1:
                # Graphics stays on its owning thread, and its cost is included
                # in the final freshness/deadline checks before any motion send.
                record_overview = captured_overview if captured_overview is not None else camera.world.overview()
            overview_render_s = time.monotonic()-overview_started
            sent_at = time.monotonic()
            # Recheck expiry after all rendering/IPC work, immediately at send.
            final_request = active_request(command, replace(state, time_s=sent_at))
            speed = min(depth['forward_speed'], final_request['forward_speed'])
            if not smoke and (safety_capture is None or sent_at-safety_capture > .1
                              or sent_at-state.time_s > .1):
                speed = 0.
                depth = dict(forward_speed=0., reason='stale_sensor_at_send')
            if sent_at-tick > .1:
                speed = 0.
                depth = dict(forward_speed=0., reason='control_tick_deadline')
            yaw = final_request['yaw_target']
            if recovery_turn is not None and not depth_dropped and now-recovery_turn['start_s'] < 3:
                speed, yaw = 0., recovery_turn['target_yaw']
            if smoke:
                # A hover test holds position; zero velocity alone permits GPS
                # velocity bias to accumulate into uncorrected position drift.
                link.setpoint(home, position=True, yaw_ned=0.)
            else:
                link.setpoint((speed*math.cos(state.yaw), -speed*math.sin(state.yaw), 0),
                              altitude_ned=float(home[2]), yaw_ned=-yaw)
            row = dict(elapsed_s=sent_at-follow_start, phase=phase,
                sent_at_s=sent_at,
                command_expiry_s=None if command is None else command['valid_until_s'],
                source_time_s=link.truth_source,
                stall_elapsed_s=0. if stall_start is None else sent_at-stall_start,
                position=state.position.tolist(), velocity=state.velocity.tolist(),
                horizontal_speed=float(np.linalg.norm(state.velocity[:2])),
                yaw=state.yaw, sent_speed=speed, requested_speed=final_request['forward_speed'],
                sent_yaw=yaw, scripted_recovery_turn=bool(recovery_turn is not None and not depth_dropped),
                depth_reason=depth['reason'], tick_s=sent_at-tick,
                overview_render_s=overview_render_s,
                vision_receive_s=receive_seconds, studio_publish_s=publish_seconds,
                state_age_s=sent_at-state.time_s,
                depth_age_s=None if safety_capture is None else sent_at-safety_capture,
                truth_position=None if camera is None or camera.truth_position is None
                    else camera.truth_position.tolist(),
                truth_velocity_ned=[link.truth.vx/100, link.truth.vy/100, link.truth.vz/100],
                truth_altitude_above_start_m=link.truth.alt/1000-truth_origin[2],
                command_sequence=final_request['sequence'])
            rows.append(row)
            publish('running', depth['reason'], state, depth=depth)
            if record_overview is not None:
                recorder.append(sent_at-follow_start, last_rgb, record_overview, row)
                last_record = sent_at
            time.sleep(max(0., .025-(time.monotonic()-tick)))
        event('mission_window_complete')
        mission_end = time.monotonic()
    except StudioStop:
        mission_end = time.monotonic()
        event('studio_stop_requested')
    except BaseException as exc:
        failure = f'{type(exc).__name__}: {exc}'
        event('failure', error=failure)
        if link is not None:
            now = time.monotonic()
            sources = dict(attitude=link.attitude_source, truth=link.truth_source,
                           position=(link.telemetry.boot_ms or 0)/1000)
            ages = {}
            for label, source in sources.items():
                receipt = next(((r, s) for r, s in reversed(link.clock.receipts) if r < source), None)
                ages[label] = None if receipt is None else now-receipt[1]
            event('failure_clock_diagnostics', source_times=sources, age_upper_bounds_s=ages,
                pending_sync=len(link.clock.pending), clock_reset=link.clock.reset,
                recent_sync=list(link.clock.receipts)[-5:], status=list(link.last_status))
    finally:
        mission_end = mission_end or time.monotonic()
        if link is not None and armed_requested:
            try:
                link.command(21)  # NAV_LAND; never force-disarm in the air
                event('land_requested')
                link.wait_for(lambda: link.telemetry.landed and not link.telemetry.armed
                    and time.monotonic()-link.telemetry.landed_at < .5
                    and time.monotonic()-link.telemetry.heartbeat_at < 1.5, 30)
                event('landed_disarmed')
            except Exception as exc:
                event('landing_unconfirmed', error=str(exc))
                failure = failure or 'Landing confirmation unavailable'
        try:
            result = score(rows, observations, events, smoke)
            if mission and not smoke and studio is None:
                from experiments.px4_evaluation import mission_score
                try:
                    result = mission_score(rows, observations, events,
                        mission_start_s=follow_start or epoch, mission_end_s=mission_end,
                        require_recovery=recovery, min_tracking_fraction=.8)
                    if faults:
                        fault_checks = dict(
                            actual_delay_injected=any(e['event']=='mission_delay_injected' for e in events),
                            delayed_output_rejected=any(o['command']['reason']=='stale_perception_result'
                                and o['packet']['completed_s']-o['packet']['capture_time_s'] >= 1
                                for o in observations))
                        result['checks'].update(fault_checks)
                        result['safety_passed'] = result['safety_passed'] and all(fault_checks.values())
                        result['tracking_success'] = result['tracking_success'] and all(fault_checks.values())
                        result['passed'] = result['passed'] and all(fault_checks.values())
                except Exception as exc:
                    result = dict(passed=False, safety_passed=False, tracking_success=False,
                        checks=dict(evaluation_completed=False), metrics={}, recovery={})
                    failure = failure or f'Evaluation failed: {exc}'
            if studio is not None:
                names = {e['event'] for e in events}
                checks = dict(no_runtime_failure=failure is None,
                    operator_stop_or_window_complete=bool(names & {'studio_stop_requested','mission_window_complete'}),
                    landed_if_armed=not armed_requested or 'landed_disarmed' in names)
                result = dict(passed=all(checks.values()), checks=checks,
                    evaluation_scope='operator_demonstration',
                    evaluation_note='Explicit person switches are not a fixed-identity benchmark',
                    tracking_success=None, operator_stopped='studio_stop_requested' in names,
                    flight_started=armed_requested, observations=len(observations),
                    accepted_selection_actions=sum(bool((o['packet'].get('selection_action') or {}).get('accepted'))
                        for o in observations))
            result.update(mode='autopilot_smoke' if smoke else method+'_follow', error=failure,
                session=session, px4_sensor_profile=sensor_profile,
                estimator_profile=estimator_profile,
                method=method, trajectory=trajectory, duration_s=duration_s,
                mission_config=provenance['mission_config'],
                mission_start_s=follow_start or epoch, mission_end_s=mission_end,
                actual_px4=any(e['event']=='owned_px4_verified' for e in events),
                passed=result['passed'] and failure is None,
                limits=['simulation only', 'stylized actors', 'ideal RGB and depth',
                        'no obstacle collision physics', 'no neural superiority claim'])
            if smoke and rows:
                result['hover_metrics'] = dict(duration_s=rows[-1]['elapsed_s'],
                    estimated_position_error_max_m=max(float(np.linalg.norm(
                        np.asarray(r['position'])-[0,0,ALTITUDE])) for r in rows),
                    estimated_speed_p95_m_s=float(np.percentile([np.linalg.norm(r['velocity']) for r in rows],95)),
                    truth_speed_p95_m_s=float(np.percentile([np.linalg.norm(r['truth_velocity_ned']) for r in rows],95)),
                    truth_altitude_range_m=float(np.ptp([r['truth_altitude_above_start_m'] for r in rows])))
        except BaseException as exc:
            result = dict(passed=False, error=f'Final scoring failed: {type(exc).__name__}: {exc}',
                actual_px4=any(e['event']=='owned_px4_verified' for e in events),
                session=session, method=method, px4_sensor_profile=sensor_profile)
        # One cleanup failure must never skip stopping the owned PX4 child.
        for name, resource in [('link', link), ('px4', lease), ('vision', worker),
                               ('camera', camera), ('recording', recorder)]:
            if resource is not None:
                try:
                    resource.finish(result) if name == 'recording' else resource.close()
                except BaseException as exc:
                    event('cleanup_failed', resource=name, error=str(exc))
                    result.update(passed=False, error=result['error'] or f'{name} cleanup failed: {exc}')
        for name, value in [('summary.json', result), ('events.json', events),
                            ('control.json', rows), ('observations.json', observations),
                            ('takeoff.json', takeoff_samples)]:
            (folder/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--estimator-profile', choices=ESTIMATOR_PROFILES, default='stock')
    parser.add_argument('--hover-seconds', type=float, default=3.)
    parser.add_argument('--mission', action='store_true', help='Bounded long mission; no legacy four-second stall protocol')
    parser.add_argument('--method', choices=('neural','direct','filtered'), default='neural')
    parser.add_argument('--trajectory', choices=('stationary','walk','crossing','occlusion'), default='stationary')
    parser.add_argument('--target-speed', type=float, default=.1)
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--faults', action='store_true', help='Inject one real 1s worker delay at the declared midpoint')
    parser.add_argument('--recovery', action='store_true', help='Inject one missing front depth frame during a measured turn')
    parser.add_argument('--low-noise-sensors', action='store_true',
                        help='Diagnostic: 1%% stock PX4 GPS/baro/mag/IMU noise; not realistic sensor validation')
    parser.add_argument('--select-track', type=int, default=1,
                        help='Select this observed track once; never switch automatically')
    args = parser.parse_args()
    if not 3 <= args.hover_seconds <= 120:
        parser.error('Hover duration must be between 3 and 120 seconds')
    if args.select_track < 1:
        parser.error('Track must be positive')
    result = run(args.output, args.smoke, args.select_track, args.low_noise_sensors,
                 estimator_profile=args.estimator_profile, hover_seconds=args.hover_seconds,
                 method=args.method, trajectory=args.trajectory, target_speed=args.target_speed,
                 duration_s=args.duration, mission=args.mission, faults=args.faults, recovery=args.recovery)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
