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
import time
import uuid

import numpy as np

from experiments.flight_guidance import active_request, release
from experiments.flight_safety import DepthGuardian
from experiments.px4_camera import SihCamera, VisionWorker, estimated_state
from experiments.px4_recording import AsyncRecorder
from flybrain_sim.px4_sih import ALTITUDE, ROOT, OwnedSih, SihLink, sha
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


def run(folder, smoke=False, selected_track=1, low_noise_sensors=False):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    session = str(uuid.uuid4())
    source_files = [p for directory in ('experiments', 'perception', 'flybrain_sim')
                    for p in (ROOT/directory).glob('*.py')]
    source_files += [ROOT/'scripts/build_px4_sih.py', ROOT/'integrations/px4/sih.px4board',
                    ROOT/'models/flyvis_0000_000.manifest.json']
    sensor_profile = '1% stock GPS/baro/mag/IMU noise' if low_noise_sensors else 'stock sensor noise'
    provenance = dict(session=session, px4_sensor_profile=sensor_profile,
        source_sha256={str(p.relative_to(ROOT)): sha(p)
        for p in sorted(source_files)}, selection=dict(track_id=selected_track,
        policy='fixed observed track selected once; no truth or automatic replacement'),
        acceptance='takeoff; >=5 fresh neural results; >=10 advancing ticks; >=.3m movement; '
        '>=.2m selected-depth reduction; reach all following milestones before active >.1m/s stall '
        'between 4s and 18s in the 24s mission; '
        '4s worker delay rejected; zero speed after capture expiry; late speed <.1m/s; landed/disarmed')
    (folder/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    events, rows, observations, takeoff_samples = [], [], [], []
    lease = link = worker = camera = recorder = None
    failure = None
    armed_requested = False
    epoch = time.monotonic()
    def event(name, **values):
        value = dict(event=name, elapsed_s=time.monotonic()-epoch, **values)
        events.append(value)
        print(json.dumps(value), flush=True)
    try:
        if not smoke:
            worker = VisionWorker(session, selected_track)
            deadline = time.monotonic()+180
            while True:
                result = worker.receive()
                if result is not None:
                    if result['kind'] != 'ready':
                        raise SitlError('Unexpected vision startup receipt')
                    event('vision_ready', model=result['model'])
                    break
                if time.monotonic() > deadline:
                    raise SitlError('Vision startup timed out before arming')
                time.sleep(.05)
        lease = OwnedSih(folder / 'px4', low_noise_sensors=low_noise_sensors)
        link = SihLink(lease)
        link.verify_simulator()
        event('owned_px4_verified', pid=lease.process.pid, build=lease.receipt,
              px4_sensor_profile=sensor_profile)
        for message_id, hz in ((31, 40), (32, 40), (230, 10), (1, 2), (245, 5)):
            link.command(511, (message_id, 1e6/hz))
        for name, value, integer in [('COM_OBL_RC_ACT', 4, True), ('COM_OF_LOSS_T', .5, False),
                                      ('COM_RC_IN_MODE', 4, True), ('SDLOG_MODE', -1, True)]:
            link.set_parameter(name, value, integer=integer)
        started = time.monotonic()
        next_report = started
        while True:
            link.poll()
            now = time.monotonic()
            error = link.health_error(now)
            if not error and link.truth is not None:
                try:
                    link.clock.capture_bound(link.truth_source, now, .1)
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
        home = origin + [0., 0., -ALTITUDE]
        if not smoke:
            camera = SihCamera(link.truth, epoch)
            # Allocate native graphics contexts before arming. These warmup
            # images are not submitted as observations or used as depth.
            camera.world.capture(safety=True)
            camera.world.capture()
            camera.world.overview()
            from experiments.mantis_recording import SessionRecorder
            recorder = AsyncRecorder(SessionRecorder(folder/'recording', fps=10, max_bytes=32*1024**2,
                provenance=dict(backend='patched PX4 SIH / MuJoCo render only',
                    build=lease.receipt, session=session, selected_track=selected_track,
                    neural_clock='sampled target cue; .1 model seconds per observation',
                    collision_physics=False, px4_sensor_profile=sensor_profile,
                    sensor_model='ideal RGB/depth; estimated camera pose')))
            link.wait_for(lambda: not link.health_error(time.monotonic()), 10)

        def prime():
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
        stall_start = None
        last_rgb = None
        last_record = -math.inf
        last_submitted_capture = -math.inf
        while (time.monotonic()-follow_start < (3 if smoke else 24)
               and (stall_start is None or time.monotonic()-stall_start < 6)):
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
            progress = following_metrics(rows, observations)
            if (not smoke and 4 <= now-follow_start <= 18 and stall_start is None and rows
                    and rows[-1]['sent_speed'] > .1 and now-follow_start-rows[-1]['elapsed_s'] < .1
                    and command and command['valid'] and now < command['valid_until_s']
                    and all(following_checks(progress).values())):
                stall_start = now
                event('vision_stall_started', duration_s=4., sent_speed=rows[-1]['sent_speed'],
                    command_sequence=command['sequence'], capture_time_s=command['capture_time_s'],
                    expiry_s=command['valid_until_s'], following_metrics=progress)
            if not smoke and stall_start is None and now-follow_start > 18:
                raise SitlError('Following milestones not reached before stall-window deadline')
            if stall_start is not None:
                phase = 'stale_hold'
            if worker:
                result = worker.receive()
                if result is not None:
                    completed = time.monotonic()
                    accepted = gate.accept(result, state, completed)
                    if stall_start is None:
                        command = accepted
                    # The test withholds post-stall data. An old in-flight result
                    # still expires at its original capture+.9 deadline.
                    observations.append(dict(packet=result, command=accepted))
                    event('vision_result', valid=accepted['valid'], reason=accepted['reason'],
                          capture_age_s=completed-result['capture_time_s'])
            request = active_request(command, replace(state, time_s=now))
            depth = dict(forward_speed=0., reason='smoke_hold')
            safety_capture = None
            if camera:
                try:
                    safety = camera.capture(link, state, safety=True)
                    safety_capture = safety.capture_time_s
                    depth = guardian.check(safety, state, request['forward_speed'], time.monotonic())
                    if not worker.busy and (stall_start is None or worker.sequence == 0):
                        frame = camera.capture(link, state)
                        if frame.capture_time_s > last_submitted_capture:
                            worker.submit(frame)
                            last_submitted_capture, last_rgb = frame.capture_time_s, frame.rgb
                    elif stall_start is not None and not worker.busy and not any(
                            e['event'] == 'worker_stall_injected' for e in events):
                        frame = camera.capture(link, state)
                        if frame.capture_time_s > last_submitted_capture:
                            worker.submit(frame, delay=4.)
                            last_submitted_capture = frame.capture_time_s
                            event('worker_stall_injected')
                except SitlError as exc:
                    depth = dict(forward_speed=0., reason=str(exc))
            record_overview = None
            overview_started = time.monotonic()
            if recorder and last_rgb is not None and now-last_record >= .1:
                # Graphics stays on its owning thread, and its cost is included
                # in the final freshness/deadline checks before any motion send.
                record_overview = camera.world.overview()
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
                depth_reason=depth['reason'], tick_s=sent_at-tick,
                overview_render_s=overview_render_s,
                state_age_s=sent_at-state.time_s,
                depth_age_s=None if safety_capture is None else sent_at-safety_capture,
                truth_position=None if camera is None or camera.truth_position is None
                    else camera.truth_position.tolist(),
                command_sequence=final_request['sequence'])
            rows.append(row)
            if record_overview is not None:
                recorder.append(sent_at-follow_start, last_rgb, record_overview, row)
                last_record = sent_at
            time.sleep(max(0., .025-(time.monotonic()-tick)))
        event('mission_window_complete')
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
        result = score(rows, observations, events, smoke)
        result.update(mode='autopilot_smoke' if smoke else 'neural_follow', error=failure,
            session=session, px4_sensor_profile=sensor_profile,
            actual_px4=any(e['event']=='owned_px4_verified' for e in events),
            passed=result['passed'] and failure is None,
            limits=['simulation only', 'stylized stationary actors', 'ideal RGB and depth',
                    'no obstacle collision physics', 'no neural superiority claim'])
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
    parser.add_argument('--low-noise-sensors', action='store_true',
                        help='Diagnostic: 1%% stock PX4 GPS/baro/mag/IMU noise; not realistic sensor validation')
    parser.add_argument('--select-track', type=int, default=1,
                        help='Select this observed track once; never switch automatically')
    args = parser.parse_args()
    if args.select_track < 1:
        parser.error('Track must be positive')
    result = run(args.output, args.smoke, args.select_track, args.low_noise_sensors)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
