"""SIH truth renders pixels; PX4 estimates supply camera pose and control state.

MuJoCo is a renderer only in this adapter. No mj_step or MantisAutopilot.
"""
from __future__ import annotations

from dataclasses import replace
import multiprocessing as mp
import queue
import time
import traceback

import numpy as np

from experiments.flight_contracts import CameraFrame, FlightState, R_BODY_CAMERA
from flybrain_sim.px4_sitl import SitlError

D = np.diag([1., -1., -1.])  # NED->NWU, and FRD->FLU


def quaternion_rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > .01:
        raise SitlError('Invalid PX4 quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def estimated_state(link, origin_ned, now):
    error = link.health_error(now)
    if error:
        raise SitlError(error)
    a, t = link.attitude, link.telemetry
    captured = min(link.clock.capture_bound(link.attitude_source, now, .2),
                   link.clock.capture_bound(t.boot_ms/1000, now, .2))
    rates = np.array([a.rollspeed, a.pitchspeed, a.yawspeed])
    if not np.isfinite(rates).all():
        raise SitlError('Invalid estimated angular velocity')
    return FlightState(captured, D @ (np.asarray(t.position_ned)-origin_ned),
        D @ np.asarray(t.velocity_ned), D @ quaternion_rotation([a.q1, a.q2, a.q3, a.q4]) @ D,
        D @ rates, np.zeros(4))


class SihCamera:
    def __init__(self, truth, epoch):
        from experiments.mantis_arena import ArenaWorld
        self.world = ArenaWorld()
        self.origin = np.array([truth.lat*1e-7, truth.lon*1e-7, truth.alt/1000])
        self.epoch = epoch
        self.truth_position = None
        self.last_source = 0.

    def capture(self, link, state, safety=False):
        truth = link.truth
        now = time.monotonic()
        source = link.truth_source
        bound = min(link.clock.capture_bound(source, now, .1),
                    link.clock.capture_bound(link.attitude_source, now, .1),
                    link.clock.capture_bound(link.telemetry.boot_ms/1000, now, .1))
        source_times = [source, link.attitude_source, link.telemetry.boot_ms/1000]
        if max(source_times)-min(source_times) > .06:
            raise SitlError('estimated_pose_skew')
        if not np.isfinite([truth.lat, truth.lon, truth.alt, truth.vx, truth.vy, truth.vz,
                           truth.rollspeed, truth.pitchspeed, truth.yawspeed]).all():
            raise SitlError('Invalid truth telemetry')
        # Local tangent approximation over this small fixed room; origin is
        # independently captured on the ground for truth and estimator.
        latitude, longitude = truth.lat*1e-7, truth.lon*1e-7
        self.truth_position = np.array([
            np.deg2rad(latitude-self.origin[0])*6371000.,
            -np.deg2rad(longitude-self.origin[1])*6371000.*np.cos(np.deg2rad(self.origin[0])),
            truth.alt/1000-self.origin[2]])
        world = self.world
        world.data.qpos[:3] = self.truth_position
        q = np.asarray(truth.attitude_quaternion)
        quaternion_rotation(q)
        q = q / np.linalg.norm(q)
        world.data.qpos[3:7] = q * [1, 1, -1, -1]
        world.data.qvel[:3] = D @ np.array([truth.vx, truth.vy, truth.vz]) / 100
        world.data.qvel[3:6] = D @ np.array([truth.rollspeed, truth.pitchspeed, truth.yawspeed])
        world.data.time = max(0., bound-self.epoch)
        world.update_scene(world.data.time, trajectory='stationary', target_speed=0., obstacle=False)
        frame = world.capture(safety=safety)
        # These metadata are estimated, not the renderer's true camera pose.
        return replace(frame, capture_time_s=bound,
            rotation_world_camera=state.rotation @ R_BODY_CAMERA,
            position_world_camera=state.position + state.rotation @ np.array([.25, 0, 0])).validate()

    def close(self):
        self.world.close()


class StationaryAssociationMemory:
    """One depth-gap association grace, exclusively for this stationary fixture.

    Old world corners remain an association hypothesis, with their original
    capture time. They never populate current depth, detections or guidance.
    Studio and the general tracker retain their existing missing-depth policy.
    """
    max_capture_gap_s = 1.2

    def __init__(self):
        self.last_capture = -float('inf')
        self.grace = set()
        self.previous = {}

    def prepare(self, tracker, frame):
        capture = frame.capture_time_s
        if capture <= self.last_capture or not frame.registration_verified:
            tracker.anchors.clear()
            self.grace.clear()
        self.last_capture = capture
        tracker.anchors = {tid: anchor for tid, anchor in tracker.anchors.items()
            if tid in tracker.tracks and 0 <= capture-anchor['capture_time_s'] <= self.max_capture_gap_s}
        self.grace.intersection_update(tracker.anchors)
        self.previous = {tid: dict(corners=anchor['corners'].copy(),
                                  capture_time_s=anchor['capture_time_s'])
                         for tid, anchor in tracker.anchors.items()}

    def complete(self, tracker, result, frame):
        detections = {d['track_id']: d for d in result['detections'].get('detections', [])
                      if d['class_id'] == 0}
        receipts = []
        for tid in list(tracker.anchors):
            if tid not in detections:
                tracker.anchors.pop(tid)
        for tid, detection in detections.items():
            if detection.get('surface_measurement') is not None:
                self.grace.discard(tid)
                continue
            old = self.previous.get(tid)
            if (old is not None and tid in tracker.tracks and tid not in self.grace
                    and frame.registration_verified
                    and 0 <= frame.capture_time_s-old['capture_time_s'] <= self.max_capture_gap_s):
                tracker.anchors[tid] = old
                self.grace.add(tid)
                receipts.append(dict(track_id=tid, retained_for_association_only=True,
                    anchor_capture_time_s=old['capture_time_s'],
                    missing_depth_capture_time_s=frame.capture_time_s))
            else:
                tracker.anchors.pop(tid, None)
                self.grace.discard(tid)
        self.grace.intersection_update(detections)
        return receipts


def _worker(incoming, outgoing, session, selected_track):
    try:
        from experiments.mantis_vision import MantisVision
        from experiments.mantis_pipeline import MantisPerceptionPipeline
        from experiments.mantis_selection import SelectionGuard
        from experiments.mantis_studio_vision import StudioPersonRetryDetector

        class LivePipeline(MantisPerceptionPipeline):
            def process(self, sample):
                now = time.monotonic()
                return super().process(replace(sample, received_monotonic_s=now,
                    capture_age_at_receive_s=now-sample.capture_time_s,
                    stream_id=session, clock_domain='host_monotonic_source_lower_bound',
                    frame_id='px4_sih_front_optical'))

        class LiveSelection(SelectionGuard):
            def update(self, result, now_s, image_hw, max_age_s=.65):
                return super().update(result, time.monotonic(), image_hw, max_age_s)

        vision = MantisVision()
        tracker = vision.pipeline.tracker
        vision.detector = StudioPersonRetryDetector(vision.detector)
        vision.pipeline = LivePipeline(vision.detector, max_frame_age_s=.65, appearance_tracking=True)
        vision.pipeline.tracker = tracker
        guard = vision.bridge = LiveSelection()
        association_memory = StationaryAssociationMemory()
        # Load/JIT warmup occurs before arming, on a blank visual cue. It grants
        # no target or command authority. Recurrence resets afterward.
        for _ in range(2):
            vision.brain.step(np.full((391, 391), .5, np.float32), .1)
        vision.brain.reset()
        outgoing.put(dict(kind='ready', model=vision.frozen))
        selected_once = False
        while True:
            packet = incoming.get()
            if packet is None:
                return
            sequence, frame, delay = packet
            if delay:
                time.sleep(delay)  # deliberate independent-worker stall test
            guard.observe_frame(frame)
            association_memory.prepare(vision.pipeline.tracker, frame)
            result = vision.process(frame)
            association_receipts = association_memory.complete(vision.pipeline.tracker, result, frame)
            selection = guard.status()
            if not selected_once and any(p['track_id'] == selected_track and p['selectable']
                                         for p in selection['people']):
                try:
                    guard.select(selected_track, selection['sequence'], now_s=time.monotonic())
                    selected_once = True
                except ValueError:
                    pass  # Initial preview expired; wait for a fresh observation.
            outgoing.put(dict(kind='result', session=session, sequence=sequence,
                capture_time_s=frame.capture_time_s, completed_s=time.monotonic(),
                candidate=result['candidate'], observation=result['observation'],
                detections=result['detections'], selection=guard.status(),
                stationary_association_memory=association_receipts,
                reprojections=vision.pipeline.tracker.last_reprojections,
                neural_features=result['supplied_features'].tolist(),
                neural_valid=bool(result['neural']['valid']),
                neural_elapsed_s=result['neural']['elapsed_wall_s']))
    except BaseException:
        outgoing.put(dict(kind='error', error=traceback.format_exc()))


class VisionWorker:
    def __init__(self, session, selected_track):
        context = mp.get_context('spawn')
        self.incoming, self.outgoing = context.Queue(1), context.Queue(2)
        self.process = context.Process(target=_worker,
            args=(self.incoming, self.outgoing, session, selected_track), daemon=True)
        self.process.start()
        self.busy = False
        self.sequence = 0
        self.last_capture = -float('inf')

    def receive(self):
        try:
            result = self.outgoing.get_nowait()
        except queue.Empty:
            if not self.process.is_alive():
                raise SitlError('Vision worker exited')
            return None
        self.busy = False
        if result['kind'] == 'error':
            raise SitlError(result['error'])
        return result

    def submit(self, frame, delay=0.):
        if self.busy or frame.capture_time_s <= self.last_capture:
            return False
        self.incoming.put_nowait((self.sequence, frame, delay))
        self.sequence += 1
        self.last_capture = frame.capture_time_s
        self.busy = True
        return True

    def close(self):
        if self.process.is_alive():
            self.process.terminate()
        self.process.join(timeout=5)
        for channel in (self.incoming, self.outgoing):
            channel.cancel_join_thread()
            channel.close()
