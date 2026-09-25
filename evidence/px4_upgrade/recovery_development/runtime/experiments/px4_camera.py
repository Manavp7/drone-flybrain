"""SIH truth renders pixels; PX4 estimates supply camera pose and control state.

MuJoCo is a renderer only in this adapter. No mj_step or MantisAutopilot.
"""
from __future__ import annotations

from dataclasses import replace
import math
import multiprocessing as mp
import queue
import time
import traceback

import numpy as np

from experiments.flight_contracts import CameraFrame, FlightState, R_BODY_CAMERA, ROOT
from flybrain_sim.px4_sitl import SitlError

D = np.diag([1., -1., -1.])  # NED->NWU, and FRD->FLU
TRAJECTORIES = ('stationary', 'walk', 'crossing', 'occlusion')
GUIDANCE_METHODS = ('neural', 'direct', 'filtered')


class ScenarioClock:
    """Actor phase starts explicitly from SIH time, never model-loading time."""
    def __init__(self):
        self.start_source_s = None
        self.last_source_s = None

    def start(self, source_time_s):
        if (isinstance(source_time_s, bool) or not isinstance(source_time_s, (int, float))
                or not math.isfinite(source_time_s) or source_time_s < 0):
            raise ValueError('Scenario start requires finite nonnegative SIH time')
        if self.start_source_s is not None:
            raise ValueError('Scenario clock is already started')
        if self.last_source_s is not None and source_time_s < self.last_source_s:
            raise ValueError('Scenario start must not precede the last rendered source')
        self.start_source_s = float(source_time_s)

    def elapsed(self, source_time_s):
        if (isinstance(source_time_s, bool) or not isinstance(source_time_s, (int, float))
                or not math.isfinite(source_time_s) or source_time_s < 0
                or self.last_source_s is not None and source_time_s < self.last_source_s
                or self.start_source_s is not None and source_time_s < self.start_source_s):
            raise ValueError('Scenario source time must be finite and monotonic')
        self.last_source_s = float(source_time_s)
        return 0. if self.start_source_s is None else float(source_time_s-self.start_source_s)


def drop_frame_depth(frame):
    """One owned missing-depth sample; RGB and all acquisition metadata survive."""
    frame.validate()
    return replace(frame, depth_m=np.full_like(frame.depth_m, np.nan)).validate()


def selection_request(value):
    """Snapshot the only two accepted UI actions; no identity/evaluator input."""
    if value is None:
        return None
    if isinstance(value, dict) and set(value) == {'clear'} and value['clear'] is True:
        return {'clear': True}
    if (isinstance(value, dict) and set(value) == {'track_id', 'sequence'}
            and type(value['track_id']) is int and value['track_id'] > 0
            and type(value['sequence']) is int and value['sequence'] >= 0):
        return dict(value)
    raise ValueError('Selection requires {track_id, sequence} or {clear: true}')


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
    def __init__(self, truth, epoch, *, trajectory='stationary', target_speed=.1):
        if trajectory not in TRAJECTORIES:
            raise ValueError('Unsupported SIH scene trajectory')
        if (isinstance(target_speed, bool) or not isinstance(target_speed, (int, float))
                or not math.isfinite(target_speed) or not 0 <= target_speed <= .5):
            raise ValueError('Scene target speed must be in [0, .5] m/s')
        from experiments.mantis_arena import ArenaWorld
        self.world = ArenaWorld()
        self.origin = np.array([truth.lat*1e-7, truth.lon*1e-7, truth.alt/1000])
        self.epoch = epoch
        self.truth_position = None
        self.last_source = 0.
        self.trajectory, self.target_speed = trajectory, float(target_speed)
        self.scenario_clock = ScenarioClock()
        self.last_evaluation = None
        self.depth_drop_used = False

    def start_scenario(self, source_time_s):
        """Call after stable takeoff; projections remain main-process evaluator data."""
        self.scenario_clock.start(source_time_s)

    def capture(self, link, state, safety=False, *, drop_depth=False):
        if type(drop_depth) is not bool:
            raise ValueError('Depth-drop selection must be boolean')
        if drop_depth and (safety or self.depth_drop_used):
            raise ValueError('Only one front-camera depth drop is allowed per scene')
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
        world.data.time = self.scenario_clock.elapsed(source)
        world.update_scene(world.data.time, trajectory=self.trajectory,
                           target_speed=self.target_speed, obstacle=False)
        frame = world.capture(safety=safety)
        if not safety:
            # Evaluate the renderer's true pose before replacing metadata with
            # estimates. Never attach these actor labels/projections to frame.
            self.last_evaluation = dict(capture_time_s=bound, source_time_s=source,
                scenario_time_s=world.data.time, trajectory=self.trajectory,
                target_speed_m_s=self.target_speed,
                projections=world.evaluation_people(frame),
                truth_position=self.truth_position.tolist(), depth_drop=drop_depth,
                annotation='evaluator only; full projected mesh, not visible segmentation')
        # These metadata are estimated, not the renderer's true camera pose.
        estimated = replace(frame, capture_time_s=bound,
            rotation_world_camera=state.rotation @ R_BODY_CAMERA,
            position_world_camera=state.position + state.rotation @ np.array([.25, 0, 0])).validate()
        if drop_depth:
            estimated = drop_frame_depth(estimated)
            self.depth_drop_used = True
        return estimated

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


def _live_pipeline(detector, session, clock):
    from experiments.mantis_pipeline import MantisPerceptionPipeline

    class LivePipeline(MantisPerceptionPipeline):
        def process(self, sample):
            now = clock()
            return super().process(replace(sample, received_monotonic_s=now,
                capture_age_at_receive_s=now-sample.capture_time_s,
                stream_id=session, clock_domain='host_monotonic_source_lower_bound',
                frame_id='px4_sih_front_optical'))

    return LivePipeline(detector, max_frame_age_s=.65, clock=clock, appearance_tracking=True)


def _live_selection(clock):
    from experiments.hybrid_target import TargetBridge
    from experiments.mantis_selection import SelectionGuard, _appearance_similarity

    class LiveSelection(SelectionGuard):
        """Recent UI frames express intent; only new observations grant selection.

        Descriptors never leave this worker. Each cached descriptor stays fixed
        to its clicked image and is invalidated permanently by an intervening
        loss, ambiguity or appearance change. This does not extend frame age.
        """
        intent_max_age_s = 3.
        intent_max_snapshots = 8

        def __init__(self):
            super().__init__()
            self._intent_cache = []
            self._intent_stream = None
            self._pending_intent = None

        def _prune_intents(self, now):
            self._intent_cache = [snapshot for snapshot in self._intent_cache
                if 0 <= now-snapshot['capture_time_s'] <= self.intent_max_age_s]

        def selection_history(self):
            self._prune_intents(clock())
            return [dict(sequence=snapshot['sequence'], capture_time_s=snapshot['capture_time_s'],
                people=[dict(person, bbox_xyxy=list(person['bbox_xyxy']),
                    selectable=person['track_id'] in snapshot['tracks'] and
                        snapshot['tracks'][person['track_id']]['reason'] is None)
                    for person in snapshot['people']]) for snapshot in self._intent_cache]

        def request_selection(self, action):
            # Revocation precedes inference, including rejected requests.
            self.clear()
            self._pending_intent = None
            now = clock()
            receipt = dict(request=action, accepted=False, applied_at_s=now)
            if action.get('clear'):
                receipt.update(accepted=True, reason='cleared')
                return receipt
            self._prune_intents(now)
            snapshot = next((item for item in self._intent_cache
                if item['sequence'] == action['sequence']), None)
            if snapshot is None:
                receipt['reason'] = 'selection_snapshot_unavailable_or_expired'
                return receipt
            receipt.update(source_capture_time_s=snapshot['capture_time_s'],
                           source_age_s=now-snapshot['capture_time_s'])
            target = snapshot['tracks'].get(action['track_id'])
            if self._bridge.stream_changed or snapshot['stream'] != self._bridge.stream:
                receipt['reason'] = 'selection_stream_changed'
            elif target is None:
                receipt['reason'] = 'selection_source_not_selectable'
            elif target['reason'] is not None:
                receipt['reason'] = target['reason']
            else:
                receipt['reason'] = 'waiting_for_fresh_selection_observation'
                self._pending_intent = (snapshot, target, receipt)
            return receipt

        def _observe_intents(self, result, now, max_age_s):
            self._prune_intents(now)
            stream = tuple(result.get(k) for k in ('stream_id', 'clock_domain', 'frame_id'))
            changed = self._bridge.stream_changed or (
                self._intent_stream is not None and stream != self._intent_stream)
            fresh = (not changed and result.get('status') == 'ok'
                and result.get('inference_executed') is True
                and self._sequence == result.get('sequence')
                and self._capture == result.get('capture_time_s')
                and self._capture is not None and 0 <= now-self._capture <= max_age_s
                and self._bridge.stream == stream)
            people = {person['track_id']: person for person in self._people} if fresh else {}
            vanished = [person for person in self._previous_people
                        if person['track_id'] not in people]
            # Include an in-flight intent even when its source expires during
            # inference; it must receive a definitive rejection, never renewal.
            snapshots = list(self._intent_cache)
            if self._pending_intent is not None and not any(
                    item is self._pending_intent[0] for item in snapshots):
                snapshots.append(self._pending_intent[0])
            for snapshot in snapshots:
                for track_id, target in snapshot['tracks'].items():
                    if target['reason'] is not None:
                        continue
                    person, feature = people.get(track_id), self._features.get(track_id)
                    if changed or stream != snapshot['stream']:
                        reason = 'selection_stream_changed'
                    elif not 0 <= now-snapshot['capture_time_s'] <= self.intent_max_age_s:
                        reason = 'selection_snapshot_expired'
                    elif not fresh:
                        reason = 'selection_continuity_unavailable'
                    elif person is None:
                        reason = 'selection_track_lost'
                    elif self._ambiguous(person) or self._ambiguous(person, vanished):
                        reason = 'selection_ambiguous'
                    elif feature is None:
                        reason = 'selection_appearance_unavailable'
                    elif _appearance_similarity(target['anchor'], feature) < self.appearance_threshold:
                        reason = 'selection_appearance_changed'
                    else:
                        reason = None
                    target['reason'] = reason
            if changed:
                self._intent_cache.clear()
            if fresh:
                self._intent_stream = stream
            return fresh, stream

        def _resolve_intent(self, result, observation, now, image_hw, max_age_s):
            pending, self._pending_intent = self._pending_intent, None
            if pending is None:
                return observation
            snapshot, target, receipt = pending
            receipt.update(resolved_sequence=result.get('sequence'),
                resolved_capture_time_s=result.get('capture_time_s'), resolved_at_s=now,
                source_age_s=now-snapshot['capture_time_s'])
            if target['reason'] is not None:
                receipt['reason'] = target['reason']
                return observation
            track_id = receipt['request']['track_id']
            try:
                # Both validators use this new frame's actual timestamp and
                # boxes. No cached geometry or timestamp enters the observation.
                current = TargetBridge(track_id=track_id).update(result, now, image_hw, max_age_s)
                if not current['valid']:
                    raise ValueError(current['reason'])
                self.select(track_id, result['sequence'], now_s=now)
            except ValueError as exc:
                self.clear()
                receipt['reason'] = str(exc)
                return self._invalid('selection_required')
            self._anchor = dict(mode=target['anchor']['mode'],
                                histogram=target['anchor']['histogram'].copy())
            similarity = _appearance_similarity(self._anchor, self._features[track_id])
            self._last_similarity = similarity
            self._reason = 'observed'
            current['appearance_similarity'] = similarity
            receipt.update(accepted=True, reason='selected_from_fresh_continuous_observation',
                           appearance_similarity=similarity)
            return current

        def update(self, result, now_s, image_hw, max_age_s=.65):
            now = clock()
            observation = super().update(result, now, image_hw, max_age_s)
            fresh, stream = self._observe_intents(result, now, max_age_s)
            observation = self._resolve_intent(result, observation, now, image_hw, max_age_s)
            if fresh:
                people = self.status()['people']
                self._intent_cache.append(dict(sequence=self._sequence,
                    capture_time_s=self._capture, stream=stream,
                    people=[dict(person, bbox_xyxy=list(person['bbox_xyxy'])) for person in people],
                    tracks={person['track_id']: dict(reason=None,
                        anchor=dict(mode=self._features[person['track_id']]['mode'],
                            histogram=self._features[person['track_id']]['histogram'].copy()))
                        for person in people if person['selectable']}))
                self._intent_cache = self._intent_cache[-self.intent_max_snapshots:]
            return observation

    return LiveSelection()


def _neural_vision():
    # Conventional modes never enter this branch or import the Flyvis runtime.
    from experiments.mantis_vision import MantisVision
    return MantisVision()


def _conventional_detector():
    import cv2
    from perception.detector import YOLOXDetector
    cv2.setNumThreads(2)
    # The same official detector and frozen operating point as FlightVision.
    return YOLOXDetector(ROOT/'models/yolox_tiny_official/yolox_tiny.onnx',
        expected_sha256='427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7',
        confidence=.5, nms_iou=.45, input_size=416)


def conventional_candidate(method, frame, sequence, observation, surface, angular_filter):
    """Actual selected-box bearing, with the frozen common cue/depth envelope."""
    from experiments.flight_guidance import cue_geometry
    from experiments.mantis_comparison import BearingInput, direct_bearing
    if method not in ('direct', 'filtered'):
        raise ValueError('Conventional guidance must be direct or filtered')
    frame.validate()
    cue = cue_geometry(observation, frame.rgb.shape[:2])
    candidate = dict(valid=False, reason='target_or_cue_unavailable',
        heading_world_rad=None, surface_optical_z_m=None, decoded=None,
        cue_input=None if cue is None else cue.tolist(),
        capture_time_s=float(frame.capture_time_s), track_id=observation.get('track_id'))
    sensor = BearingInput(sequence, frame.capture_time_s, frame.rgb.shape[:2],
        observation.get('bbox_xyxy') if observation.get('valid') else None,
        frame.intrinsics, frame.rotation_world_camera)
    bearing = direct_bearing(sensor)
    filtered = angular_filter.update(frame.capture_time_s, bearing)
    heading = bearing if method == 'direct' else filtered
    if heading is None:
        return candidate
    if not frame.registration_verified or not isinstance(surface, dict):
        return dict(candidate, reason='target_depth_unavailable')
    z, fraction, spread = (surface.get('surface_optical_z_m'),
        surface.get('valid_depth_fraction', 0.), surface.get('depth_spread_p90_p10_m', float('inf')))
    try:
        supported = (not isinstance(z, bool) and np.isfinite([z, fraction, spread]).all()
                     and .3 <= z <= 20 and fraction >= .8 and spread <= .35)
    except (TypeError, ValueError):
        supported = False
    if not supported:
        return dict(candidate, reason='target_depth_unsupported')
    return dict(candidate, valid=True, reason=method+'_bearing_and_registered_depth',
                heading_world_rad=float(heading), surface_optical_z_m=float(z))


class ConventionalVision:
    """Shared YOLO/association/surface contracts without a neural model object."""
    def __init__(self, method, detector):
        from experiments.mantis_comparison import AlphaBeta
        self.method, self.detector = method, detector
        self.angular_filter = AlphaBeta()
        self.sequence = 0
        self.pipeline = self.bridge = None

    def reset_guidance(self):
        from experiments.mantis_comparison import AlphaBeta
        self.angular_filter = AlphaBeta()

    def process(self, frame):
        from experiments.mantis_depth import coherent_foreground_surface
        from perception.pipeline import CameraSample
        frame.validate()
        self.pipeline.tracker.prepare(frame)
        sample = CameraSample(frame.rgb, frame.capture_time_s, self.pipeline.clock(), self.sequence,
            'px4_sih', 'host_monotonic_source_lower_bound', 'px4_sih_front_optical',
            capture_age_at_receive_s=0., depth_m=frame.depth_m, depth_time_s=frame.capture_time_s,
            camera_intrinsics=frame.intrinsics, registration_verified=frame.registration_verified)
        detections = self.pipeline.process(sample)
        observation = self.bridge.update(detections, frame.capture_time_s, frame.rgb.shape[:2], .65)
        selected_surface = None
        fx, fy, cx, cy = frame.intrinsics
        for detection in detections.get('detections', []):
            if detection['class_id'] != 0:
                continue
            box, track_id = detection['bbox_xyxy'], detection['track_id']
            surface = coherent_foreground_surface(frame, box)
            detection['original_central_box_surface'] = detection.get('surface_measurement')
            detection['surface_measurement'] = surface
            self.pipeline.tracker.anchors.pop(track_id, None)
            if surface is not None:
                z = surface['surface_optical_z_m']
                pixels = np.array([[u, v] for u in (box[0], box[2]) for v in (box[1], box[3])])
                optical = np.column_stack([(pixels[:, 0]-cx)*z/fx,
                                           (pixels[:, 1]-cy)*z/fy, np.full(4, z)])
                corners = optical @ frame.rotation_world_camera.T + frame.position_world_camera
                self.pipeline.tracker.anchors[track_id] = dict(corners=corners,
                    capture_time_s=float(frame.capture_time_s))
            if track_id == observation.get('track_id') and observation['valid']:
                selected_surface = surface
        candidate = conventional_candidate(self.method, frame, self.sequence, observation,
                                            selected_surface, self.angular_filter)
        result = dict(sequence=self.sequence, observation=observation, detections=detections,
                      surface=selected_surface, candidate=candidate)
        self.sequence += 1
        return result


class PerceptionEngine:
    """One worker's state; constructor branches before any neural loading."""
    def __init__(self, session, selected_track=None, *, method='neural', trajectory='stationary',
                 detector=None, clock=time.monotonic, neural_factory=None):
        if not isinstance(session, str) or not 1 <= len(session) <= 256:
            raise ValueError('Perception session identity is required')
        if method not in GUIDANCE_METHODS or trajectory not in TRAJECTORIES:
            raise ValueError('Unsupported guidance method or trajectory')
        if selected_track is not None and (type(selected_track) is not int or selected_track < 1):
            raise ValueError('Initial selected track must be a positive integer or None')
        from experiments.flight_tracking import EgoMotionTracker
        from experiments.mantis_studio_vision import StudioPersonRetryDetector
        self.session, self.method, self.clock = session, method, clock
        self.selected_track, self.selected_once = selected_track, False
        self.auto_select = selected_track is not None
        if method == 'neural':
            self.vision = (neural_factory or _neural_vision)()
            tracker = self.vision.pipeline.tracker
            backend = self.vision.detector if detector is None else detector
            self.model = self.vision.frozen
        else:
            backend = _conventional_detector() if detector is None else detector
            self.vision = ConventionalVision(method, backend)
            tracker = EgoMotionTracker(max_age_s=3., appearance_threshold=.65)
            self.model = None
        self.vision.detector = StudioPersonRetryDetector(backend, clock=clock)
        self.vision.pipeline = _live_pipeline(self.vision.detector, session, clock)
        self.vision.pipeline.tracker = tracker
        self.guard = self.vision.bridge = _live_selection(clock)
        self.association_memory = StationaryAssociationMemory() if trajectory == 'stationary' else None
        self.last_sequence = -1
        self.flyvis_calls = 0

    def warmup(self):
        if self.method == 'neural':
            # These pre-arm calls carry no observation/command authority.
            for _ in range(2):
                self.vision.brain.step(np.full((391, 391), .5, np.float32), .1)
            self.vision.brain.reset()
        return dict(kind='ready', model=self.model, method=self.method,
                    flyvis_used=self.method == 'neural')

    def _reset_guidance(self):
        if self.method == 'neural':
            self.vision.brain.reset()
        else:
            self.vision.reset_guidance()

    def _selection_action(self, selection):
        action = selection_request(selection)
        if action is None:
            return None
        self.auto_select = False
        receipt = self.guard.request_selection(action)
        self._reset_guidance()
        return receipt

    def process(self, sequence, frame, *, selection=None):
        if type(sequence) is not int or sequence <= self.last_sequence:
            raise ValueError('Worker frame sequence must increase')
        frame.validate()
        action = self._selection_action(selection)
        self.last_sequence = sequence
        self.guard.observe_frame(frame)
        tracker = self.vision.pipeline.tracker
        if self.association_memory is not None:
            self.association_memory.prepare(tracker, frame)
        self.vision.detector.last_receipt = None
        self.vision.sequence = sequence
        result = self.vision.process(frame)
        association_receipts = [] if self.association_memory is None else self.association_memory.complete(
            tracker, result, frame)
        status = self.guard.status()
        if self.auto_select and not self.selected_once and any(
                p['track_id'] == self.selected_track and p['selectable'] for p in status['people']):
            try:
                self.guard.select(self.selected_track, status['sequence'], now_s=self.clock())
                self.selected_once = True
                action = dict(request=dict(track_id=self.selected_track, sequence=status['sequence']),
                              accepted=True, reason='initial_selection', applied_at_s=self.clock())
            except ValueError:
                pass  # Initial preview expired; wait for a fresh observation.
        neural = self.method == 'neural'
        self.flyvis_calls += int(neural)
        return dict(kind='result', session=self.session, sequence=sequence, method=self.method,
            capture_time_s=frame.capture_time_s, completed_s=self.clock(),
            candidate=result['candidate'], observation=result['observation'],
            detections=result['detections'], selection=self.guard.status(), selection_action=action,
            selection_history=self.guard.selection_history(),
            detector_receipt=self.vision.detector.last_receipt,
            stationary_association_memory=association_receipts,
            association_anchors=[dict(track_id=tid, capture_time_s=anchor['capture_time_s'],
                retained_for_depth_gap=bool(self.association_memory is not None
                    and tid in self.association_memory.grace)) for tid, anchor in tracker.anchors.items()],
            reprojections=tracker.last_reprojections,
            input_finite_depth_pixels=int(np.isfinite(frame.depth_m).sum()),
            flyvis_used=neural, flyvis_observations=self.flyvis_calls,
            neural_features=result['supplied_features'].tolist() if neural else [],
            neural_valid=bool(result['neural']['valid']) if neural else False,
            neural_elapsed_s=result['neural']['elapsed_wall_s'] if neural else 0.)


def _worker(incoming, outgoing, session, selected_track, method='neural', trajectory='stationary'):
    try:
        engine = PerceptionEngine(session, selected_track, method=method, trajectory=trajectory)
        outgoing.put(engine.warmup())
        while True:
            packet = incoming.get()
            if packet is None:
                return
            sequence, frame, delay, selection = packet
            if delay:
                time.sleep(delay)  # Deliberate worker stall; capture time is unchanged.
            outgoing.put(engine.process(sequence, frame, selection=selection))
    except BaseException:
        outgoing.put(dict(kind='error', error=traceback.format_exc()))


class VisionWorker:
    def __init__(self, session, selected_track=None, *, method='neural', trajectory='stationary'):
        if method not in GUIDANCE_METHODS or trajectory not in TRAJECTORIES:
            raise ValueError('Unsupported guidance method or trajectory')
        context = mp.get_context('spawn')
        self.incoming, self.outgoing = context.Queue(1), context.Queue(2)
        self.process = context.Process(target=_worker,
            args=(self.incoming, self.outgoing, session, selected_track, method, trajectory), daemon=True)
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

    def submit(self, frame, delay=0., selection=None):
        if (isinstance(delay, bool) or not isinstance(delay, (int, float))
                or not math.isfinite(delay) or not 0 <= delay <= 30):
            raise ValueError('Worker delay must be in [0, 30] seconds')
        selection = selection_request(selection)
        if self.busy or frame.capture_time_s <= self.last_capture:
            return False
        self.incoming.put_nowait((self.sequence, frame, delay, selection))
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
