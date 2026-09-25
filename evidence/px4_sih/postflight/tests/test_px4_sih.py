"""Clock/authority/coordinate regressions; not substitutes for actual SIH runs."""
from dataclasses import replace
from types import SimpleNamespace
import math
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.flight_contracts import FlightState
from experiments.flight_guidance import active_request
from experiments.px4_camera import (D, quaternion_rotation, estimated_state, VisionWorker,
                                    StationaryAssociationMemory)
from experiments.px4_follow import ResultGate, score, following_metrics, following_checks
from experiments.px4_recording import AsyncRecorder
from flybrain_sim.px4_sih import (SourceClock, SihLink, FOLLOW_MASK, launch_environment,
                                validate_build_receipt, PX4_COMMIT)
from flybrain_sim.px4_sitl import SitlError


class ClockTests(unittest.TestCase):
    def test_stale_build_recipe_cannot_mislabel_sensor_profile(self):
        receipt = dict(px4_commit=PX4_COMMIT, binary_sha256='current',
                       overlay_sha256='current', startup_tree_sha256='current')
        with patch('flybrain_sim.px4_sih.sha', return_value='current'), \
                patch('flybrain_sim.px4_sih.tree_sha', return_value='current'):
            with self.assertRaises(SitlError):
                validate_build_receipt(receipt)
            with self.assertRaises(SitlError):
                validate_build_receipt(dict(receipt, build_recipe_sha256='old'))
            validate_build_receipt(dict(receipt, build_recipe_sha256='current'))

    def test_low_noise_sensor_mode_requires_explicit_choice(self):
        inherited = dict(MANTIS_SIH_LOW_NOISE_SENSORS='1', PX4_SIM_SPEED_FACTOR='20',
                         PX4_PARAM_COM_ARM_WO_GPS='1', PATH='/bin')
        normal = launch_environment(inherited)
        self.assertNotIn('MANTIS_SIH_LOW_NOISE_SENSORS', normal)
        self.assertNotIn('PX4_PARAM_COM_ARM_WO_GPS', normal)
        self.assertEqual(normal['PX4_SIM_SPEED_FACTOR'], '1')
        self.assertEqual(normal['PATH'], '/bin')
        self.assertEqual(launch_environment(inherited, True)['MANTIS_SIH_LOW_NOISE_SENSORS'], '1')

    def test_encoding_backpressure_is_bounded_and_drops_without_blocking(self):
        started, resume = threading.Event(), threading.Event()
        seen = []
        def append(at, rgb, overview, row):
            started.set()
            if not resume.wait(3):
                raise RuntimeError('Test encoder wait timed out')
            seen.append((at, rgb.copy()))
        recorder = Mock(append=append, finish=Mock(return_value=dict(status='completed', error=None)))
        worker = AsyncRecorder(recorder)
        frame = np.ones((2, 2, 3), dtype=np.uint8)
        try:
            self.assertTrue(worker.append(0, frame, frame, {}))
            self.assertTrue(started.wait(1))
            self.assertTrue(worker.append(1, frame, frame, {}))
            self.assertTrue(worker.append(2, frame, frame, {}))
            self.assertFalse(worker.append(3, frame, frame, {}))
            frame[:] = 0
        finally:
            resume.set()
            worker.finish(dict(passed=True))
        self.assertEqual([item[0] for item in seen], [0, 1, 2])
        self.assertTrue(all(image.all() for _, image in seen))
        self.assertEqual(recorder.finish.call_args.args[0]['recording_dropped_samples'], 1)

    def test_encoder_error_receipt_cannot_be_reported_as_success(self):
        recorder = Mock(finish=Mock(return_value=dict(status='error', error='encoder exited')))
        worker = AsyncRecorder(recorder)
        with self.assertRaisesRegex(RuntimeError, 'Recording worker failed'):
            worker.finish(dict(passed=True))
        self.assertFalse(worker.thread.is_alive())

    def sync(self, clock, host, remote, rtt=.005):
        nonce = clock.request(host)
        return clock.receive(nonce, round(remote*1e9), host+rtt)

    def test_conservative_bound_survives_simulation_slowdown(self):
        c = SourceClock()
        self.assertTrue(self.sync(c, 100, 1))
        self.assertEqual(c.capture_bound(1.001, 100.08, .1), 100)
        # Source advanced a millisecond, host advanced a second. Arrival of
        # a recent sync must never re-date an older buffered source sample.
        self.assertTrue(self.sync(c, 101, 1.002))
        with self.assertRaises(SitlError):
            c.capture_bound(1.001, 101.02, .1)

    def test_stall_and_duplicate_never_renew_capture(self):
        c = SourceClock()
        self.sync(c, 10, 3)
        self.assertFalse(self.sync(c, 10.1, 3))
        with self.assertRaises(SitlError):
            c.capture_bound(3, 10.11, .1)
        with self.assertRaises(SitlError):
            c.capture_bound(3.001, 10.2, .1)

    def test_nonce_replay_rtt_and_clock_reset(self):
        c = SourceClock()
        n = c.request(5)
        self.assertTrue(c.receive(n, 2_000_000_000, 5.01))
        self.assertFalse(c.receive(n, 3_000_000_000, 5.02))
        self.assertFalse(self.sync(c, 6, 3, rtt=.1))
        self.assertFalse(self.sync(c, 7, 1))
        with self.assertRaises(SitlError):
            c.capture_bound(4, 7.02)


class StationaryAssociationTests(unittest.TestCase):
    def setUp(self):
        self.memory = StationaryAssociationMemory()
        self.anchor = dict(corners=np.ones((4, 3)), capture_time_s=10.)
        self.tracker = SimpleNamespace(anchors={1: self.anchor}, tracks={1: {'time': 10.}})
        self.result = dict(candidate=dict(valid=False, reason='target_depth_unavailable'),
                           detections=dict(detections=[dict(track_id=1, class_id=0, surface_measurement=None)]))

    def gap(self, capture):
        frame = SimpleNamespace(capture_time_s=capture, registration_verified=True)
        self.memory.prepare(self.tracker, frame)
        self.tracker.anchors.clear()  # unchanged base policy drops unavailable current depth
        return self.memory.complete(self.tracker, self.result, frame)

    def test_one_gap_preserves_original_anchor_without_depth_or_authority(self):
        receipt = self.gap(10.4)
        self.assertEqual(receipt[0]['anchor_capture_time_s'], 10.)
        self.assertEqual(self.tracker.anchors[1]['capture_time_s'], 10.)
        np.testing.assert_array_equal(self.tracker.anchors[1]['corners'], self.anchor['corners'])
        self.assertFalse(self.result['candidate']['valid'])
        self.assertIsNone(self.result['detections']['detections'][0]['surface_measurement'])
        self.assertEqual(self.gap(10.8), [])
        self.assertNotIn(1, self.tracker.anchors)

    def test_anchor_expiry_does_not_use_refreshed_track_time(self):
        self.tracker.tracks[1]['time'] = 11.4
        self.assertEqual(self.gap(11.4), [])
        self.assertFalse(self.tracker.anchors)

    def test_absent_match_and_clock_regression_clear_memory(self):
        self.gap(10.4)
        self.result['detections']['detections'] = []
        self.assertEqual(self.gap(10.6), [])
        self.assertFalse(self.tracker.anchors)
        self.tracker.anchors[1] = self.anchor
        self.assertEqual(self.gap(10.3), [])
        self.assertFalse(self.tracker.anchors)

    def test_actual_camera_turn_can_match_after_one_missing_depth_frame(self):
        from experiments.flight_tracking import EgoMotionTracker
        from tests.test_flight_tracking import frame, detection, projected_box
        tracker = EgoMotionTracker(max_age_s=1.2, appearance_threshold=.65)
        memory = StationaryAssociationMemory()
        for at, yaw, depth in [(0., 0., 5.), (.4, .1, float('nan')), (.9, .3, 5.)]:
            current = frame(at, yaw=yaw, depth=depth)
            memory.prepare(tracker, current)
            tracker.prepare(current)
            found = tracker.update([detection(projected_box(current))], at, current.rgb)
            self.assertEqual(found[0]['track_id'], 1)
            found[0]['surface_measurement'] = None if not math.isfinite(depth) else {'depth': depth}
            memory.complete(tracker, dict(detections=dict(detections=found)), current)
        self.assertTrue(tracker.last_reprojections[0]['valid'])


class AuthorityTests(unittest.TestCase):
    def test_stall_waits_for_completed_following_milestones(self):
        rows = [dict(phase='follow', sent_speed=.45, position=[x, 0, 1.1])
                for x in np.linspace(0, .24, 10)]
        observations = [dict(command=dict(valid=True), packet=dict(candidate=dict(
            surface_optical_z_m=z))) for z in np.linspace(5, 4.85, 5)]
        self.assertFalse(all(following_checks(following_metrics(rows, observations)).values()))
        rows.append(dict(phase='follow', sent_speed=.45, position=[.31, 0, 1.1]))
        observations.append(dict(command=dict(valid=True), packet=dict(candidate=dict(
            surface_optical_z_m=4.79))))
        self.assertTrue(all(following_checks(following_metrics(rows, observations)).values()))

    def setUp(self):
        self.state = FlightState(10., np.array([0., 0., 1.1]), np.zeros(3),
                                 np.eye(3), np.zeros(3), np.zeros(4))
        self.packet = dict(session='owned', sequence=0, capture_time_s=10., completed_s=10.1,
            selection=dict(track_id=1, held=False), observation=dict(track_id=1, valid=True),
            candidate=dict(capture_time_s=10., valid=True, reason='neural',
                           heading_world_rad=0., surface_optical_z_m=5.))

    def test_capture_anchored_expiry_under_worker_stall(self):
        gate = ResultGate('owned', 1)
        command = gate.accept(self.packet, self.state, 10.2)
        self.assertEqual(command['valid_until_s'], 10.9)
        self.assertEqual(active_request(command, replace(self.state, time_s=10.89))['forward_speed'], .45)
        self.assertEqual(active_request(command, replace(self.state, time_s=10.9))['forward_speed'], 0)

    def test_delayed_queue_result_is_rejected(self):
        command = ResultGate('owned', 1).accept(self.packet, self.state, 10.7)
        self.assertFalse(command['valid'])
        self.assertEqual(command['reason'], 'stale_perception_result')

    def test_session_order_and_selection_are_required(self):
        gate = ResultGate('owned', 1)
        with self.assertRaises(SitlError):
            gate.accept(dict(self.packet, session='other'), self.state, 10.2)
        packet = dict(self.packet, selection=dict(track_id=2, held=False))
        self.assertFalse(gate.accept(packet, self.state, 10.2)['valid'])
        with self.assertRaises(SitlError):
            gate.accept(self.packet, self.state, 10.3)

    def test_setpoint_has_altitude_hold_and_yaw_control(self):
        link = object.__new__(SihLink)
        link.lease, link.connection = Mock(), Mock()
        link.arm_authorized = True
        link.setpoint((.2, -.1, 0), yaw_ned=.3, altitude_ned=-1.1)
        args = link.connection.mav.set_position_target_local_ned_send.call_args.args
        self.assertEqual(args[4], FOLLOW_MASK)
        self.assertEqual(args[5:11], (0, 0, -1.1, .2, -.1, 0))
        self.assertEqual(args[-2], .3)
        with self.assertRaises(SitlError):
            link.setpoint((.5, 0, 0), altitude_ned=-1.1)
        link.arm_authorized = False
        with self.assertRaises(SitlError):
            link.setpoint((0, 0, -1.1), position=True)

    def test_dead_child_prevents_send(self):
        link = object.__new__(SihLink)
        link.lease, link.connection = Mock(), Mock()
        link.lease.verify.side_effect = SitlError('exited')
        link.arm_authorized = True
        with self.assertRaises(SitlError):
            link.setpoint((0, 0, 0), altitude_ned=-1.1)
        link.connection.mav.set_position_target_local_ned_send.assert_not_called()

    def test_empty_or_stopping_only_run_cannot_pass_neural_follow(self):
        events = [dict(event='stable_takeoff'), dict(event='landed_disarmed')]
        self.assertFalse(score([], [], events)['passed'])
        self.assertTrue(score([], [], events, smoke=True)['passed'])


class CoordinateTests(unittest.TestCase):
    def test_east_yaw_becomes_right_in_nwu(self):
        q = [math.sqrt(.5), 0, 0, math.sqrt(.5)]
        rotation = D @ quaternion_rotation(q) @ D
        np.testing.assert_allclose(rotation @ [1, 0, 0], [0, -1, 0], atol=1e-14)
        transformed_q = np.array(q)*[1, 1, -1, -1]
        np.testing.assert_allclose(quaternion_rotation(transformed_q), rotation, atol=1e-14)

    def test_reject_invalid_quaternion(self):
        for q in ([0, 0, 0, 0], [math.nan, 0, 0, 1], [1, 2, 3, 4]):
            with self.assertRaises(SitlError):
                quaternion_rotation(q)

    def test_estimated_state_keeps_acquisition_age(self):
        clock = SourceClock()
        nonce = clock.request(9.9)
        clock.receive(nonce, 1_000_000_000, 9.91)
        link = SimpleNamespace(health_error=lambda now: '', clock=clock, attitude_source=1.01,
            telemetry=SimpleNamespace(position_ned=(1, 2, -3), velocity_ned=(.1, .2, -.3), boot_ms=1010),
            attitude=SimpleNamespace(q1=1, q2=0, q3=0, q4=0, rollspeed=0, pitchspeed=0, yawspeed=0))
        state = estimated_state(link, np.zeros(3), 10.)
        self.assertEqual(state.time_s, 9.9)
        np.testing.assert_allclose(state.position, [1, -2, 3])

    def test_duplicate_capture_not_submitted(self):
        worker = object.__new__(VisionWorker)
        worker.busy, worker.sequence, worker.last_capture = False, 1, 5.
        worker.incoming = Mock()
        self.assertFalse(worker.submit(SimpleNamespace(capture_time_s=5.)))
        worker.incoming.put_nowait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
