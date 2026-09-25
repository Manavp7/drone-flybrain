"""Model-free SIH perception contracts; no PX4 process or native rendering."""
from dataclasses import replace
import builtins
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.flight_contracts import CameraFrame, FlightState, R_BODY_CAMERA, rotation_from_euler
from experiments.px4_camera import (ConventionalVision, PerceptionEngine, ScenarioClock,
    SihCamera, VisionWorker, conventional_candidate, drop_frame_depth, selection_request)
from perception.detector import Detection


BOX = (160., 130., 230., 260.)


class Clock:
    def __init__(self, now=10.02):
        self.now = now

    def __call__(self):
        return self.now


class Detector:
    def __init__(self):
        self.output = [Detection(BOX, .9, 0, 'person')]
        self.calls = 0

    def detect(self, rgb):
        self.calls += 1
        return self.output


def frame(at=10., depth=4., yaw=0.):
    return CameraFrame(np.full((391, 391, 3), [220, 20, 20], np.uint8),
        np.full((391, 391), depth, np.float32), (280., 280., 195., 195.),
        rotation_from_euler(yaw=yaw) @ R_BODY_CAMERA, np.array([.25, 0., 1.1]), at)


class SceneTests(unittest.TestCase):
    def test_model_startup_does_not_shift_motion_phase(self):
        first, second = ScenarioClock(), ScenarioClock()
        self.assertEqual(first.elapsed(1.), 0.)
        self.assertEqual(second.elapsed(90.), 0.)
        first.start(2.)
        second.start(120.)
        self.assertAlmostEqual(first.elapsed(2.7), second.elapsed(120.7))
        self.assertEqual(first.elapsed(2.7), first.elapsed(2.7))
        with self.assertRaises(ValueError):
            first.elapsed(2.6)
        with self.assertRaises(ValueError):
            first.start(3.)

    def test_drop_changes_only_owned_depth_array(self):
        original = frame()
        dropped = drop_frame_depth(original)
        self.assertIs(dropped.rgb, original.rgb)
        self.assertTrue(np.isfinite(original.depth_m).all())
        self.assertTrue(np.isnan(dropped.depth_m).all())
        self.assertEqual(dropped.capture_time_s, original.capture_time_s)
        self.assertEqual(dropped.intrinsics, original.intrinsics)
        self.assertIs(dropped.rotation_world_camera, original.rotation_world_camera)
        self.assertTrue(dropped.registration_verified)

    def test_renderer_truth_evaluation_never_enters_estimated_frame(self):
        data = SimpleNamespace(qpos=np.zeros(7), qvel=np.zeros(6), time=0.)
        seen = []
        def render(safety=False):
            return replace(frame(data.time), position_world_camera=np.array([.25, 0., 0.]))
        def evaluate(actual):
            np.testing.assert_array_equal(actual.position_world_camera, [.25, 0., 0.])
            self.assertEqual(actual.capture_time_s, data.time)
            return [dict(actor_id='blue', bbox_xyxy=[1, 2, 3, 4], visible=True)]
        world = SimpleNamespace(data=data, capture=render, evaluation_people=evaluate,
            update_scene=lambda at, **kwargs: seen.append((at, kwargs)))
        truth = SimpleNamespace(lat=470000000, lon=80000000, alt=500000,
            vx=0, vy=0, vz=0, rollspeed=0, pitchspeed=0, yawspeed=0,
            attitude_quaternion=[1, 0, 0, 0])
        with patch('experiments.mantis_arena.ArenaWorld', return_value=world):
            camera = SihCamera(truth, -10000., trajectory='crossing', target_speed=.2)
        camera.start_scenario(20.)
        link = SimpleNamespace(truth=truth, truth_source=20.5, attitude_source=20.5,
            telemetry=SimpleNamespace(boot_ms=20500),
            clock=SimpleNamespace(capture_bound=lambda *args: 99.95))
        state = FlightState(99.95, np.array([10., 20., 1.1]), np.zeros(3),
                            np.eye(3), np.zeros(3), np.zeros(4))
        captured = camera.capture(link, state, drop_depth=True)
        np.testing.assert_array_equal(captured.position_world_camera, [10.25, 20., 1.1])
        self.assertEqual(captured.capture_time_s, 99.95)
        self.assertNotIn('projections', vars(captured))
        self.assertEqual(camera.last_evaluation['projections'][0]['actor_id'], 'blue')
        self.assertEqual(camera.last_evaluation['capture_time_s'], captured.capture_time_s)
        self.assertEqual(camera.last_evaluation['scenario_time_s'], .5)
        self.assertEqual(seen[-1][1]['trajectory'], 'crossing')
        self.assertEqual(seen[-1][1]['target_speed'], .2)
        self.assertTrue(camera.last_evaluation['depth_drop'])
        self.assertTrue(np.isnan(captured.depth_m).all())
        with self.assertRaises(ValueError):
            camera.capture(link, state, drop_depth=True)
        with self.assertRaises(ValueError):
            camera.capture(link, state, safety=True, drop_depth=True)
        old_evaluation = camera.last_evaluation
        camera.capture(link, state, safety=True)
        self.assertIs(camera.last_evaluation, old_evaluation)


class GuidanceTests(unittest.TestCase):
    def make(self, method='direct', selected_track=1, trajectory='stationary'):
        clock, detector = Clock(), Detector()
        engine = PerceptionEngine('owned-session', selected_track, method=method,
            trajectory=trajectory, detector=detector, clock=clock)
        return engine, clock, detector

    def test_conventional_modes_do_not_import_or_construct_flyvis(self):
        original_import = builtins.__import__
        def guarded_import(name, *args, **kwargs):
            if name.startswith(('flyvis', 'torch', 'experiments.mantis_vision',
                                'experiments.hybrid_flyvis', 'experiments.runtime')):
                raise AssertionError('Conventional mode imported neural runtime: '+name)
            return original_import(name, *args, **kwargs)
        with patch('builtins.__import__', side_effect=guarded_import), \
                patch('experiments.px4_camera._neural_vision', side_effect=AssertionError('neural factory')):
            for method in ('direct', 'filtered'):
                engine, clock, detector = self.make(method)
                self.assertIsInstance(engine.vision, ConventionalVision)
                self.assertFalse(engine.warmup()['flyvis_used'])
                engine.process(0, frame())
                clock.now = 10.12
                result = engine.process(1, frame(10.1))
                self.assertTrue(result['candidate']['valid'])
                self.assertFalse(result['flyvis_used'])
                self.assertEqual(result['flyvis_observations'], 0)
                self.assertEqual(result['neural_features'], [])
                self.assertEqual(detector.calls, 2)

    def test_filtered_has_same_first_measurement_then_causal_smoothing(self):
        engines = [self.make(method) for method in ('direct', 'filtered')]
        first, changed = [], []
        for engine, clock, detector in engines:
            engine.process(0, frame())
            clock.now = 10.12
            first.append(engine.process(1, frame(10.1))['candidate']['heading_world_rad'])
            detector.output = [Detection((170., 130., 240., 260.), .9, 0, 'person')]
            clock.now = 10.22
            changed.append(engine.process(2, frame(10.2))['candidate']['heading_world_rad'])
        self.assertEqual(first[0], first[1])
        self.assertLess(abs(changed[1]-first[1]), abs(changed[0]-first[0]))

    def test_common_cue_depth_registration_and_missing_input_limits(self):
        from experiments.mantis_comparison import AlphaBeta
        good = dict(valid=True, bbox_xyxy=BOX, track_id=1)
        surface = dict(surface_optical_z_m=4., valid_depth_fraction=1., depth_spread_p90_p10_m=.1)
        for method in ('direct', 'filtered'):
            self.assertTrue(conventional_candidate(method, frame(), 0, good, surface, AlphaBeta())['valid'])
            for observation, sample, measured in [
                    (dict(good, valid=False), frame(), surface),
                    (dict(good, bbox_xyxy=(310., 130., 380., 260.)), frame(), surface),
                    (good, frame(), None),
                    (good, replace(frame(), registration_verified=False), surface),
                    (good, frame(), dict(surface, valid_depth_fraction=.79)),
                    (good, frame(), dict(surface, depth_spread_p90_p10_m=.36))]:
                self.assertFalse(conventional_candidate(method, sample, 0, observation,
                                                       measured, AlphaBeta())['valid'])

    def test_stale_source_is_rejected_before_detector(self):
        engine, clock, detector = self.make()
        clock.now = 10.7
        result = engine.process(0, frame())
        self.assertEqual(detector.calls, 0)
        self.assertFalse(result['candidate']['valid'])
        self.assertEqual(result['detections']['reason'], 'stale_camera_frame')

    def test_depth_gap_receipt_does_not_grant_current_guidance(self):
        engine, clock, detector = self.make()
        engine.process(0, frame())
        clock.now = 10.12
        engine.process(1, frame(10.1))
        clock.now = 10.22
        missing = engine.process(2, drop_frame_depth(frame(10.2)))
        self.assertFalse(missing['candidate']['valid'])
        self.assertEqual(missing['candidate']['reason'], 'target_depth_unavailable')
        self.assertEqual(missing['input_finite_depth_pixels'], 0)
        self.assertEqual(missing['stationary_association_memory'][0]['anchor_capture_time_s'], 10.1)
        clock.now = 10.32
        recovered = engine.process(3, frame(10.3))
        self.assertTrue(recovered['candidate']['valid'])
        self.assertEqual(recovered['observation']['track_id'], 1)
        self.assertEqual(recovered['reprojections'][0]['anchor_capture_time_s'], 10.1)
        self.assertFalse(recovered['association_anchors'][0]['retained_for_depth_gap'])

    def test_motion_scenarios_never_enable_stationary_grace(self):
        for trajectory in ('walk', 'crossing', 'occlusion'):
            engine, clock, detector = self.make(trajectory=trajectory)
            self.assertIsNone(engine.association_memory)
            engine.process(0, frame())
            clock.now = 10.12
            missing = engine.process(1, drop_frame_depth(frame(10.1)))
            self.assertEqual(missing['stationary_association_memory'], [])
            self.assertEqual(missing['association_anchors'], [])

    def test_depth_drop_during_turn_preserves_observed_id_without_reselection(self):
        from perception.pipeline import iou
        engine, clock, detector = self.make()
        initial = frame()
        pixels = np.array([[u, v] for u in (BOX[0], BOX[2]) for v in (BOX[1], BOX[3])])
        optical = np.column_stack([(pixels[:, 0]-195)*4/280,
                                   (pixels[:, 1]-195)*4/280, np.full(4, 4.)])
        world = optical @ initial.rotation_world_camera.T + initial.position_world_camera
        boxes, packets = [], []
        for sequence, at, yaw, missing in [(0, 10., 0., False), (1, 10.3, .01, True),
                                           (2, 10.9, .24, False)]:
            current = frame(at, yaw=yaw)
            local = (world-current.position_world_camera) @ current.rotation_world_camera
            projected = local[:, :2]/local[:, 2:]*280+195
            box = tuple(np.r_[projected.min(0), projected.max(0)])
            boxes.append(box)
            detector.output = [Detection(box, .9, 0, 'person')]
            clock.now = at+.02
            packets.append(engine.process(sequence, drop_frame_depth(current) if missing else current))
        self.assertLess(iou(boxes[1], boxes[2]), .3)
        self.assertFalse(packets[1]['candidate']['valid'])
        self.assertEqual(packets[1]['stationary_association_memory'][0]['anchor_capture_time_s'], 10.)
        self.assertTrue(packets[2]['observation']['valid'])
        self.assertEqual(packets[2]['observation']['track_id'], 1)
        self.assertTrue(packets[2]['candidate']['valid'])
        self.assertIsNone(packets[2]['selection_action'])
        self.assertEqual(packets[2]['reprojections'][0]['anchor_capture_time_s'], 10.)

    def test_neural_and_baselines_share_detector_selection_tracker_and_surface(self):
        # A fake brain tests wiring only; no official weights or model inference.
        from tests.test_mantis_tracking import make_vision
        neural_vision = make_vision()
        neural_vision.frozen = {'fixture': 'fake neural model for API test only'}
        original_step = neural_vision.brain.step
        neural_vision.brain.step = lambda *args: dict(original_step(*args), elapsed_wall_s=0.)
        clock, detector = Clock(), Detector()
        neural = PerceptionEngine('owned-session', 1, detector=detector, clock=clock,
                                  neural_factory=lambda: neural_vision)
        baseline, base_clock, base_detector = self.make()
        for field in ('minimum_iou', 'max_age_s', 'max_tracks', 'appearance_threshold'):
            self.assertEqual(getattr(neural.vision.pipeline.tracker, field),
                             getattr(baseline.vision.pipeline.tracker, field))
        packets = []
        for engine, source_clock in [(neural, clock), (baseline, base_clock)]:
            engine.process(0, frame())
            source_clock.now = 10.12
            packets.append(engine.process(1, frame(10.1)))
        first, second = packets
        self.assertEqual(first['observation'], second['observation'])
        self.assertEqual(first['selection'], second['selection'])
        self.assertEqual(first['detections']['detections'], second['detections']['detections'])
        self.assertEqual(first['candidate']['surface_optical_z_m'], second['candidate']['surface_optical_z_m'])
        self.assertTrue(first['flyvis_used'])
        self.assertEqual(first['flyvis_observations'], 2)
        self.assertFalse(second['flyvis_used'])
        self.assertEqual(second['flyvis_observations'], 0)


class SelectionTests(unittest.TestCase):
    make = GuidanceTests.make
    def test_none_means_no_selection_and_explicit_previous_preview_selects(self):
        engine, clock, detector = self.make(selected_track=None)
        first = engine.process(0, frame())
        self.assertIsNone(first['selection']['track_id'])
        clock.now = 10.12
        chosen = engine.process(1, frame(10.1), selection={'track_id': 1, 'sequence': 0})
        self.assertTrue(chosen['selection_action']['accepted'])
        self.assertEqual(chosen['selection_action']['request']['sequence'], 0)
        self.assertTrue(chosen['candidate']['valid'])
        self.assertEqual(chosen['candidate']['track_id'], 1)
        self.assertNotIn('projections', chosen)
        self.assertNotIn('truth_position', chosen)

    def test_explicit_clear_disables_initial_auto_selection(self):
        engine, clock, detector = self.make()
        engine.process(0, frame())
        clock.now = 10.12
        cleared = engine.process(1, frame(10.1), selection={'clear': True})
        self.assertTrue(cleared['selection_action']['accepted'])
        self.assertIsNone(cleared['selection']['track_id'])
        clock.now = 10.22
        self.assertIsNone(engine.process(2, frame(10.2))['selection']['track_id'])

    def test_wrong_sequence_or_expired_click_revokes_previous_authority(self):
        for now, sequence in [(10.12, 99), (13.1, 0)]:
            engine, clock, detector = self.make()
            engine.process(0, frame())
            clock.now = now
            rejected = engine.process(1, frame(now-.02),
                                      selection={'track_id': 1, 'sequence': sequence})
            self.assertFalse(rejected['selection_action']['accepted'])
            self.assertIsNone(rejected['selection']['track_id'])
            self.assertFalse(rejected['candidate']['valid'])

    def test_old_displayed_snapshot_selects_only_current_continuous_geometry(self):
        engine, clock, detector = self.make(selected_track=None)
        engine.process(0, frame())
        for sequence in range(1, 4):
            clock.now = 10.02+sequence*.2
            engine.process(sequence, frame(10.+sequence*.2))
        detector.output = [Detection((170., 130., 240., 260.), .9, 0, 'person')]
        clock.now = 10.82
        selected = engine.process(4, frame(10.8), selection={'track_id': 1, 'sequence': 0})
        receipt = selected['selection_action']
        self.assertTrue(receipt['accepted'])
        self.assertGreater(receipt['source_age_s'], .65)
        self.assertEqual(receipt['source_capture_time_s'], 10.)
        self.assertEqual(receipt['resolved_capture_time_s'], 10.8)
        self.assertEqual(receipt['resolved_sequence'], 4)
        self.assertEqual(selected['observation']['bbox_xyxy'], [170., 130., 240., 260.])
        self.assertEqual(selected['observation']['capture_time_s'], 10.8)
        self.assertTrue(selected['candidate']['valid'])
        self.assertEqual(selected['candidate']['capture_time_s'], 10.8)
        self.assertEqual(detector.calls, 5)
        self.assertNotIn('anchor', repr(selected['selection_history']))

    def test_cache_is_eight_snapshots_and_three_original_seconds(self):
        engine, clock, detector = self.make(selected_track=None)
        for sequence in range(10):
            clock.now = 10.02+sequence*.1
            result = engine.process(sequence, frame(10.+sequence*.1))
        self.assertEqual([item['sequence'] for item in result['selection_history']], list(range(2, 10)))
        clock.now = 11.02
        evicted = engine.process(10, frame(11.), selection={'track_id': 1, 'sequence': 0})
        self.assertFalse(evicted['selection_action']['accepted'])
        clock.now = 14.02
        expired = engine.process(11, frame(14.), selection={'track_id': 1, 'sequence': 10})
        self.assertFalse(expired['selection_action']['accepted'])
        self.assertEqual(expired['selection_action']['reason'], 'selection_snapshot_unavailable_or_expired')
        self.assertEqual(len(expired['selection_history']), 1)

    @staticmethod
    def striped_frame(at, red_columns):
        sample = frame(at)
        sample.rgb[:, np.arange(sample.rgb.shape[1]) % 10 >= red_columns] = [20, 20, 220]
        return sample

    def test_original_clicked_clothing_anchor_survives_fresh_selection(self):
        from experiments.mantis_selection import _clothing_descriptor
        engine, clock, detector = self.make(selected_track=None)
        original = frame()
        anchor = _clothing_descriptor(original.rgb, BOX)
        engine.process(0, original)
        clock.now = 10.82
        selected = engine.process(1, self.striped_frame(10.8, 6),
                                  selection={'track_id': 1, 'sequence': 0})
        self.assertTrue(selected['selection_action']['accepted'])
        np.testing.assert_array_equal(engine.guard._anchor['histogram'], anchor['histogram'])
        clock.now = 11.02
        changed = engine.process(2, self.striped_frame(11., 1))
        self.assertEqual(changed['detections']['detections'][0]['track_id'], 1)
        self.assertEqual(changed['observation']['reason'], 'selected_appearance_changed')
        self.assertFalse(changed['candidate']['valid'])

    def test_gradual_clothing_change_rejects_old_intent_even_with_same_id(self):
        engine, clock, detector = self.make(selected_track=None)
        engine.process(0, frame())
        clock.now = 10.42
        engine.process(1, self.striped_frame(10.4, 6))
        clock.now = 10.82
        changed = engine.process(2, self.striped_frame(10.8, 1),
                                 selection={'track_id': 1, 'sequence': 0})
        self.assertEqual(changed['detections']['detections'][0]['track_id'], 1)
        self.assertFalse(changed['selection_action']['accepted'])
        self.assertEqual(changed['selection_action']['reason'], 'selection_appearance_changed')
        self.assertIsNone(changed['selection']['track_id'])
        self.assertFalse(changed['candidate']['valid'])

    def test_lost_reappeared_same_id_does_not_revive_old_intent_or_auto_select(self):
        engine, clock, detector = self.make()
        engine.process(0, frame())
        detector.output = []
        clock.now = 10.22
        engine.process(1, frame(10.2))
        detector.output = [Detection(BOX, .9, 0, 'person')]
        clock.now = 10.42
        rejected = engine.process(2, frame(10.4), selection={'track_id': 1, 'sequence': 0})
        self.assertEqual(rejected['detections']['detections'][0]['track_id'], 1)
        self.assertFalse(rejected['selection_action']['accepted'])
        self.assertEqual(rejected['selection_action']['reason'], 'selection_track_lost')
        self.assertIsNone(rejected['selection']['track_id'])
        clock.now = 10.62
        later = engine.process(3, frame(10.6))
        self.assertIsNone(later['selection']['track_id'])
        self.assertFalse(later['candidate']['valid'])
        self.assertIsNone(later['selection_action'])

    def test_intervening_overlap_permanently_rejects_cached_intent(self):
        engine, clock, detector = self.make(selected_track=None)
        engine.process(0, frame())
        detector.output.append(Detection((205., 130., 275., 260.), .8, 0, 'person'))
        clock.now = 10.22
        engine.process(1, frame(10.2))
        detector.output = [Detection(BOX, .9, 0, 'person')]
        clock.now = 10.42
        rejected = engine.process(2, frame(10.4), selection={'track_id': 1, 'sequence': 0})
        self.assertFalse(rejected['selection_action']['accepted'])
        self.assertEqual(rejected['selection_action']['reason'], 'selection_ambiguous')
        self.assertIsNone(rejected['selection']['track_id'])

    def test_fresh_intent_does_not_extend_depth_or_observation_freshness(self):
        for at, now, missing in [(10.8, 10.82, True), (10.1, 10.82, False)]:
            engine, clock, detector = self.make(selected_track=None)
            engine.process(0, frame())
            clock.now = now
            current = drop_frame_depth(frame(at)) if missing else frame(at)
            result = engine.process(1, current, selection={'track_id': 1, 'sequence': 0})
            self.assertFalse(result['candidate']['valid'])
            self.assertEqual(result['selection_action']['accepted'], missing)
            if not missing:
                self.assertIsNone(result['selection']['track_id'])
                self.assertEqual(result['selection_action']['reason'], 'selection_continuity_unavailable')

    def test_stream_change_rejects_cached_intent(self):
        engine, clock, detector = self.make(selected_track=None)
        engine.process(0, frame())
        process = engine.vision.pipeline.process
        def changed_stream(sample):
            return dict(process(sample), stream_id='different-session')
        engine.vision.pipeline.process = changed_stream
        clock.now = 10.22
        result = engine.process(1, frame(10.2), selection={'track_id': 1, 'sequence': 0})
        self.assertFalse(result['selection_action']['accepted'])
        self.assertEqual(result['selection_action']['reason'], 'selection_stream_changed')
        self.assertIsNone(result['selection']['track_id'])
        self.assertEqual(result['selection_history'], [])

    def test_intent_expiring_during_inference_is_rejected_with_fresh_current_frame(self):
        engine, clock, detector = self.make(selected_track=None)
        engine.process(0, frame())
        clock.now = 12.98
        original = detector.detect
        def delayed(rgb):
            clock.now = 13.02
            return original(rgb)
        detector.detect = delayed
        result = engine.process(1, frame(12.9), selection={'track_id': 1, 'sequence': 0})
        self.assertEqual(result['detections']['status'], 'ok')
        self.assertEqual(result['detections']['detections'][0]['track_id'], 1)
        self.assertFalse(result['selection_action']['accepted'])
        self.assertEqual(result['selection_action']['reason'], 'selection_snapshot_expired')
        self.assertIsNone(result['selection']['track_id'])
        self.assertFalse(result['candidate']['valid'])

    def test_failed_intent_disables_unfulfilled_initial_selection(self):
        engine, clock, detector = self.make()
        detector.output = []
        engine.process(0, frame())
        detector.output = [Detection(BOX, .9, 0, 'person')]
        clock.now = 10.22
        result = engine.process(1, frame(10.2), selection={'track_id': 1, 'sequence': 0})
        self.assertFalse(result['selection_action']['accepted'])
        self.assertFalse(engine.auto_select)
        self.assertIsNone(result['selection']['track_id'])
        clock.now = 10.42
        self.assertIsNone(engine.process(2, frame(10.4))['selection']['track_id'])

    def test_contract_rejects_hidden_evaluator_fields_and_boolean_ids(self):
        for action in ({'track_id': True, 'sequence': 0}, {'clear': False},
                       {'track_id': 1, 'sequence': 0, 'actor_id': 'blue'},
                       {'track_id': 1}, {'clear': True, 'sequence': 0}):
            with self.assertRaises(ValueError):
                selection_request(action)

    def test_busy_does_not_consume_pending_action_and_queue_snapshots_it(self):
        worker = object.__new__(VisionWorker)
        worker.busy, worker.sequence, worker.last_capture = True, 4, 10.
        worker.incoming = Mock()
        action = {'track_id': 1, 'sequence': 3}
        self.assertFalse(worker.submit(frame(10.1), selection=action))
        worker.incoming.put_nowait.assert_not_called()
        worker.busy = False
        self.assertTrue(worker.submit(frame(10.1), selection=action))
        action['track_id'] = 2
        packet = worker.incoming.put_nowait.call_args.args[0]
        self.assertEqual(packet[0], 4)
        self.assertEqual(packet[3], {'track_id': 1, 'sequence': 3})


if __name__ == '__main__':
    unittest.main()
