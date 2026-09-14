"""Deadline memory regressions using actual association and scripted clocks."""
from dataclasses import replace
import json
import unittest

import numpy as np

from experiments.flight_contracts import CameraFrame
from experiments.flight_tracking import EgoMotionTracker
from experiments.hybrid_target import TargetBridge, salience_mask
from experiments.mantis_pipeline import MantisPerceptionPipeline
from perception.detector import Detection
from perception.pipeline import CameraSample, PerceptionPipeline


BOX = (20., 10., 36., 52.)
PERSON = Detection(BOX, .9, 0, 'person')


class Clock:
    def __init__(self):
        self.values, self.calls = [], 0

    def __call__(self):
        self.calls += 1
        if not self.values:
            raise AssertionError('pipeline made an extra clock call')
        return self.values.pop(0)


class Detector:
    def __init__(self):
        self.output, self.calls = [PERSON], 0

    def detect(self, image):
        self.calls += 1
        if isinstance(self.output, Exception):
            raise self.output
        return self.output


class Fixture:
    def __init__(self, kind=MantisPerceptionPipeline):
        self.detector, self.clock, self.bridge = Detector(), Clock(), TargetBridge()
        self.pipeline = kind(self.detector, max_frame_age_s=.65, clock=self.clock)
        self.pipeline.tracker = EgoMotionTracker(max_age_s=3.)

    def run(self, capture, sequence, *, readings=None, detections=None,
            color=(220, 20, 20), camera_x=0., depth=4., **changes):
        image = np.full((64, 64, 3), 110, np.uint8)
        output = [PERSON] if detections is None else detections
        if isinstance(output, (list, tuple)):
            for detection in output:
                if isinstance(detection, Detection) and isinstance(detection.bbox_xyxy, tuple):
                    box = detection.bbox_xyxy
                    if len(box) == 4 and np.isfinite(box).all():
                        x0, y0, x1, y1 = np.clip(box, 0, 64).astype(int)
                        image[y0:y1, x0:x1] = color
        frame = CameraFrame(image, np.full((64, 64), depth, np.float32),
                            (40., 40., 32., 32.), np.eye(3),
                            np.array([camera_x, 0., 0.]), capture)
        self.pipeline.tracker.prepare(frame)
        sample = CameraSample(image, capture, 100.+capture, sequence,
                              'camera', 'simulation', 'optical',
                              capture_age_at_receive_s=0., depth_m=frame.depth_m,
                              depth_time_s=capture, camera_intrinsics=frame.intrinsics,
                              registration_verified=True)
        sample = replace(sample, **changes)
        self.detector.output = output
        self.clock.values = ([100.+capture, 100.+capture+.01, 100.+capture+.02]
                             if readings is None else list(readings))
        result = self.pipeline.process(sample)
        return result, self.bridge.update(result, capture, (64, 64), .65)

    def seed(self):
        result, observation = self.run(0., 0)
        assert observation['valid'] and observation['track_id'] == 1
        return result


class MantisPipelineTests(unittest.TestCase):
    def assert_rejected_without_observation(self, result, observation, reason):
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['reason'], reason)
        self.assertEqual(result['detections'], [])
        self.assertFalse(result['control_authority'])
        self.assertFalse(observation['valid'])
        self.assertIsNone(observation['bbox_xyxy'])
        self.assertFalse(observation['control_authority'])
        self.assertTrue(np.all(salience_mask(observation, (64, 64)) == .5))
        json.dumps(result, allow_nan=False)

    def test_original_deadline_reset_loses_id_but_mantis_fresh_observation_recovers(self):
        for kind in (PerceptionPipeline, MantisPerceptionPipeline):
            with self.subTest(kind=kind.__name__):
                fixture = Fixture(kind)
                fixture.seed()
                rejected, observation = fixture.run(.2, 1, readings=[100.2, 101.2])
                self.assert_rejected_without_observation(rejected, observation,
                                                        'inference_deadline_missed')
                fresh, observation = fixture.run(1.3, 2)
                self.assertEqual(observation['valid'], kind is MantisPerceptionPipeline)
                self.assertEqual(fresh['detections'][0]['track_id'],
                                 1 if kind is MantisPerceptionPipeline else 2)
                self.assertEqual(fixture.detector.calls, 3)
                self.assertEqual(fixture.clock.calls, 8)

    def test_inference_timeout_preserves_only_previous_observed_times_and_anchor(self):
        fixture = Fixture()
        fixture.seed()
        tracker = fixture.pipeline.tracker
        corners, appearance = tracker.anchors[1]['corners'].copy(), tracker.tracks[1]['appearance'].copy()
        rejected, observation = fixture.run(.2, 1, readings=[100.2, 101.2], depth=7.)
        self.assert_rejected_without_observation(rejected, observation, 'inference_deadline_missed')
        self.assertTrue(rejected['tracking_memory']['preserved_after_deadline'])
        self.assertEqual(tracker.last_time, 0.)
        self.assertEqual(tracker.tracks[1]['time'], 0.)
        self.assertEqual(tracker.anchors[1]['capture_time_s'], 0.)
        np.testing.assert_array_equal(tracker.anchors[1]['corners'], corners)
        np.testing.assert_array_equal(tracker.tracks[1]['appearance'], appearance)
        self.assertEqual(tracker.last_reprojections, [])
        self.assertEqual(tracker.prepared_frame.capture_time_s, .2)
        self.assertEqual(fixture.pipeline.last_sequence, 1)
        self.assertEqual(fixture.pipeline.last_capture, .2)
        self.assertIs(fixture.pipeline.detector, fixture.detector)
        self.assertIs(fixture.pipeline.clock, fixture.clock)

    def test_processing_timeout_rolls_back_updated_anchor_and_burns_new_ids(self):
        fixture = Fixture()
        fixture.seed()
        corners = fixture.pipeline.tracker.anchors[1]['corners'].copy()
        far = Detection((43., 10., 59., 52.), .8, 0, 'person')
        rejected, observation = fixture.run(.2, 1, readings=[100.2, 100.21, 101.2],
                                            detections=[PERSON, far], depth=7.)
        self.assert_rejected_without_observation(rejected, observation, 'processing_deadline_missed')
        tracker = fixture.pipeline.tracker
        self.assertEqual(set(tracker.tracks), {1})
        self.assertEqual(tracker.next_id, 3)
        self.assertEqual(tracker.tracks[1]['time'], 0.)
        self.assertEqual(tracker.anchors[1]['capture_time_s'], 0.)
        np.testing.assert_array_equal(tracker.anchors[1]['corners'], corners)
        result, observation = fixture.run(1.3, 2, detections=[PERSON, far])
        self.assertTrue(observation['valid'])
        self.assertEqual([d['track_id'] for d in result['detections']], [1, 3])

    def test_restored_anchor_compensates_later_camera_motion(self):
        fixture = Fixture()
        fixture.seed()
        fixture.run(.2, 1, readings=[100.2, 101.2])
        shifted = Detection((36., 10., 52., 52.), .9, 0, 'person')
        result, observation = fixture.run(1.3, 2, detections=[shifted], camera_x=-1.6)
        self.assertTrue(observation['valid'])
        self.assertEqual(result['detections'][0]['track_id'], 1)
        self.assertEqual(fixture.pipeline.tracker.last_reprojections[0]['anchor_capture_time_s'], 0.)

    def test_timeout_does_not_renew_memory_and_fresh_frame_after_three_seconds_expires(self):
        fixture = Fixture()
        fixture.seed()
        fixture.run(.2, 1, readings=[100.2, 101.2])
        missing, observation = fixture.run(1.3, 2, detections=[])
        self.assertFalse(observation['valid'])
        self.assertEqual(missing['detections'], [])
        self.assertEqual(fixture.pipeline.tracker.tracks[1]['time'], 0.)
        fixture.run(1.4, 3, readings=[101.4, 102.2])
        expired, observation = fixture.run(3.01, 4)
        self.assertFalse(observation['valid'])
        self.assertNotEqual(expired['detections'][0]['track_id'], 1)

    def test_finish_age_alone_can_expire_memory_during_long_inference(self):
        fixture = Fixture()
        fixture.seed()
        result, observation = fixture.run(.2, 1, readings=[100.2, 103.01])
        self.assert_rejected_without_observation(result, observation, 'inference_deadline_missed')
        self.assertFalse(fixture.pipeline.tracker.tracks)
        self.assertFalse(fixture.pipeline.tracker.anchors)
        self.assertFalse(result['tracking_memory']['preserved_after_deadline'])

    def test_current_matching_clothing_and_geometry_are_still_required(self):
        scenarios = [dict(color=(20, 20, 220)),
                     dict(detections=[Detection((43., 10., 59., 52.), .9, 0, 'person')])]
        for changes in scenarios:
            with self.subTest(changes=changes):
                fixture = Fixture()
                fixture.seed()
                fixture.run(.2, 1, readings=[100.2, 101.2])
                result, observation = fixture.run(1.3, 2, **changes)
                self.assertFalse(observation['valid'])
                self.assertNotEqual(result['detections'][0]['track_id'], 1)

    def test_changed_stream_or_frame_cannot_restore_prior_memory(self):
        for key in ('stream_id', 'clock_domain', 'frame_id'):
            with self.subTest(key=key):
                fixture = Fixture()
                fixture.seed()
                result, observation = fixture.run(.2, 1, readings=[100.2, 101.2],
                                                   **{key: 'different'})
                self.assertFalse(result['tracking_memory']['preserved_after_deadline'])
                self.assertFalse(fixture.pipeline.tracker.tracks)
                self.assertFalse(fixture.pipeline.tracker.anchors)
                self.assertFalse(observation['valid'])

    def test_reordered_input_after_timeout_clears_and_does_not_rollback_watermarks(self):
        fixture = Fixture()
        fixture.seed()
        fixture.run(.2, 1, readings=[100.2, 101.2])
        result, observation = fixture.run(.15, 0, readings=[101.3])
        self.assert_rejected_without_observation(result, observation, 'duplicate_or_reordered_frame')
        self.assertFalse(fixture.pipeline.tracker.tracks)
        self.assertEqual(fixture.pipeline.last_sequence, 1)
        self.assertEqual(fixture.pipeline.last_capture, .2)
        result, observation = fixture.run(1.4, 2)
        self.assertFalse(observation['valid'])
        self.assertNotEqual(result['detections'][0]['track_id'], 1)

    def test_nonincreasing_capture_clears_even_with_increasing_sequence(self):
        fixture = Fixture()
        fixture.seed()
        result, observation = fixture.run(0., 1, readings=[100.1])
        self.assert_rejected_without_observation(result, observation, 'duplicate_or_reordered_frame')
        self.assertFalse(fixture.pipeline.tracker.tracks)

    def test_stale_at_entry_and_future_receive_time_clear(self):
        for readings, reason in (([101.2], 'stale_camera_frame'),
                                 ([100.1], 'future_receive_time')):
            with self.subTest(reason=reason):
                fixture = Fixture()
                fixture.seed()
                result, observation = fixture.run(.2, 1, readings=readings)
                self.assert_rejected_without_observation(result, observation, reason)
                self.assertFalse(result['tracking_memory']['preserved_after_deadline'])
                self.assertFalse(fixture.pipeline.tracker.tracks)
                self.assertEqual(fixture.detector.calls, 1)

    def test_unknown_source_age_is_ineligible_for_verified_timeout_preservation(self):
        fixture = Fixture()
        fixture.seed()
        result, observation = fixture.run(.2, 1, readings=[100.2, 101.2],
                                           capture_age_at_receive_s=None)
        self.assert_rejected_without_observation(result, observation, 'inference_deadline_missed')
        self.assertFalse(result['tracking_memory']['preserved_after_deadline'])
        self.assertFalse(fixture.pipeline.tracker.tracks)

    def test_invalid_clock_clears_even_after_partial_processing(self):
        for values in ([float('nan')], [100.2, float('inf')],
                       [100.2, 100.21, float('nan')], [100.2, 100.1],
                       [100.2, 100.21, 100.205], [99.]):
            with self.subTest(values=values):
                fixture = Fixture()
                fixture.seed()
                with self.assertRaisesRegex(ValueError, 'clock must be finite and monotonic'):
                    fixture.run(.2, 1, readings=values)
                self.assertFalse(fixture.pipeline.tracker.tracks)
                self.assertFalse(fixture.pipeline.tracker.anchors)
                self.assertIs(fixture.pipeline.clock, fixture.clock)
                self.assertIs(fixture.pipeline.detector, fixture.detector)

    def test_invalid_sample_clears_old_memory_before_inference(self):
        fixture = Fixture()
        fixture.seed()
        with self.assertRaises(ValueError):
            fixture.run(.2, 1, capture_age_at_receive_s=float('nan'))
        self.assertFalse(fixture.pipeline.tracker.tracks)
        self.assertFalse(fixture.pipeline.tracker.anchors)
        self.assertEqual(fixture.detector.calls, 1)

    def test_detector_error_clears_without_memory_restore(self):
        fixture = Fixture()
        fixture.seed()
        result, observation = fixture.run(.2, 1, readings=[100.2, 100.21],
                                           detections=RuntimeError('detector failed'))
        self.assertEqual(result['status'], 'detector_error')
        self.assertFalse(result['tracking_memory']['preserved_after_deadline'])
        self.assertFalse(fixture.pipeline.tracker.tracks)
        self.assertFalse(observation['valid'])

    def test_malformed_detector_return_cannot_hide_behind_timeout(self):
        malformed = [object(), [{}], [Detection((-1., 0., 10., 20.), .9, 0, 'person')],
                     [Detection(BOX, float('nan'), 0, 'person')],
                     [Detection(BOX, .9, 'person', 'person')], [PERSON]*1001]
        for output in malformed:
            with self.subTest(output_type=type(output).__name__):
                fixture = Fixture()
                fixture.seed()
                result, observation = fixture.run(.2, 1, readings=[100.2, 101.2],
                                                   detections=output)
                self.assert_rejected_without_observation(result, observation, 'inference_deadline_missed')
                self.assertFalse(result['tracking_memory']['preserved_after_deadline'])
                self.assertFalse(fixture.pipeline.tracker.tracks)
                self.assertFalse(fixture.pipeline.tracker.anchors)

    def test_first_frame_timeout_has_no_established_memory_to_restore(self):
        fixture = Fixture()
        result, observation = fixture.run(0., 0, readings=[100., 101.])
        self.assert_rejected_without_observation(result, observation, 'inference_deadline_missed')
        self.assertFalse(result['tracking_memory']['preserved_after_deadline'])
        fresh, observation = fixture.run(1.1, 1)
        self.assertTrue(observation['valid'])
        self.assertEqual(fresh['detections'][0]['track_id'], 1)


if __name__ == '__main__':
    unittest.main()
