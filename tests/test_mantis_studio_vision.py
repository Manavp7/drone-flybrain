"""Same-frame retry geometry, provenance and actual pipeline deadline gates."""
from dataclasses import replace
import json
import unittest

import numpy as np

from experiments.mantis_pipeline import MantisPerceptionPipeline
from experiments.mantis_studio_vision import StudioPersonRetryDetector
from perception.detector import Detection, DetectorError, ModelContractError
from perception.pipeline import CameraSample


PERSON = Detection((1.25, 2., 7.75, 9.), .73, 0, 'person')
CAR = Detection((2., 1., 8., 8.), .61, 2, 'car')


class Clock:
    def __init__(self):
        self.value = 100.

    def __call__(self):
        return self.value


class ScriptedDetector:
    def __init__(self, *outputs, clock=None, delays=None, mutate=False):
        self.outputs = list(outputs)
        self.images = []
        self.clock = clock
        self.delays = list(delays or [0.] * len(outputs))
        self.mutate = mutate

    def detect(self, image):
        self.images.append(image.copy())
        if self.mutate:
            image[:] = 255
        if self.clock is not None:
            self.clock.value += self.delays.pop(0)
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class StudioVisionTests(unittest.TestCase):
    def setUp(self):
        self.image = np.arange(10*12*3, dtype=np.uint8).reshape(10, 12, 3)

    def wrapper(self, *outputs, **kwargs):
        clock = Clock()
        backend = ScriptedDetector(*outputs, clock=clock, **kwargs)
        return StudioPersonRetryDetector(backend, clock=clock), backend, clock

    def test_person_primary_returns_unchanged_without_retry(self):
        detector, backend, _ = self.wrapper([CAR, PERSON], [PERSON])
        result = detector.detect(self.image)
        self.assertEqual(result, [CAR, PERSON])
        self.assertIs(result[1], PERSON)
        self.assertEqual(len(backend.images), 1)
        self.assertEqual(detector.last_receipt['backend_calls'], 1)
        self.assertFalse(detector.last_receipt['retry_attempted'])
        self.assertEqual(detector.last_receipt['primary_person_count'], 1)

    def test_retry_inverts_continuous_edges_and_retains_original_class_and_score(self):
        edge = Detection((0., 0., 12., 10.), .91, 0, 'person')
        detector, backend, _ = self.wrapper([], [PERSON, edge])
        result = detector.detect(self.image)
        self.assertEqual(result, [replace(PERSON, bbox_xyxy=(4.25, 2., 10.75, 9.)), edge])
        np.testing.assert_array_equal(backend.images[0], self.image)
        np.testing.assert_array_equal(backend.images[1], self.image[:, ::-1])
        self.assertEqual(detector.last_receipt['backend_calls'], 2)
        self.assertEqual(detector.last_receipt['retry_person_count'], 2)
        self.assertTrue(detector.last_receipt['used_retry'])
        json.dumps(detector.last_receipt, allow_nan=False)

    def test_retry_does_not_duplicate_nonperson_classes_or_relabel_them(self):
        detector, _, _ = self.wrapper([CAR], [CAR, PERSON])
        result = detector.detect(self.image)
        self.assertEqual(result[0], CAR)
        self.assertEqual([d.class_id for d in result], [2, 0])
        self.assertEqual(result[1].confidence, PERSON.confidence)

    def test_no_person_in_either_call_returns_primary_without_third_attempt(self):
        detector, backend, _ = self.wrapper([CAR], [CAR], [PERSON])
        self.assertEqual(detector.detect(self.image), [CAR])
        self.assertEqual(len(backend.images), 2)
        self.assertEqual(detector.last_receipt['outcome'], 'retry_no_person')
        self.assertFalse(detector.last_receipt['used_retry'])

    def test_mutating_backend_cannot_overwrite_source_or_contaminate_retry(self):
        before = self.image.copy()
        detector, backend, _ = self.wrapper([], [PERSON], mutate=True)
        detector.detect(self.image)
        np.testing.assert_array_equal(self.image, before)
        np.testing.assert_array_equal(backend.images[1], before[:, ::-1])

    def test_readonly_noncontiguous_source_is_supported_and_unchanged(self):
        source = self.image[:, ::-1]
        source.setflags(write=False)
        before = source.copy()
        detector, backend, _ = self.wrapper([], [PERSON], mutate=True)
        detector.detect(source)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_array_equal(backend.images[1], before[:, ::-1])

    def test_primary_error_propagates_without_retry(self):
        for error in (DetectorError('inference failed'), TimeoutError('deadline')):
            with self.subTest(error=type(error).__name__):
                detector, backend, _ = self.wrapper(error, [PERSON])
                with self.assertRaises(type(error)) as raised:
                    detector.detect(self.image)
                self.assertIs(raised.exception, error)
                self.assertEqual(len(backend.images), 1)
                self.assertEqual(detector.last_receipt['backend_calls_completed'], 0)
                self.assertFalse(detector.last_receipt['retry_attempted'])

    def test_retry_error_propagates_instead_of_returning_primary(self):
        error = DetectorError('retry failed')
        detector, backend, _ = self.wrapper([CAR], error, [PERSON])
        with self.assertRaises(DetectorError) as raised:
            detector.detect(self.image)
        self.assertIs(raised.exception, error)
        self.assertEqual(len(backend.images), 2)
        self.assertEqual(detector.last_receipt['backend_calls_completed'], 1)
        self.assertFalse(detector.last_receipt['used_retry'])

    def test_invalid_flip_geometry_is_rejected_instead_of_clipped(self):
        for box in ((-1., 2., 8., 9.), (1., 2., 13., 9.), (1., 2., 8., 11.),
                    (7., 2., 1., 9.), (1., 2., 1., 9.), (1., 2., float('nan'), 9.),
                    (1., float('inf'), 8., 9.), (1., 2., 8.)):
            with self.subTest(box=box):
                detector, backend, _ = self.wrapper([], [replace(PERSON, bbox_xyxy=box)])
                with self.assertRaises(ModelContractError):
                    detector.detect(self.image)
                self.assertEqual(len(backend.images), 2)
                self.assertFalse(detector.last_receipt['used_retry'])

    def test_invalid_primary_is_an_error_not_a_zero_person_retry(self):
        for output in (None, {}, [object()], [replace(PERSON, class_id=True)],
                       [replace(CAR, label='person')], [replace(PERSON, confidence=float('nan'))]):
            with self.subTest(output=output):
                detector, backend, _ = self.wrapper(output, [PERSON])
                with self.assertRaises(ModelContractError):
                    detector.detect(self.image)
                self.assertEqual(len(backend.images), 1)

    def test_late_primary_does_not_start_retry(self):
        detector, backend, _ = self.wrapper([], [PERSON], delays=[.66, .1])
        self.assertEqual(detector.detect(self.image), [])
        self.assertEqual(len(backend.images), 1)
        self.assertEqual(detector.last_receipt['outcome'], 'retry_skipped_elapsed_budget')

    def test_receipt_and_results_do_not_carry_over_between_frames(self):
        detector, backend, _ = self.wrapper([], [PERSON], [], [])
        self.assertTrue(detector.detect(self.image))
        previous_receipt = detector.last_receipt
        self.assertEqual(detector.detect(self.image), [])
        self.assertIsNot(detector.last_receipt, previous_receipt)
        self.assertTrue(previous_receipt['used_retry'])
        self.assertFalse(detector.last_receipt['used_retry'])
        self.assertEqual(detector.last_receipt['backend_calls'], 2)
        self.assertEqual(detector.backend_call_count, 4)
        self.assertEqual(len(backend.images), 4)

    def test_invalid_inputs_and_limits_fail_before_backend_calls(self):
        for image in (None, self.image.astype(float), self.image[:, :, :2]):
            detector, backend, _ = self.wrapper([])
            with self.assertRaises(ValueError):
                detector.detect(image)
            self.assertEqual(len(backend.images), 0)
            self.assertEqual(detector.last_receipt['backend_calls'], 0)
        for limit in (0., -.1, .651, float('nan'), float('inf'), True, '0.65'):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                StudioPersonRetryDetector(ScriptedDetector([]), max_detection_elapsed_s=limit)

    def test_invalid_local_clock_never_starts_a_retry(self):
        for value in (99., float('nan'), float('inf'), True):
            readings = iter((100., value))
            backend = ScriptedDetector([], [PERSON])
            detector = StudioPersonRetryDetector(backend, clock=lambda: next(readings))
            with self.subTest(value=value), self.assertRaises(ValueError):
                detector.detect(self.image)
            self.assertEqual(len(backend.images), 1)

    def run_pipeline(self, *, delays, capture_age=0.):
        detector, backend, clock = self.wrapper([], [PERSON], delays=delays)
        pipeline = MantisPerceptionPipeline(detector, max_frame_age_s=.65, clock=clock)
        sample = CameraSample(self.image, 1., 100., 0, 'studio', 'simulation',
                              'optical', capture_age_at_receive_s=capture_age)
        return pipeline.process(sample), detector, backend

    def test_fresh_retry_is_associated_only_after_actual_pipeline_validation(self):
        result, detector, backend = self.run_pipeline(delays=[.12, .13])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['detections'][0]['bbox_xyxy'], [4.25, 2., 10.75, 9.])
        self.assertEqual(result['detections'][0]['class_id'], 0)
        self.assertEqual(len(backend.images), 2)
        self.assertEqual(detector.last_receipt['backend_calls'], 2)

    def test_total_retry_duration_is_rejected_by_existing_pipeline_deadline(self):
        result, detector, backend = self.run_pipeline(delays=[.35, .35])
        self.assertEqual(result['reason'], 'inference_deadline_missed')
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(result['detections'], [])
        self.assertFalse(result['control_authority'])
        self.assertEqual(len(backend.images), 2)
        self.assertTrue(detector.last_receipt['used_retry'])
        self.assertAlmostEqual(result['age_at_finish_s'], .7)

    def test_capture_age_is_not_reset_when_retry_itself_finishes_within_budget(self):
        result, _, backend = self.run_pipeline(delays=[.2, .2], capture_age=.3)
        self.assertEqual(result['reason'], 'inference_deadline_missed')
        self.assertEqual(result['detections'], [])
        self.assertFalse(result['control_authority'])
        self.assertEqual(len(backend.images), 2)
        self.assertAlmostEqual(result['age_at_finish_s'], .7)

    def test_already_stale_frame_does_not_call_either_backend_pass(self):
        result, detector, backend = self.run_pipeline(delays=[0., 0.], capture_age=.7)
        self.assertEqual(result['reason'], 'stale_camera_frame')
        self.assertEqual(len(backend.images), 0)
        self.assertIsNone(detector.last_receipt)


if __name__ == '__main__':
    unittest.main()
