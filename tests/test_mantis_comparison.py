"""Matched estimator contracts using synthetic neural stubs, never model accuracy."""
from dataclasses import fields, replace
import copy
import json
import math
import unittest

import numpy as np

from experiments.flight_contracts import R_BODY_CAMERA, rotation_from_euler
from experiments.mantis_comparison import (AlphaBeta, BearingInput, CONFIG, METHODS,
    direct_bearing, run_comparison, score_comparison, transformed_observation, wrap_angle)


def sensor(sequence=0, *, box=(150., 130., 220., 260.), origin=0., rotation=R_BODY_CAMERA):
    return BearingInput(sequence, origin + sequence*.1, (391, 391), box,
                        (280., 280., 195., 195.), rotation, detector_wall_s=.02)


class StubBrain:
    """Derive deterministic test features from masks; explicitly not Flyvis."""
    def __init__(self):
        self.reset_calls = 0
        self.calls = []

    def reset(self):
        self.reset_calls += 1
        self.sequence = 0
        self.calls = []

    def step(self, mask, hold_s):
        ys, xs = np.nonzero(mask < .2)
        features = np.zeros(8)
        valid = len(xs) > 0
        if valid:
            features[:3] = [(xs.mean()+.5)*2/391-1, (ys.mean()+.5)*2/391-1,
                            (ys.max()-ys.min()+1)/391]
            features[5], features[7] = 1., 1.
        start = self.sequence * .1
        self.sequence += 1
        self.calls.append(dict(valid=valid, hold_s=hold_s, shape=mask.shape))
        return dict(features=features, valid=valid, stimulus_time_s=start,
                    response_time_s=self.sequence*.1)


class StubReadout:
    def predict(self, features):
        return features[:3].copy()

    def fit(self, *_args, **_kwargs):
        raise AssertionError('The comparison must never fit on evaluation inputs')


def run(rows=None, **kwargs):
    return run_comparison(rows or [sensor(i) for i in range(6)], StubBrain(), StubReadout(), **kwargs)


def truths(rows):
    return [dict(sequence=row['sequence'], capture_time_s=row['capture_time_s'],
                 heading_world_rad=.04) for row in rows]


class SensorBoundaryTests(unittest.TestCase):
    def test_sensor_interface_has_no_ground_truth_and_rejects_extra_fields(self):
        self.assertEqual({field.name for field in fields(BearingInput)}, {
            'sequence', 'capture_time_s', 'image_hw', 'bbox_xyxy', 'intrinsics',
            'rotation_world_camera', 'detector_wall_s'})
        with self.assertRaises(TypeError):
            BearingInput(**vars(sensor()), heading_world_rad=0.)
        with self.assertRaises(TypeError):
            run_comparison([sensor()], StubBrain(), StubReadout(), truth_rows=[])
        with self.assertRaises(TypeError):
            run_comparison([vars(sensor())], StubBrain(), StubReadout())

    def test_mutable_input_is_snapshotted(self):
        box = [150., 130., 220., 260.]
        rotation = R_BODY_CAMERA.copy()
        record = sensor(box=box, rotation=rotation)
        expected = direct_bearing(record)
        box[:] = [0, 0, 1, 1]
        rotation[:] = 0
        self.assertEqual(direct_bearing(record), expected)
        self.assertIsInstance(record.rotation_world_camera, tuple)

    def test_camera_axes_and_world_rotation_determine_bearing(self):
        centered = sensor(box=(160., 130., 230., 260.))
        self.assertAlmostEqual(direct_bearing(centered), 0.)
        right = sensor(box=(180., 130., 250., 260.))
        self.assertLess(direct_bearing(right), 0.)
        yaw = .7
        rotated = replace(right, rotation_world_camera=rotation_from_euler(0, 0, yaw) @ R_BODY_CAMERA)
        self.assertAlmostEqual(wrap_angle(direct_bearing(rotated)-direct_bearing(right)), yaw)

    def test_missing_and_outside_training_envelope_suppress_direct_bearing(self):
        self.assertIsNone(direct_bearing(sensor(box=None)))
        self.assertIsNone(direct_bearing(sensor(box=(310., 130., 380., 260.))))
        self.assertIsNone(direct_bearing(sensor(box=(150., 170., 220., 220.))))

    def test_invalid_sensor_calibration_geometry_and_nonfinite_values(self):
        changes = [dict(sequence=True), dict(capture_time_s=float('nan')),
                   dict(detector_wall_s=-1), dict(bbox_xyxy=(-1, 130, 220, 260)),
                   dict(intrinsics=(0, 280, 195, 195)), dict(image_hw=(391., 391)),
                   dict(rotation_world_camera=np.zeros((3, 3)))]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(sensor(), **change)


class AlphaBetaTests(unittest.TestCase):
    def test_no_measurement_has_no_output_or_implicit_initialization(self):
        filter_ = AlphaBeta()
        self.assertIsNone(filter_.update(0., None))
        self.assertIsNone(filter_.update(.1, None))
        self.assertIsNone(filter_.angle)
        self.assertAlmostEqual(filter_.update(.2, .4), .4)
        self.assertIsNone(filter_.update(.3, None))

    def test_wrap_boundary_uses_short_angular_residual(self):
        filter_ = AlphaBeta()
        filter_.step(0., math.pi-.01)
        result = filter_.step(.1, -math.pi+.01)
        self.assertLess(abs(wrap_angle(result-math.pi)), .02)
        self.assertLess(abs(filter_.rate), .1)

    def test_long_gap_resets_rate_and_short_gap_keeps_causal_state(self):
        filter_ = AlphaBeta()
        filter_.step(0., 0.)
        filter_.step(.1, .1)
        self.assertGreater(filter_.rate, 0.)
        self.assertIsNone(filter_.step(.2, None))
        self.assertNotAlmostEqual(filter_.step(.3, .3), .3)
        self.assertIsNone(filter_.step(.9, None))
        self.assertAlmostEqual(filter_.step(1., -.5), -.5)
        self.assertEqual(filter_.rate, 0.)

    def test_filter_rejects_reordered_duplicate_and_nonfinite_input(self):
        filter_ = AlphaBeta()
        filter_.step(.2, 0.)
        for timestamp, angle in [(.1, 0.), (.2, None), (float('nan'), 0.), (.3, float('inf'))]:
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                filter_.step(timestamp, angle)
        self.assertEqual(filter_.last_step_s, .2)


class MatchedRunnerTests(unittest.TestCase):
    def test_clean_uses_same_geometry_and_returns_json_serializable_receipts(self):
        result = run()
        json.dumps(result, allow_nan=False)
        self.assertEqual(result['config']['version'], CONFIG['version'])
        self.assertIn('StubBrain', result['neural_backend'])
        for row in result['rows']:
            self.assertTrue(row['shared_eligible'])
            self.assertEqual(row['bbox_xyxy'], row['original_bbox_xyxy'])
            self.assertTrue(all(method['valid'] for method in row['methods'].values()))
            self.assertEqual(len(row['neural']['features']), 8)
            self.assertEqual(row['neural']['hold_s'], .1)
            self.assertAlmostEqual(row['neural']['response_time_s']-row['neural']['stimulus_time_s'], .1)

    def test_noise_is_seeded_identical_across_methods_and_prefix_invariant(self):
        first = run(variant='noise', seed=27101)
        again = run(variant='noise', seed=27101)
        different = run(variant='noise', seed=27102)
        self.assertEqual([r['bbox_xyxy'] for r in first['rows']], [r['bbox_xyxy'] for r in again['rows']])
        self.assertNotEqual(first['rows'][0]['bbox_xyxy'], different['rows'][0]['bbox_xyxy'])
        for row in first['rows']:
            noisy = replace(sensor(row['sequence']), bbox_xyxy=row['bbox_xyxy'])
            self.assertEqual(row['methods']['direct_yolo']['heading_world_rad'], direct_bearing(noisy))

    def test_adding_future_inputs_does_not_change_any_prior_estimate(self):
        prefix = run([sensor(i) for i in range(3)], variant='noise')
        longer = run([sensor(i, box=(160., 130., 230., 260.) if i > 2 else (150., 130., 220., 260.))
                      for i in range(8)], variant='noise')
        for before, after in zip(prefix['rows'], longer['rows']):
            self.assertEqual(before['bbox_xyxy'], after['bbox_xyxy'])
            self.assertEqual(before['neural'], after['neural'])
            for name in METHODS:
                self.assertEqual(before['methods'][name]['heading_world_rad'], after['methods'][name]['heading_world_rad'])

    def test_missing_gap_and_unsupported_inputs_suppress_every_method(self):
        records = [sensor(i) for i in range(17)]
        records[2] = sensor(2, box=None)
        records[3] = sensor(3, box=(310., 130., 380., 260.))
        brain = StubBrain()
        result = run_comparison(records, brain, StubReadout(), variant='gaps')
        for index in (2, 3, 12, 13, 14, 15):
            row = result['rows'][index]
            self.assertFalse(row['shared_eligible'])
            self.assertTrue(all(not m['valid'] and m['heading_world_rad'] is None for m in row['methods'].values()))
        self.assertFalse(brain.calls[12]['valid'])
        self.assertEqual(len(brain.calls), 17)  # Blank input still advances neural recurrence.
        self.assertTrue(result['rows'][16]['methods']['direct_yolo']['valid'])

    def test_noisy_offimage_box_is_unavailable_instead_of_clipped(self):
        record = sensor(box=(0., 130., 391., 260.))
        observation, shared = transformed_observation(record, 'noise', 27101)
        self.assertIsNone(observation['bbox_xyxy'])
        self.assertFalse(shared['shared_eligible'])
        self.assertEqual(shared['shared_reason'], 'perturbed_box_outside_image')

    def test_future_reordered_or_duplicate_stream_rejected_before_neural_execution(self):
        for records in ([sensor(1), sensor(0)], [sensor(0), replace(sensor(1), capture_time_s=0.)],
                        [sensor(0), sensor(0)]):
            brain = StubBrain()
            with self.subTest(records=records), self.assertRaises(ValueError):
                run_comparison(records, brain, StubReadout())
            self.assertEqual(brain.reset_calls, 0)

    def test_variable_recorded_clock_is_distinct_from_fixed_neural_hold(self):
        times = [2., 2.15, 2.37, 3.]
        records = [replace(sensor(i, box=(150.+i*4, 130., 220.+i*4, 260.)), capture_time_s=t)
                   for i, t in enumerate(times)]
        result = run(records)
        self.assertIsNone(result['config']['recording_interval_s'])
        self.assertEqual(result['config']['recording_clock'], 'recorded_capture_timestamps')
        self.assertEqual([row['capture_time_s'] for row in result['rows']], times)
        filter_ = AlphaBeta()
        for i, row in enumerate(result['rows']):
            self.assertAlmostEqual(row['neural']['stimulus_time_s'], i*.1)
            self.assertAlmostEqual(row['neural']['response_time_s'], (i+1)*.1)
            expected = filter_.step(times[i], direct_bearing(records[i]))
            self.assertAlmostEqual(row['methods']['alpha_beta']['heading_world_rad'], expected)
        self.assertEqual(filter_.rate, 0.)  # Actual .63-second capture gap resets.

    def test_invalid_seed_or_variant_rejected_before_neural_execution(self):
        for kwargs in (dict(seed=-1), dict(seed=True), dict(variant='tuned_on_truth')):
            brain = StubBrain()
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                run_comparison([sensor()], brain, StubReadout(), **kwargs)
            self.assertEqual(brain.reset_calls, 0)

    def test_common_release_includes_each_method_and_shared_detector_cost(self):
        result = run([replace(sensor(i, origin=10.), detector_wall_s=.3) for i in range(2)])
        for row in result['rows']:
            delay = max(.1, row['detector_wall_s'] + max(m['wall_s'] for m in row['methods'].values()))
            self.assertAlmostEqual(row['common_available_time_s'], row['capture_time_s']+delay)
            for method in row['methods'].values():
                self.assertLessEqual(method['standalone_available_time_s'], row['common_available_time_s'])
                self.assertGreaterEqual(method['wall_s'], 0.)
        self.assertEqual(result['rows'][0]['neural']['stimulus_time_s'], 0.)
        self.assertFalse(result['timing']['achieved_real_time'])

    def test_incorrect_neural_response_clock_is_not_accepted(self):
        class FutureBrain(StubBrain):
            def step(self, mask, hold_s):
                response = super().step(mask, hold_s)
                response['response_time_s'] += .1
                return response
        with self.assertRaisesRegex(ValueError, 'clock'):
            run_comparison([sensor()], FutureBrain(), StubReadout())


class EvaluatorTests(unittest.TestCase):
    def test_truth_changes_scores_only_never_mutates_predictions(self):
        rows = run()['rows']
        original = copy.deepcopy(rows)
        first = score_comparison(rows, truths(rows))
        changed = truths(rows)
        for row in changed: row['heading_world_rad'] += .5
        second = score_comparison(rows, changed)
        self.assertEqual(rows, original)
        self.assertNotEqual(first['methods'], second['methods'])
        self.assertEqual(first['matched_count'], len(rows))
        self.assertFalse(first['automatic_winner_or_superiority_test'])

    def test_rmse_uses_same_matched_subset_and_discloses_neural_abstention(self):
        rows = run()['rows']
        rows[0]['methods']['mantis_neural'].update(valid=False, heading_world_rad=None)
        rows[0]['methods']['direct_yolo']['heading_world_rad'] = 2.
        score = score_comparison(rows, truths(rows))
        self.assertEqual(score['matched_count'], len(rows)-1)
        self.assertEqual(score['unmatched_truth_count'], 1)
        self.assertLess(score['methods']['direct_yolo']['matched_capture_rmse_rad'], .1)
        self.assertLess(score['methods']['mantis_neural']['full_truth_coverage'], 1.)
        self.assertEqual(score['methods']['direct_yolo']['full_truth_coverage'], 1.)
        self.assertFalse(score['neural_coverage_at_least_baselines'])

    def test_missing_inputs_and_unavailable_truth_are_counted_explicitly(self):
        rows = run([sensor(i, box=None if i == 1 else (150., 130., 220., 260.)) for i in range(6)])['rows']
        ground = truths(rows)
        ground[2]['heading_world_rad'] = None
        score = score_comparison(rows, ground)
        self.assertEqual(score['recorded_count'], 6)
        self.assertEqual(score['truth_available_count'], 5)
        self.assertEqual(score['truth_unavailable_count'], 1)
        self.assertEqual(score['matched_count'], 4)
        self.assertEqual(score['shared_missing_reasons']['target_not_observed'], 1)
        self.assertEqual(score['methods']['direct_yolo']['full_truth_coverage'], .8)
        self.assertEqual(score['methods']['direct_yolo']['eligible_truth_coverage'], 1.)

    def test_wrapped_errors_are_small_across_pi_boundary(self):
        rows = run([sensor()])['rows']
        for method in rows[0]['methods'].values(): method['heading_world_rad'] = -math.pi+.01
        score = score_comparison(rows, [dict(sequence=0, heading_world_rad=math.pi-.01)])
        self.assertAlmostEqual(score['methods']['direct_yolo']['matched_capture_rmse_rad'], .02)

    def test_missing_duplicate_or_misaligned_truth_cannot_score(self):
        rows = run()['rows']
        for ground in (truths(rows)[:-1], truths(rows)[:-1]+[truths(rows)[0]],
                       [dict(row, capture_time_s=row['capture_time_s']+1) for row in truths(rows)]):
            with self.subTest(ground=ground), self.assertRaises(ValueError):
                score_comparison(rows, ground)

    def test_future_release_or_output_during_missing_input_cannot_score(self):
        rows = run()['rows']
        corrupt = copy.deepcopy(rows)
        corrupt[0]['common_available_time_s'] = -.1
        with self.assertRaises(ValueError): score_comparison(corrupt, truths(rows))
        corrupt = copy.deepcopy(rows)
        corrupt[0]['shared_eligible'] = False
        with self.assertRaises(ValueError): score_comparison(corrupt, truths(rows))

    def test_no_matched_evidence_has_no_numeric_accuracy_or_winner(self):
        rows = run([sensor(i, box=None) for i in range(2)])['rows']
        score = score_comparison(rows, truths(rows))
        self.assertEqual(score['matched_count'], 0)
        self.assertIsNone(score['neural_improvement_over_stronger_baseline_fraction'])
        self.assertTrue(all(m['matched_capture_rmse_rad'] is None for m in score['methods'].values()))
        json.dumps(score, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
