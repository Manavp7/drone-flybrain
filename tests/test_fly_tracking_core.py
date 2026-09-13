"""Check propagation mathematics, evidence boundaries and causal visibility."""
import unittest

import numpy as np

from experiments.fly_tracking_core import (
    FlowBoxTracker, evaluate_boxes, evaluate_tracking, fit_affine_flow, pool_flow,
    raw_flow_mapping, supported_indices, visible_response_indices,
)


def grid(rows=(100., 110., 120.), cols=(100., 110., 120.)):
    return np.asarray([(row, col) for row in rows for col in cols])


IDENTITY = np.asarray([[1., 0.], [0., 1.], [0., 0.]])


class SupportTests(unittest.TestCase):
    def test_row_col_mapping_and_half_open_box(self):
        centers = np.asarray([[20., 50.], [50., 20.], [20., 60.], [30., 50.]])
        np.testing.assert_array_equal(supported_indices(centers, [50, 20, 60, 30], min_support=1), [0])

    def test_kernel_boundary_excludes_partial_windows(self):
        centers = np.asarray([[6, 6], [384, 384], [5, 100], [100, 385], [0, 0], [390, 390]])
        np.testing.assert_array_equal(supported_indices(centers, [0, 0, 391, 391], min_support=1), [0, 1])

    def test_insufficient_support_does_not_borrow_nearby_points(self):
        centers = grid()
        self.assertEqual(supported_indices(centers, [100, 100, 120, 121]).size, 0)
        with self.assertRaises(ValueError):
            pool_flow(np.zeros((2, len(centers))), centers, [100, 100, 120, 121])

    def test_pooling_resists_one_large_outlier(self):
        flow = np.repeat([[2.], [-3.]], 9, axis=1)
        flow[:, 0] = [1000, -1000]
        pooled, selected = pool_flow(flow, grid(), [99, 99, 121, 121])
        np.testing.assert_array_equal(pooled, [2., -3.])
        self.assertEqual(len(selected), 9)

    def test_bad_geometry_and_flow_are_rejected(self):
        for centers, box in (([[np.nan, 5]], [0, 0, 20, 20]), ([[1, 2, 3]], [0, 0, 20, 20]),
                             (grid(), [10, 0, 10, 20]), (grid(), [0, 0, 20])):
            with self.subTest(centers=centers, box=box), self.assertRaises(ValueError):
                supported_indices(centers, box)
        with self.assertRaises(ValueError):
            pool_flow(np.zeros((9, 2)), grid(), [90, 90, 130, 130])
        with self.assertRaises(ValueError):
            supported_indices(grid(), [90, 90, 130, 130], min_support=True)


class TrackerTests(unittest.TestCase):
    def test_training_mapping_sign_scale_and_zero_intercept(self):
        mapping = raw_flow_mapping()
        # A 1px-right, 2px-down Sintel displacement summed over 169 pixels.
        normalized = np.asarray([169 / 436, -2 * 169 / 436, 1.])
        np.testing.assert_allclose(normalized @ mapping / 24, [1., 2.])
        np.testing.assert_array_equal(mapping[2], [0., 0.])

    def test_actual_elapsed_time_and_fixed_box_size(self):
        tracker = FlowBoxTracker(grid(), [90, 90, 140, 140], IDENTITY)
        flow = np.repeat([[4.], [-2.]], 9, axis=1)
        first = tracker.step(flow, .25)
        np.testing.assert_allclose(first["box_xyxy"], [91, 89.5, 141, 139.5])
        second = tracker.step(flow, .75)
        np.testing.assert_allclose(second["box_xyxy"], [94, 88, 144, 138])
        self.assertEqual(second["status"], "tracking")

    def test_support_is_selected_from_predicted_roi(self):
        centers = grid(cols=(100., 110., 120., 130., 140., 150.))
        tracker = FlowBoxTracker(centers, [99, 99, 121, 121], IDENTITY)
        first = tracker.step(np.repeat([[30.], [0.]], len(centers), axis=1), 1.)
        second_flow = np.zeros((2, len(centers)))
        second_flow[0, centers[:, 1] >= 130] = -5.
        second = tracker.step(second_flow, 1.)
        self.assertNotEqual(first["selected_indices"], second["selected_indices"])
        np.testing.assert_allclose(second["box_xyxy"], [124, 99, 146, 121])

    def test_insufficient_support_is_permanent(self):
        tracker = FlowBoxTracker(grid(), [99, 99, 121, 121], IDENTITY)
        tracker.step(np.repeat([[50.], [0.]], 9, axis=1), 1.)
        lost = tracker.step(np.zeros((2, 9)), 1.)
        self.assertEqual(lost["status"], "insufficient_support")
        later = tracker.step(np.repeat([[-50.], [0.]], 9, axis=1), 1.)
        self.assertEqual(later, lost)
        self.assertIsNone(later["pooled_flow"])

    def test_outside_candidate_freezes_last_accepted_box(self):
        tracker = FlowBoxTracker(grid(), [90, 90, 140, 140], IDENTITY)
        lost = tracker.step(np.repeat([[300.], [0.]], 9, axis=1), 1.)
        self.assertEqual(lost["status"], "outside_frame")
        self.assertEqual(lost["box_xyxy"], [90, 90, 140, 140])
        later = tracker.step(np.repeat([[-10.], [0.]], 9, axis=1), 1.)
        self.assertEqual(later["box_xyxy"], lost["box_xyxy"])
        self.assertEqual(later["status"], "outside_frame")
        self.assertEqual(later["selected_indices"], [])

    def test_initial_state_does_not_alias_callers(self):
        centers, box, mapping = grid(), np.array([90., 90., 140., 140.]), IDENTITY.copy()
        tracker = FlowBoxTracker(centers, box, mapping)
        centers[:] = 0
        box[:] = 0
        mapping[:] = 0
        result = tracker.step(np.ones((2, 9)), 1.)
        self.assertEqual(result["box_xyxy"], [91, 91, 141, 141])

    def test_invalid_time_and_nonfinite_values_raise_without_mutation(self):
        tracker = FlowBoxTracker(grid(), [90, 90, 140, 140], IDENTITY)
        for dt in (0, -1, np.nan, np.inf, True, [1]):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                tracker.step(np.zeros((2, 9)), dt)
        with self.assertRaises(ValueError):
            tracker.step(np.full((2, 9), np.nan), .1)
        self.assertEqual(tracker.status, "tracking")
        np.testing.assert_array_equal(tracker.box, [90, 90, 140, 140])

    def test_invalid_constructor_inputs_raise(self):
        for box, mapping in (([-1, 0, 20, 20], IDENTITY), ([0, 0, 400, 20], IDENTITY),
                             ([90, 90, 140, 140], np.eye(2)), ([90, 90, 140, 140], np.full((3, 2), np.inf))):
            with self.subTest(box=box, mapping=mapping), self.assertRaises(ValueError):
                FlowBoxTracker(grid(), box, mapping)


class CalibrationEvaluationTests(unittest.TestCase):
    def test_affine_readout_recovers_separate_velocity_relation(self):
        calibration = np.asarray([[-2, 0], [0, -2], [2, 0], [0, 2], [1, -1], [-1, 1.]])
        expected = np.asarray([[3., 2.], [-4., 5.], [7., -6.]])
        target = np.column_stack((calibration, np.ones(len(calibration)))) @ expected
        mapping, diagnostics = fit_affine_flow(calibration, target)
        np.testing.assert_allclose(mapping, expected, atol=1e-7, rtol=0)
        held_out = np.asarray([[.5, -.5, 1.], [-3., 4., 1.]])
        np.testing.assert_allclose(held_out @ mapping, held_out @ expected, atol=1e-7, rtol=0)
        self.assertEqual(diagnostics["design_rank"], 3)
        self.assertLess(diagnostics["calibration_rmse_px_per_s"], 1e-7)

    def test_degenerate_or_nonfinite_calibration_is_rejected(self):
        for pooled, velocities in ((np.zeros((5, 2)), np.zeros((5, 2))),
                                    (np.array([[0, 0], [1, 1], [2, 2]]), np.zeros((3, 2))),
                                    (np.ones((2, 2)), np.ones((2, 2))),
                                    (np.full((3, 2), np.nan), np.zeros((3, 2)))):
            with self.subTest(pooled=pooled), self.assertRaises(ValueError):
                fit_affine_flow(pooled, velocities)

    def test_box_metrics_known_geometry(self):
        truth = np.repeat([[0., 0., 10., 10.]], 3, axis=0)
        pred = np.asarray([[0, 0, 10, 10], [5, 0, 15, 10], [20, 0, 30, 10.]])
        metrics = evaluate_boxes(pred, truth)
        self.assertAlmostEqual(metrics["mean_center_error"], 25 / 3)
        self.assertEqual(metrics["final_center_error"], 20)
        self.assertAlmostEqual(metrics["mean_iou"], (1 + 1 / 3) / 3)
        self.assertAlmostEqual(metrics["fraction_iou_ge_0_5"], 1 / 3)

    def test_invalid_box_sequences_rejected(self):
        for pred, truth in (([], []), ([[0, 0, 0, 1]], [[0, 0, 1, 1]]),
                             ([[0, 0, 1, 1]], [[0, 0, 1, 1], [0, 0, 1, 1]])):
            with self.subTest(pred=pred), self.assertRaises(ValueError):
                evaluate_boxes(pred, truth)

    def test_lost_box_overlapping_returning_target_is_not_tracking_success(self):
        tracker = FlowBoxTracker(grid(), [90, 90, 140, 140], IDENTITY)
        states = [tracker.step(np.zeros((2, 9)), .1),
                  tracker.step(np.repeat([[300.], [0.]], 9, axis=1), 1.),
                  tracker.step(np.zeros((2, 9)), .1)]
        prediction = [state["box_xyxy"] for state in states]
        truth = [[90, 90, 140, 140], [200, 90, 250, 140], [90, 90, 140, 140]]
        active = [state["status"] == "tracking" for state in states]
        metrics = evaluate_tracking(prediction, truth, active)
        self.assertAlmostEqual(metrics["fraction_iou_ge_0_5"], 2 / 3)
        self.assertAlmostEqual(metrics["fraction_active_iou_ge_0_5"], 1 / 3)
        self.assertAlmostEqual(metrics["fraction_active"], 1 / 3)
        self.assertEqual(metrics["scored_frames"], 3)
        # All geometry fields remain available with their original semantics.
        for key, value in evaluate_boxes(prediction, truth).items():
            self.assertEqual(metrics[key], value)

    def test_all_lost_rows_score_zero_even_with_perfect_retained_boxes(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        lost = evaluate_tracking(boxes, boxes, [False, False])
        active = evaluate_tracking(boxes, boxes, [True, True])
        self.assertEqual(lost["fraction_iou_ge_0_5"], 1.)
        self.assertEqual(lost["fraction_active_iou_ge_0_5"], 0.)
        self.assertEqual(lost["fraction_active"], 0.)
        self.assertEqual(active["fraction_active_iou_ge_0_5"], 1.)

    def test_active_mask_requires_matching_one_dimensional_booleans(self):
        boxes = [[0, 0, 10, 10], [0, 0, 10, 10]]
        for mask in ([1, 0], [1., 0.], [[True, False]], [True],
                     np.asarray([True, False], dtype=object), [True, np.nan]):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                evaluate_tracking(boxes, boxes, mask)

    def test_visibility_never_uses_even_slightly_future_response(self):
        responses = np.asarray([.02, .04, .06])
        displays = np.asarray([0., .019999999999999, .02, .039, .04, .07])
        np.testing.assert_array_equal(visible_response_indices(responses, displays), [-1, -1, 0, 0, 1, 2])
        np.testing.assert_array_equal(visible_response_indices([], [0., 1.]), [-1, -1])

    def test_visibility_rejects_disordered_and_nonfinite_times(self):
        for response, display in (([.02, .02], [0]), ([.04, .02], [0]),
                                   ([.02], [1, 0]), ([np.nan], [0])):
            with self.subTest(response=response, display=display), self.assertRaises(ValueError):
                visible_response_indices(response, display)


if __name__ == "__main__":
    unittest.main()
