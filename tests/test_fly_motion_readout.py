"""Verify broader neural readout fitting and target-local tracking boundaries."""
import copy
import json
import unittest

import numpy as np

from experiments.fly_motion_readout import (
    LearnedFlowTracker, RidgeReadout, flow_features,
)


def grid(rows=(100., 110., 120.), cols=(100., 110., 120.)):
    return np.asarray([(row, col) for row in rows for col in cols])


def identity_readout():
    X = np.asarray([[-2., -1.], [2., -1.], [-2., 1.], [2., 1.]])
    return RidgeReadout.fit(X, X, ridge=0.)


class FeatureTests(unittest.TestCase):
    def test_statistics_order_and_quadrant_coordinates(self):
        centers = grid(rows=(100, 110, 120), cols=(100, 110, 120))
        field = np.vstack((centers[:, 1], centers[:, 0] * 2))
        box = [95, 95, 125, 125]
        result = flow_features(field, centers, box, "statistics")
        expected = [110, 220, 110, 220, np.std([100, 110, 120]),
                    2 * np.std([100, 110, 120]), 100, 200, 115, 200, 100, 230, 115, 230]
        np.testing.assert_allclose(result, expected)
        np.testing.assert_array_equal(flow_features(field, centers, box), [110, 220])

    def test_empty_quadrants_use_overall_mean(self):
        centers = grid()
        field = np.vstack((np.arange(9), np.arange(9) * 2))
        # All nine sites lie in the top-left quadrant of this box.
        result = flow_features(field, centers, [90, 90, 200, 200], "statistics")
        np.testing.assert_array_equal(result[6:], np.tile([4., 8.], 4))

    def test_supported_roi_does_not_borrow_outside_or_partial_windows(self):
        centers = np.vstack((grid(), [[0, 0], [100, 385], [100, 200]]))
        field = np.ones((2, len(centers)))
        field[:, -3:] = 9999
        np.testing.assert_array_equal(flow_features(field, centers, [90, 90, 140, 140]), [1, 1])
        with self.assertRaises(ValueError):
            flow_features(field, centers, [100, 100, 120, 121])

    def test_invalid_inputs_and_modes(self):
        field, centers, box = np.zeros((2, 9)), grid(), [90, 90, 140, 140]
        for mode in (None, [], "quadrants"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                flow_features(field, centers, box, mode)
        for minimum in (1, 8, True, 9.):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                flow_features(field, centers, box, min_support=minimum)
        for bad in (np.zeros((9, 2)), np.full((2, 9), np.nan), np.ones((2, 9), dtype=bool)):
            with self.subTest(field=bad), self.assertRaises(ValueError):
                flow_features(bad, centers, box)
        with self.assertRaises(ValueError):
            flow_features(field, [[0, 1, 2]], box)
        with self.assertRaises(ValueError):
            flow_features(field, centers, [0, 0, 0, 2])


class RidgeTests(unittest.TestCase):
    def test_fit_recovers_offsets_including_stationary_object(self):
        velocities = np.asarray([[-20, 0], [20, 0], [0, -20], [0, 20], [0, 0.]])
        offset = np.asarray([1.4, -2.2])
        features = velocities / [10., -5.] + offset
        model = RidgeReadout.fit(features, velocities, ridge=0.)
        np.testing.assert_allclose(model.predict(offset), [0, 0], atol=1e-12)
        new_velocities = np.asarray([[7., 3.], [-5., 8.]])
        np.testing.assert_allclose(model.predict(new_velocities / [10., -5.] + offset),
                                   new_velocities, atol=1e-12)

    def test_weighted_standardization_and_ridge_match_explicit_solution(self):
        X = np.asarray([[0., 3.], [1., -2.], [2., 0.], [4., 1.]])
        y = np.asarray([[1., 0.], [3., 4.], [5., -2.], [0., 1.]])
        weights = np.asarray([1., 2., 5., 2.])
        penalty = .3
        model = RidgeReadout.fit(X, y, weights, penalty)
        w = weights / weights.sum()
        mean = w @ X
        scale = np.sqrt(w @ (X - mean)**2)
        z = (X - mean) / scale
        bias = w @ y
        slopes = np.linalg.solve(z.T @ (w[:, None] * z) + penalty * np.eye(2),
                                 z.T @ (w[:, None] * (y - bias)))
        np.testing.assert_allclose(model.mean, mean)
        np.testing.assert_allclose(model.scale, scale)
        np.testing.assert_allclose(model.coefficients, slopes)
        np.testing.assert_allclose(model.predict(X), z @ slopes + bias)

    def test_weight_scale_and_sequence_duplication_do_not_change_mapping(self):
        X = np.asarray([[0., 1.], [2., 1.], [1., 3.]])
        y = np.asarray([[0., 2.], [1., 3.], [5., 7.]])
        weight = np.asarray([1., 2., 3.])
        first = RidgeReadout.fit(X, y, weight, .1)
        second = RidgeReadout.fit(X, y, weight * 1e100, .1)
        repeated = RidgeReadout.fit(np.repeat(X, 3, axis=0), np.repeat(y, 3, axis=0),
                                    np.repeat(weight / 3, 3), .1)
        np.testing.assert_allclose(first.predict(X), second.predict(X), atol=1e-14)
        np.testing.assert_allclose(first.predict(X), repeated.predict(X), atol=1e-14)

    def test_constant_feature_has_zero_slope_and_unpenalized_intercept(self):
        X = np.repeat([[5., -8.]], 4, axis=0)
        y = np.repeat([[12., -7.]], 4, axis=0)
        model = RidgeReadout.fit(X, y, ridge=100000)
        np.testing.assert_array_equal(model.coefficients, np.zeros((2, 2)))
        np.testing.assert_array_equal(model.scale, [1., 1.])
        np.testing.assert_array_equal(model.predict([999., 999.]), [12., -7.])

    def test_zero_weight_rows_do_not_change_fit_or_training_domain(self):
        X = np.asarray([[0., 0.], [2., 3.], [-2., -3.], [1e300, -1e300]])
        y = X.copy()
        model = RidgeReadout.fit(X, y, [1, 1, 1, 0], .1)
        comparison = RidgeReadout.fit(X[:3], y[:3], ridge=.1)
        np.testing.assert_allclose(model.predict(X[:3]), comparison.predict(X[:3]))
        np.testing.assert_array_equal(model.training_min, [-2, -3])
        np.testing.assert_array_equal(model.training_max, [2, 3])
        self.assertEqual(model.diagnostics["positive_weight_samples"], 3)

    def test_extreme_training_scale_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "standardization"):
            RidgeReadout.fit([[-1e300, 0], [1e300, 0]], [[0, 0], [1, 0]])

    def test_prediction_does_not_refit_or_change_training_domain(self):
        model = identity_readout()
        before = model.to_dict()
        np.testing.assert_allclose(model.predict([100., -50.]), [100., -50.])
        self.assertEqual(model.to_dict(), before)
        domain = model.feature_domain([100., -50.])
        self.assertEqual(domain["outside_training_range_count"], 2)
        self.assertEqual(domain["feature_count"], 2)
        self.assertEqual(domain["max_abs_training_zscore"], 50.)

    def test_json_roundtrip_copies_and_validates_schema(self):
        model = identity_readout()
        payload = json.loads(json.dumps(model.to_dict(), allow_nan=False))
        restored = RidgeReadout.from_dict(payload)
        np.testing.assert_array_equal(restored.predict([[1, 2], [3, 4]]), model.predict([[1, 2], [3, 4]]))
        payload["mean"][0] = 999
        self.assertNotEqual(restored.mean[0], 999)
        for key, value in (("feature_count", 14), ("schema_version", True),
                           ("scale", [0, 1]), ("coefficients", [[1, 2]]),
                           ("ridge", -1), ("intercept", [np.inf, 1])):
            bad = copy.deepcopy(model.to_dict())
            bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                RidgeReadout.from_dict(bad)
        bad = model.to_dict()
        bad["new_field"] = 0
        with self.assertRaises(ValueError):
            RidgeReadout.from_dict(bad)

    def test_invalid_fit_and_prediction_inputs(self):
        X, y = np.eye(2), np.eye(2)
        for bad_x, bad_y in (([], []), ([[1, 2]], [[0]]), (X, [[0, 0]]),
                             (np.full((2, 2), np.nan), y), (X.astype(bool), y)):
            with self.subTest(X=bad_x, y=bad_y), self.assertRaises(ValueError):
                RidgeReadout.fit(bad_x, bad_y)
        for weights in ([0, 0], [-1, 1], [1], [[1, 1]], [np.inf, 1], [True, True]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                RidgeReadout.fit(X, y, weights)
        for ridge in (-1, True, np.nan, np.inf, [1]):
            with self.subTest(ridge=ridge), self.assertRaises(ValueError):
                RidgeReadout.fit(X, y, ridge=ridge)
        model = identity_readout()
        for input_x in ([1], [[[1, 2]]], [np.nan, 2], [True, True]):
            with self.subTest(X=input_x), self.assertRaises(ValueError):
                model.predict(input_x)


class TrackerTests(unittest.TestCase):
    def test_uses_actual_dt_and_previous_predicted_roi(self):
        centers = grid(cols=(100, 110, 120, 130, 140, 150))
        tracker = LearnedFlowTracker(centers, [99, 99, 121, 121], identity_readout())
        first = tracker.step(np.repeat([[30.], [0.]], len(centers), axis=1), 1.)
        second_flow = np.zeros((2, len(centers)))
        second_flow[0, centers[:, 1] >= 130] = -5
        second = tracker.step(second_flow, .5)
        np.testing.assert_allclose(first["box_xyxy"], [129, 99, 151, 121])
        np.testing.assert_allclose(second["box_xyxy"], [126.5, 99, 148.5, 121])
        self.assertNotEqual(first["selected_indices"], second["selected_indices"])
        self.assertEqual(second["status"], "tracking")
        self.assertEqual(second["feature_count"], 2)

    def test_no_startup_stationarity_assumption(self):
        tracker = LearnedFlowTracker(grid(), [90, 90, 140, 140], identity_readout())
        first = tracker.step(np.repeat([[4.], [-2.]], 9, axis=1), .25)
        np.testing.assert_allclose(first["box_xyxy"], [91, 89.5, 141, 139.5])

    def test_loss_on_excess_speed_is_permanent_without_clipping(self):
        tracker = LearnedFlowTracker(grid(), [90, 90, 140, 140], identity_readout())
        first = tracker.step(np.repeat([[100.], [100.]], 9, axis=1), .01)
        self.assertEqual(first["status"], "uncertain_speed")
        np.testing.assert_array_equal(first["box_xyxy"], [90, 90, 140, 140])
        np.testing.assert_allclose(first["velocity_xy"], [100, 100])
        later = tracker.step(np.zeros((2, 9)), .01)
        self.assertEqual(later["status"], "uncertain_speed")
        self.assertIsNone(later["velocity_xy"])
        self.assertEqual(later["selected_indices"], [])

    def test_missing_support_and_outside_frame_remain_lost(self):
        tracker = LearnedFlowTracker(grid(), [99, 99, 121, 121], identity_readout())
        tracker.step(np.repeat([[40.], [0.]], 9, axis=1), 1.)
        lost = tracker.step(np.zeros((2, 9)), .1)
        self.assertEqual(lost["status"], "insufficient_support")
        self.assertEqual(tracker.step(np.ones((2, 9)), .1), lost)
        tracker = LearnedFlowTracker(grid(), [90, 90, 140, 140], identity_readout())
        lost = tracker.step(np.repeat([[100.], [0.]], 9, axis=1), 3.)
        self.assertEqual(lost["status"], "outside_frame")
        np.testing.assert_array_equal(lost["box_xyxy"], [90, 90, 140, 140])

    def test_statistics_tracker_feature_contract(self):
        centers = grid()
        features = np.vstack([flow_features(np.repeat([[x], [0]], 9, axis=1), centers,
                                           [90, 90, 140, 140], "statistics") for x in (-2, 0, 2)])
        model = RidgeReadout.fit(features, [[-2, 0], [0, 0], [2, 0]], ridge=0)
        tracker = LearnedFlowTracker(centers, [90, 90, 140, 140], model, mode="statistics")
        result = tracker.step(np.ones((2, 9)), .1)
        self.assertEqual(len(result["features"]), 14)
        self.assertEqual(result["feature_domain"]["feature_count"], 14)

    def test_invalid_inputs_do_not_mutate_tracker(self):
        tracker = LearnedFlowTracker(grid(), [90, 90, 140, 140], identity_readout())
        for elapsed in (0, -1, True, np.inf, np.nan, [1]):
            with self.subTest(dt=elapsed), self.assertRaises(ValueError):
                tracker.step(np.zeros((2, 9)), elapsed)
        with self.assertRaises(ValueError):
            tracker.step(np.full((2, 9), np.nan), .1)
        np.testing.assert_array_equal(tracker.box, [90, 90, 140, 140])
        self.assertEqual(tracker.status, "tracking")

    def test_constructor_copies_inputs_and_validates_contract(self):
        centers, box = grid(), np.asarray([90, 90, 140, 140])
        tracker = LearnedFlowTracker(centers, box, identity_readout())
        centers[:] = 0
        box[:] = 0
        result = tracker.step(np.ones((2, 9)), .1)
        np.testing.assert_allclose(result["box_xyxy"], [90.1, 90.1, 140.1, 140.1])
        for kwargs in ({"mode": "statistics"}, {"max_speed": 0}, {"max_speed": True},
                       {"min_support": 8}, {"min_support": 9.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                LearnedFlowTracker(grid(), [90, 90, 140, 140], identity_readout(), **kwargs)
        with self.assertRaises(ValueError):
            LearnedFlowTracker(grid(), [-1, 90, 140, 140], identity_readout())


if __name__ == "__main__":
    unittest.main()
