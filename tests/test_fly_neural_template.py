"""Evidence and causal-memory checks for the frozen-feature template tracker."""
import unittest

import numpy as np

from experiments.fly_neural_template import NeuralTemplateTracker


def centers():
    return np.asarray([(row, col) for row in range(6, 385, 6)
                       for col in range(6, 385, 6)], dtype=np.float64)


def feature_pattern(points, dx=0, dy=0):
    row, col = points.T
    x, y = (col - dx - 190) / 23, (row - dy - 180) / 27
    return np.asarray([np.exp(-((x + 0.5) ** 2 + (y - 0.3) ** 2))
                       + 0.35 * np.exp(-((x - 1.5) ** 2 + (y + 1.2) ** 2) / 0.2),
                       np.sin(x * 2.1 + y * 0.7) * np.exp(-(x ** 2 + y ** 2) / 4)])


class NeuralTemplateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = centers()
        cls.initial = feature_pattern(cls.points)
        cls.box = np.asarray([150., 140., 230., 224.])

    def tracker(self, **kwargs):
        return NeuralTemplateTracker(self.points, self.box, self.initial, **kwargs)

    def test_translation_uses_rc_mapping_and_preserves_size(self):
        tracker = self.tracker()
        result = tracker.step(feature_pattern(self.points, dx=12, dy=-8))
        self.assertEqual(result["status"], "tracking")
        np.testing.assert_allclose(result["box_xyxy"], self.box + [12, -8, 12, -8])
        self.assertGreater(result["score"], 0.99)
        result = tracker.step(feature_pattern(self.points, dx=20, dy=-4), 0.04)
        np.testing.assert_allclose(result["box_xyxy"], self.box + [20, -4, 20, -4])
        self.assertEqual(result["dt_s"], 0.04)

    def test_static_template_has_no_drift(self):
        tracker = self.tracker()
        for _ in range(5):
            result = tracker.step(self.initial)
            self.assertEqual(result["status"], "tracking")
            np.testing.assert_array_equal(result["displacement_xy"], [0, 0])
            np.testing.assert_array_equal(result["box_xyxy"], self.box)

    def test_zero_or_flat_initial_features_cannot_track(self):
        for features in (np.zeros_like(self.initial), np.ones_like(self.initial),
                         np.vstack((np.full(len(self.points), 3.), np.full(len(self.points), 7.)))):
            with self.subTest(value=float(features[0, 0])):
                tracker = NeuralTemplateTracker(self.points, self.box, features)
                result = tracker.step(self.initial)
                self.assertEqual(result["status"], "insufficient_evidence")
                self.assertEqual(result["reason"], "unusable_initial_template")
                np.testing.assert_array_equal(result["box_xyxy"], self.box)

    def test_zero_or_flat_current_features_freeze_then_local_recovery(self):
        tracker = self.tracker()
        for features in (np.zeros_like(self.initial), np.ones_like(self.initial)):
            result = tracker.step(features)
            self.assertEqual(result["status"], "insufficient_evidence")
            np.testing.assert_array_equal(result["box_xyxy"], self.box)
        recovered = tracker.step(feature_pattern(self.points, dx=8))
        self.assertEqual(recovered["status"], "tracking")
        np.testing.assert_allclose(recovered["box_xyxy"], self.box + [8, 0, 8, 0])

    def test_low_score_freezes_and_default_template_is_unchanged(self):
        tracker = self.tracker(minimum_score=0.99999)
        template = tracker.template.copy()
        result = tracker.step(-self.initial)
        self.assertEqual(result["status"], "uncertain")
        np.testing.assert_array_equal(result["box_xyxy"], self.box)
        np.testing.assert_array_equal(tracker.template, template)
        result = tracker.step(self.initial)
        self.assertEqual(result["status"], "tracking")
        np.testing.assert_array_equal(tracker.template, template)

    def test_manual_box_and_feature_inputs_are_not_aliased(self):
        points, box, initial, scale = self.points.copy(), self.box.copy(), self.initial.copy(), np.ones(2)
        tracker = NeuralTemplateTracker(points, box, initial, channel_scale=scale)
        points[:] = 0
        box[:] = 0
        initial[:] = 0
        scale[:] = 100
        result = tracker.step(self.initial)
        np.testing.assert_array_equal(result["box_xyxy"], self.box)
        self.assertEqual(result["status"], "tracking")

    def test_fractional_initial_box_keeps_fractional_offset(self):
        box = self.box + 0.75
        tracker = NeuralTemplateTracker(self.points, box, self.initial)
        result = tracker.step(feature_pattern(self.points, dx=-8, dy=4))
        np.testing.assert_allclose(result["box_xyxy"], box + [-8, 4, -8, 4])

    def test_one_channel_baseline_works_and_scale_divides_channels(self):
        tracker = NeuralTemplateTracker(self.points, self.box, self.initial[:1], channel_scale=[2.])
        result = tracker.step(feature_pattern(self.points, dx=8)[:1])
        np.testing.assert_allclose(result["box_xyxy"], self.box + [8, 0, 8, 0])
        self.assertNotIn("label", result)
        self.assertNotIn("class_id", result)

    def test_large_resting_offsets_do_not_erase_small_spatial_signals(self):
        initial = self.initial * 0.001 + np.asarray([[1000.], [-250.]])
        tracker = NeuralTemplateTracker(self.points, self.box, initial)
        later = feature_pattern(self.points, dx=8, dy=4) * 0.001 + np.asarray([[1200.], [-190.]])
        result = tracker.step(later)
        self.assertEqual(result["status"], "tracking")
        np.testing.assert_allclose(result["box_xyxy"], self.box + [8, 4, 8, 4])

    def test_radius_is_a_displacement_disk(self):
        tracker = self.tracker(search_radius_px=12, minimum_score=-1)
        result = tracker.step(feature_pattern(self.points, dx=24, dy=24))
        self.assertLessEqual(np.linalg.norm(result["displacement_xy"]), 12)

    def test_insufficient_hull_support_is_not_extrapolated(self):
        sparse = np.asarray([[100., 100.], [100., 280.], [280., 100.]])
        initial = np.asarray([[0., 2., 1.]])
        tracker = NeuralTemplateTracker(sparse, [240., 240., 300., 300.], initial)
        result = tracker.step(initial)
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["template_supported_points"], 0)

    def test_kernel_edge_receptors_are_excluded(self):
        points = np.vstack((self.points, [[0, 0], [390, 390]]))
        features = np.column_stack((self.initial, [[1e6, -1e6], [1e6, -1e6]]))
        tracker = NeuralTemplateTracker(points, self.box, features)
        result = tracker.step(features)
        self.assertEqual(result["status"], "tracking")
        np.testing.assert_array_equal(result["box_xyxy"], self.box)
        self.assertEqual(len(tracker.receptor_indices), len(self.points))

    def test_symmetry_tie_prefers_no_movement(self):
        row, col = self.points.T
        periodic = (np.cos(col * np.pi / 12)[None, :] + 0.1 * np.sin(row * np.pi / 18))
        tracker = NeuralTemplateTracker(self.points, self.box, periodic)
        result = tracker.step(periodic)
        np.testing.assert_array_equal(result["displacement_xy"], [0, 0])

    def test_invalid_step_does_not_mutate_state(self):
        tracker = self.tracker()
        for dt in (0, -1, np.nan, True, [0.02]):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                tracker.step(self.initial, dt)
        for features in (self.initial[:, :-1], np.full_like(self.initial, np.nan)):
            with self.assertRaises(ValueError):
                tracker.step(features)
        np.testing.assert_array_equal(tracker.box, self.box)
        self.assertEqual(tracker.status, "tracking")

    def test_invalid_constructor_configuration_is_rejected(self):
        for kwargs in ({"grid_step": True}, {"grid_step": 0}, {"search_radius_px": 97},
                       {"minimum_score": np.nan}, {"template_update": 1.1},
                       {"channel_scale": [1]}, {"channel_scale": [0, 1]}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.tracker(**kwargs)


if __name__ == "__main__":
    unittest.main()
