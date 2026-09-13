"""Checks for the independent verifier's boundary and tamper detection."""
import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location(
    "audit_v3_results", Path(__file__).parents[1] / "scripts/audit_v3_results.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class IndependentAuditTests(unittest.TestCase):
    def test_swept_contact_detects_tunnelling_and_includes_tangency(self):
        box = {"low": [1., 1., 1.], "high": [2., 2., 2.]}
        self.assertTrue(audit.segment_hit([0., 1.5, 1.5], [3., 1.5, 1.5], box, 0.))
        self.assertTrue(audit.segment_hit([0., .55, 1.5], [3., .55, 1.5], box, .45))
        self.assertFalse(audit.segment_hit([0., .54, 1.5], [3., .54, 1.5], box, .45))

    def test_dynamic_box_comparison_detects_hidden_motion_and_duplicate_ids(self):
        box = {"id": "fixture", "low": [1., 1., 1.], "high": [2., 2., 2.], "velocity": [0., 1., 0.]}
        self.assertTrue(audit.boxes_equal([box], [box]))
        self.assertFalse(audit.boxes_equal([], [box]))
        self.assertFalse(audit.boxes_equal([box, box], [box, box]))
        changed = {**box, "velocity": [0., 0., 0.]}
        self.assertFalse(audit.boxes_equal([changed], [box]))

    def test_motion_reconstruction_supports_axis_swap(self):
        ybox = {"id": "fixture", "low": [1., 2., 3.], "high": [2., 3., 4.], "velocity": [0., 1., 0.]}
        xbox = {**ybox, "low": [2., 1., 3.], "high": [3., 2., 4.], "velocity": [1., 0., 0.]}
        y = audit.scene_at({"seed": 12, "obstacles": [ybox]}, 4.)[0]
        x = audit.scene_at({"seed": 12, "obstacles": [xbox]}, 4.)[0]
        self.assertEqual(x["low"], [y["low"][1], y["low"][0], y["low"][2]])
        self.assertEqual(x["velocity"], [y["velocity"][1], y["velocity"][0], y["velocity"][2]])

    def test_motion_reconstruction_preserves_reflected_direction(self):
        positive = {"id": "fixture", "low": [1., 2., 3.], "high": [2., 3., 4.], "velocity": [0., 1., 0.]}
        negative = {**positive, "velocity": [0., -1., 0.]}
        forward = audit.scene_at({"seed": 12, "obstacles": [positive]}, 4.)[0]
        backward = audit.scene_at({"seed": 12, "obstacles": [negative]}, 4.)[0]
        self.assertAlmostEqual(forward["low"][1] + backward["low"][1], 4.)
        self.assertAlmostEqual(forward["velocity"][1], -backward["velocity"][1])

    def test_stationary_zero_wind_power_is_180_watts(self):
        p, v, energy = audit.physics_prediction(
            {"p": [1., 2., 3.], "v": [0., 0., 0.], "t": 0., "command": {"velocity": [0., 0., 0.]}},
            {"dt": .2, "seed": 0, "wind": [0., 0., 0.]})
        self.assertEqual(p, [1., 2., 3.])
        self.assertEqual(v, [0., 0., 0.])
        self.assertAlmostEqual(energy, .01)


if __name__ == "__main__":
    unittest.main()
