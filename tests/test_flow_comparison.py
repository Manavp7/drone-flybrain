"""Numerical contract tests, not claims about real neural performance."""
import unittest

import numpy as np

from experiments.compare_flow import agreement, align_pairs, decoder_unit_fields


class PairAlignmentTests(unittest.TestCase):
    def test_irregular_pts_never_uses_step_before_current_image(self):
        frames = np.array([0., .071, .129, .191, .249])
        stimulus = np.arange(12) * .02
        pairs, steps = align_pairs(frames, stimulus, stimulus + .02, warmup_s=0)
        np.testing.assert_array_equal(pairs, [0, 1, 2])
        np.testing.assert_array_equal(steps, [4, 7, 10])
        self.assertTrue(np.all(stimulus[steps] >= frames[pairs + 1]))

    def test_fixed_warmup_offset_and_exact_boundary(self):
        frames = np.array([5., 5.25, 5.5, 5.75, 6.])
        stimulus = 5 + np.arange(50) * .02
        pairs, steps = align_pairs(frames, stimulus, stimulus + .02)
        np.testing.assert_array_equal(pairs, [1, 2])
        np.testing.assert_array_equal(steps, [25, 38])

    def test_corrupt_times_rejected(self):
        for frames, stimulus, response in (([0, 0], [0, .1], [.1, .2]),
                                          ([0, .1], [0, .1], [.1]),
                                          ([0, .1], [0, .1], [0, .1])):
            with self.assertRaises(ValueError):
                align_pairs(frames, stimulus, response)


class FlowUnitsTests(unittest.TestCase):
    def test_right_down_to_right_up_with_original_training_height(self):
        field = np.full((3, 5, 2), [436., 218.], dtype=np.float32)
        actual = decoder_unit_fields(field, 1 / 24)
        np.testing.assert_allclose(actual[0], 1.)
        np.testing.assert_allclose(actual[1], -.5)
        np.testing.assert_array_equal(field[0, 0], [436., 218.])

    def test_actual_elapsed_time_scales_variable_rate_displacement(self):
        field = np.full((2, 2, 2), [3., -6.], dtype=np.float32)
        np.testing.assert_allclose(decoder_unit_fields(field * 2, .08),
                                   decoder_unit_fields(field, .04))
        for dt in (0, -1, np.nan):
            with self.assertRaises(ValueError):
                decoder_unit_fields(field, dt)


class AgreementTests(unittest.TestCase):
    def test_opposite_direction_and_vector_difference(self):
        ref = np.array([[2., 0., 0.], [0., 2., 0.]])
        result = agreement(-ref, ref)
        self.assertEqual(result["mean_cosine_nonzero_reference"], -1.)
        self.assertEqual(result["cosine_receptor_count"], 2)
        self.assertAlmostEqual(result["mean_absolute_vector_difference"], 8 / 3)

    def test_zero_motion_has_no_direction_metric(self):
        result = agreement(np.zeros((2, 3)), np.zeros((2, 3)))
        self.assertIsNone(result["mean_cosine_nonzero_reference"])
        self.assertEqual(result["cosine_receptor_count"], 0)


if __name__ == "__main__":
    unittest.main()
