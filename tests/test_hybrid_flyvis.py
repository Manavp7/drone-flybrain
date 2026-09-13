"""Numerical contracts for the hybrid neural boundary; no checkpoint loaded."""
import unittest

import numpy as np

from experiments.hybrid_flyvis import (
    FEATURE_NAMES, HybridFlyvis, population_features, validate_mask,
)


def lattice_centers():
    return np.array([[int(13*(a+b/2))+195, 13*b+195] for a in range(-15, 16)
                     for b in range(max(-15, -15-a), min(15, 15-a)+1)])


class MaskContractTests(unittest.TestCase):
    def setUp(self):
        self.mask = np.full((391, 391), .5, np.float32)

    def test_integration_steps_are_exact(self):
        self.assertEqual(validate_mask(self.mask), 5)
        self.assertEqual(validate_mask(self.mask, .04), 2)
        self.assertEqual(validate_mask(self.mask, .06), 3)

    def test_bad_hold_rejected(self):
        for hold in (True, 0, -.1, .01, .03, .21, float('nan'), float('inf'), '.1'):
            with self.subTest(hold=hold), self.assertRaises(ValueError):
                validate_mask(self.mask, hold)

    def test_invalid_input_never_advances_state(self):
        model = HybridFlyvis.__new__(HybridFlyvis)
        initial = model._state = object()
        model._time_s = .4
        bad_masks = [self.mask.tolist(), self.mask.astype(np.float64), self.mask[:390],
                     np.full_like(self.mask, np.nan), np.full_like(self.mask, -.01),
                     np.full_like(self.mask, 1.01)]
        for mask in bad_masks:
            with self.assertRaises(ValueError):
                model.step(mask)
            self.assertIs(model._state, initial)
            self.assertEqual(model._time_s, .4)


class PopulationReadoutTests(unittest.TestCase):
    def setUp(self):
        self.centers = lattice_centers()
        self.indices = np.arange(721, dtype=np.int64)
        self.baseline = np.full(45669, .75, np.float32)
        self.activity = self.baseline.copy()

    def read(self):
        return population_features(self.activity, self.baseline, self.indices, self.centers)

    def test_blank_has_no_invented_center(self):
        features, diagnostics = self.read()
        np.testing.assert_array_equal(features, np.zeros(len(FEATURE_NAMES)))
        self.assertFalse(diagnostics['valid'])

    def test_coordinate_moments_come_from_neural_sites(self):
        chosen = [np.flatnonzero(np.all(self.centers == [195, 169], axis=1))[0],
                  np.flatnonzero(np.all(self.centers == [195, 221], axis=1))[0]]
        self.activity[chosen] += [.25, .75]
        features, diagnostics = self.read()
        self.assertTrue(diagnostics['valid'])
        expected_x = (-26*.25 + 26*.75) / 195
        expected_variance = (.25*(-26/195-expected_x)**2
                             + .75*(26/195-expected_x)**2)
        np.testing.assert_allclose(features[:4], [expected_x, 0, np.sqrt(expected_variance), 0])
        self.assertEqual(features[4], 0)
        self.assertAlmostEqual(features[5], np.log(2))

    def test_non_l2_population_cannot_change_readout(self):
        self.activity[360] += .5
        before, _ = self.read()
        self.activity[721:] = np.linspace(-100, 100, 45669-721)
        after, _ = self.read()
        np.testing.assert_array_equal(before, after)

    def test_signed_neural_contrast_is_symmetric(self):
        self.activity[360] += .5
        positive, _ = self.read()
        self.activity[360] -= 1
        negative, _ = self.read()
        np.testing.assert_array_equal(positive, negative)

    def test_incomplete_receptor_edges_are_excluded(self):
        outside = ~np.all((self.centers >= 6) & (self.centers <= 384), axis=1)
        self.activity[self.indices[outside]] += 100
        features, diagnostic = self.read()
        self.assertFalse(diagnostic['valid'])
        np.testing.assert_array_equal(features, np.zeros(8))

    def test_subthreshold_noise_is_not_a_target(self):
        self.activity[:721] += .001
        features, diagnostic = self.read()
        self.assertFalse(diagnostic['valid'])
        np.testing.assert_array_equal(features, np.zeros(8))

    def test_nonfinite_or_wrong_population_rejected(self):
        for activity in (self.activity[:-1], self.activity * np.nan):
            with self.assertRaises(ValueError):
                population_features(activity, self.baseline, self.indices, self.centers)
        with self.assertRaises(ValueError):
            population_features(self.activity, self.baseline, np.zeros(721, int), self.centers)


if __name__ == '__main__':
    unittest.main()
