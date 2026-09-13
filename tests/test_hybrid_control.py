import unittest
import numpy as np
from experiments.hybrid_control import NeuralReadout, HybridController, active_velocity


class HybridControlTests(unittest.TestCase):
    def setUp(self):
        coef = np.zeros((8, 3)); coef[0, 0] = coef[1, 1] = coef[3, 2] = 1
        self.controller = HybridController(NeuralReadout(np.zeros(8), np.ones(8), coef, np.zeros(3)))
        self.features = np.array([.3, -.2, .1, .25, 0, 1, 1, 1.])

    def command(self, **kwargs):
        args = dict(target_valid=True, neural_valid=True, capture_time_s=0, response_time_s=.1)
        args.update(kwargs)
        return self.controller.command(self.features, **args)

    def test_neural_signs_control_three_axes(self):
        v = self.command()['velocity']
        self.assertTrue(all(x > 0 for x in v))
        self.assertLessEqual(np.linalg.norm(v), 1.2+1e-9)

    def test_loss_brakes_despite_persistent_neural_activity(self):
        self.assertEqual(self.command(target_valid=False)['velocity'], [0, 0, 0])

    def test_zero_feature_ablation_has_no_bias_command(self):
        self.features[:] = 0
        self.assertEqual(self.command()['reason'], 'neural_evidence_unavailable')

    def test_nonfinite_neural_brakes(self):
        self.features[0] = np.nan
        self.assertEqual(self.command()['mode'], 'brake')

    def test_late_response_brakes(self):
        self.assertEqual(self.command(response_time_s=.4)['reason'], 'invalid_response_timing')

    def test_future_and_expired_commands_cannot_act(self):
        cmd = self.command()
        self.assertEqual(active_velocity(cmd, .09), (0, 0, 0))
        self.assertNotEqual(active_velocity(cmd, .1), (0, 0, 0))
        self.assertEqual(active_velocity(cmd, .25), (0, 0, 0))

    def test_fit_known_mapping_and_roundtrip(self):
        rng = np.random.default_rng(79)
        x = rng.normal(size=(100, 8)); coef = rng.normal(size=(8, 3)); y = x@coef+.3
        fitted = NeuralReadout.fit(x, y, ridge=1e-8)
        fresh = rng.normal(size=(10, 8))
        np.testing.assert_allclose(fitted.predict(fresh), fresh@coef+.3, atol=1e-6)
        np.testing.assert_array_equal(NeuralReadout(**fitted.to_dict()).predict(fresh), fitted.predict(fresh))

    def test_fit_invalid_data_rejected(self):
        with self.assertRaises(ValueError):
            NeuralReadout.fit(np.zeros((10, 8)), np.full((10, 3), np.nan))


if __name__ == '__main__':
    unittest.main()
