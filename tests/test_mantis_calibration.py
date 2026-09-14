"""Calibration split isolation, feature-only prediction and honest acceptance."""
from copy import deepcopy
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.hybrid_control import NeuralReadout
from experiments.mantis_calibration import (TRAIN_PHASES, HELDOUT_PHASES, definitions,
    validate_definition, decode_features, fit_training, evaluate_heldout,
    freeze_definition, run, RIDGE)


def data(seed=1, count=16):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, .1, (count, 8))
    x[:, 5] += 1.
    x[:, 7] += 1.
    y = np.column_stack([x[:, 0], x[:, 1], .35+x[:, 2]])
    return dict(features=x, labels=y, valid=np.ones(count, bool))


def readout(x_bias=0., y_bias=0., height_bias=0.):
    coefficients = np.zeros((8, 3))
    coefficients[0, 0] = coefficients[1, 1] = coefficients[2, 2] = 1.
    return NeuralReadout(np.zeros(8), np.ones(8), coefficients,
                         [x_bias, y_bias, .35+height_bias])


class MantisCalibrationTests(unittest.TestCase):
    def test_definition_is_deterministic_disjoint_and_bounded(self):
        a, b = definitions(), definitions()
        self.assertEqual(a, b)
        validate_definition(a)
        self.assertEqual(a['observation_count'], 1283)
        self.assertEqual(a['ridge'], RIDGE)
        self.assertEqual([len(a['phases'][n]['rows']) for n in (*TRAIN_PHASES, *HELDOUT_PHASES)],
                         [675, 64, 480, 64])

    def test_copying_a_training_box_into_holdout_is_rejected(self):
        definition = definitions()
        definition['phases']['heldout_grid']['rows'][0] = deepcopy(definition['phases']['train_grid']['rows'][0])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_definition(definition)

    def test_mislabelled_split_is_rejected(self):
        definition = definitions()
        definition['phases']['heldout_motion']['split'] = 'training'
        with self.assertRaisesRegex(ValueError, 'split'):
            validate_definition(definition)

    def test_fitting_rejects_heldout_phase_even_if_other_inputs_valid(self):
        training = {name: data(i) for i, name in enumerate(TRAIN_PHASES)}
        first = fit_training(training).to_dict()
        training['heldout_grid'] = data(200)
        with self.assertRaisesRegex(ValueError, 'never held-out'):
            fit_training(training)
        del training['heldout_grid']
        self.assertEqual(first, fit_training(training).to_dict())

    def test_prediction_accepts_features_only_and_does_not_use_labels(self):
        records = data()
        model = fit_training({name: data(i) for i, name in enumerate(TRAIN_PHASES)})
        predicted = decode_features(model, records['features']).copy()
        records['labels'][:] = 1000000.
        np.testing.assert_array_equal(predicted, decode_features(model, records['features']))
        self.assertEqual(tuple(inspect.signature(decode_features).parameters), ('readout', 'features'))
        for forbidden in (np.ones((16, 9)), np.ones((391, 391)), {'bbox_xyxy': [1, 2, 3, 4]}):
            with self.subTest(value=type(forbidden).__name__), self.assertRaises((ValueError, TypeError)):
                decode_features(model, forbidden)

    def test_invalid_neural_features_are_not_replaced_with_geometry(self):
        with self.assertRaises(ValueError):
            decode_features(readout(), np.full(8, np.nan))

    def test_lower_horizontal_error_with_coverage_preserved_passes(self):
        heldout = {name: data(i+20) for i, name in enumerate(HELDOUT_PHASES)}
        result = evaluate_heldout(heldout, readout(x_bias=.02), readout())
        self.assertTrue(result['passed'], result)
        self.assertEqual(result['results']['combined']['observations'], 32)
        self.assertFalse(result['neural_superiority_over_geometry_or_filter_evaluated'])
        json.dumps(result, allow_nan=False)

    def test_worse_error_and_coverage_do_not_pass(self):
        heldout = {name: data(i+20) for i, name in enumerate(HELDOUT_PHASES)}
        result = evaluate_heldout(heldout, readout(), readout(x_bias=.6))
        self.assertFalse(result['passed'])
        self.assertFalse(result['gates']['coverage_preserved'])
        self.assertFalse(result['gates']['horizontal_rmse_improved'])
        self.assertEqual(result['results']['combined']['methods']['calibrated']['count'], 32)

    def test_other_axes_cannot_be_sacrificed_for_horizontal_win(self):
        heldout = {name: data(i+20) for i, name in enumerate(HELDOUT_PHASES)}
        result = evaluate_heldout(heldout, readout(x_bias=.03), readout(y_bias=.1))
        self.assertTrue(result['gates']['horizontal_rmse_improved'])
        self.assertFalse(result['gates']['other_axes_not_materially_worse'])
        self.assertFalse(result['passed'])

    def test_missing_heldout_phase_cannot_be_silently_omitted(self):
        with self.assertRaises(ValueError):
            evaluate_heldout({'heldout_grid': data()}, readout(), readout())

    def test_freeze_refuses_overwrite_before_inference(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'attempt'
            definition = freeze_definition(output)
            self.assertTrue((output/'definition.json').is_file())
            self.assertEqual(len(definition['source_hashes']), 6)
            with self.assertRaises(FileExistsError):
                freeze_definition(output)

    def test_definition_and_candidate_freeze_precede_model_and_holdout(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/'attempt'
            events = []
            class DummyBrain:
                def __init__(self, manifest):
                    self.adapter = type('Adapter', (), {'runtime': {}, 'device': 'fixture'})()
                    if not (output/'definition.json').is_file():
                        raise AssertionError('Model initialized before definition freeze')
                    events.append('model')
            def collector(brain, phase, folder):
                folder.mkdir()
                np.savez_compressed(folder/'neural.npz', features=data()['features'])
                if phase['split'] == 'heldout':
                    self.assertTrue((output/'fit_frozen.json').is_file())
                    self.assertTrue((output/'candidate_readout.json').is_file())
                events.append(phase['split'])
                return data()
            with patch('experiments.mantis_calibration.HybridFlyvis', DummyBrain), \
                 patch('experiments.mantis_calibration.collect_phase', collector):
                run(output)
            self.assertEqual(events, ['model', 'training', 'training', 'heldout', 'heldout'])
            frozen = json.loads((output/'fit_frozen.json').read_text())
            self.assertEqual(frozen['trained_phase_names'], list(TRAIN_PHASES))
            self.assertFalse(frozen['heldout_inference_started'])


if __name__ == '__main__':
    unittest.main()
