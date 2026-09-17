"""Raw camera motion timing, units and frozen evaluation without heavy inference."""
import json
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mantis_motion import (DT, POPULATIONS, RawMotionExperiment, benchmark_definition,
    calculate_pixel_flow, score_vectors, summarize_flow, translated_texture, run_benchmark, _MotionBackend)


def centers():
    return np.column_stack((np.linspace(20, 370, 721), np.linspace(20, 370, 721)))


class Backend:
    def __init__(self, manifest):
        self.centers_rc = centers()
        self.provenance = {'manifest_sha256': 'test-double', 'control_authority': False}
        self.resets = 0
        self.calls = []
        self.conventional_calls = []

    def reset(self):
        self.resets += 1
        self.calls = []

    def prepare(self, rgb):
        gray = rgb[:, :, 0].copy()
        return gray, int(gray[0, 0]), {'source_hw': list(gray.shape)}

    def advance(self, retina, steps):
        self.calls.append((retina, steps))
        return np.zeros(45669, np.float32)

    def decode(self, activity):
        return np.repeat(np.array([[.2], [-.1]]), 721, axis=1)

    def conventional(self, previous, current, delta):
        self.conventional_calls.append((int(previous[0, 0]), int(current[0, 0]), delta))
        return np.repeat(np.array([[.15], [-.05]]), 721, axis=1)

    def populations(self, activity):
        return {name: {'mean': .1, 'std': .2, 'count': 721} for name in POPULATIONS}


def image(value=50):
    return np.full((391, 391, 3), value, np.uint8)


class Tensor:
    def __init__(self, data):
        self.data = np.asarray(data)

    def clone(self):
        return Tensor(self.data.copy())

    def __getitem__(self, index):
        return Tensor(self.data[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.data


class Namespace(dict):
    def __getattr__(self, key):
        return self[key]


class ResetNetwork:
    def __init__(self):
        self.fade_calls = 0

    def _state_api(self, state):
        state['sources'] = Namespace(activity=state.nodes.activity)
        state['targets'] = Namespace(activity=state.nodes.activity)
        return state

    def fade_in_state(self, seconds, dt, retina):
        self.fade_calls += 1
        return self._state_api(Namespace(nodes=Namespace(activity=Tensor(np.full((1, 45669), .5, np.float32))),
                                        edges=Namespace(example=Tensor(np.array([1., 2.], np.float32)))))


def reset_backend():
    backend = object.__new__(_MotionBackend)
    decoder_inputs = []
    backend.adapter = SimpleNamespace(
        network=ResetNetwork(), device='test',
        torch=SimpleNamespace(inference_mode=nullcontext, float32=np.float32,
                              full=lambda shape, value, **kwargs: Tensor(np.full(shape, value, np.float32))),
        eye=lambda value: Tensor(np.full((1, 1, 1, 721), .5, np.float32)),
        decode_flow=lambda activity: decoder_inputs.append(activity.copy()))
    backend._blank_state = None
    backend._decoder_warmed = False
    return backend, decoder_inputs


class RawMotionTests(unittest.TestCase):
    def test_decoder_warms_during_initialization_without_advancing_camera_state(self):
        backend, decoded = reset_backend()
        backend.reset()
        self.assertEqual(backend.adapter.network.fade_calls, 1)
        self.assertEqual(len(decoded), 1)
        np.testing.assert_array_equal(decoded[0], np.full((1, 45669), .5, np.float32))
        backend.reset()
        self.assertEqual(backend.adapter.network.fade_calls, 1)
        self.assertEqual(len(decoded), 1)

    def test_cached_reset_clones_all_dynamic_tensors_and_rebinds_views(self):
        backend, _ = reset_backend()
        backend.reset()
        first = backend.state
        first.nodes.activity.data[:] = 99.
        first.edges.example.data[:] = -2.
        backend.reset()
        restored = backend.state
        np.testing.assert_array_equal(restored.nodes.activity.data, np.full((1, 45669), .5, np.float32))
        np.testing.assert_array_equal(restored.edges.example.data, [1., 2.])
        self.assertIsNot(first.nodes.activity, restored.nodes.activity)
        self.assertIsNot(restored.nodes.activity, backend._blank_state.nodes.activity)
        self.assertIs(restored.sources.activity, restored.nodes.activity)
        self.assertIs(restored.targets.activity, restored.nodes.activity)
        self.assertEqual(backend.adapter.network.fade_calls, 1)

    def test_frozen_step_loop_reuses_parameters_and_preserves_every_step(self):
        backend, _ = reset_backend()
        backend.reset()
        parameter = SimpleNamespace(_version=0, requires_grad=False)
        network = backend.adapter.network
        network.parameters = lambda: [parameter]
        network.training = False
        calls = []

        class Stimulus:
            def zero(self, batches, frames):
                calls.append(('zero', batches, frames))

            def add_input(self, retina):
                self.retina = retina
                calls.append(('input', id(retina)))

            def __call__(self):
                return Tensor(np.full((1, 1, 45669), float(self.retina.data.flat[0]), np.float32))

        network.stimulus = Stimulus()
        backend._frozen_params = object()
        backend._parameter_signature = backend._parameters_signature()
        updates = []

        def next_state(params, state, frame, dt):
            self.assertIs(params, backend._frozen_params)
            self.assertEqual(dt, DT)
            updates.append(frame.data.copy())
            return network._state_api(Namespace(nodes=Namespace(
                activity=Tensor(state.nodes.activity.data+frame.data*dt)), edges=Namespace()))

        network._next_state = next_state
        value = backend.advance(Tensor(np.array([.25], np.float32)), 28)
        self.assertEqual(len(updates), 28)
        self.assertEqual([item for item in calls if item[0] == 'zero'], [('zero', 1, 1)])
        self.assertEqual(len([item for item in calls if item[0] == 'input']), 1)
        expected = np.full((45669,), .5, np.float32)
        for _ in range(28):
            expected += np.float32(.25)*DT
        np.testing.assert_array_equal(value, expected)
        previous = value.copy()
        parameter._version += 1
        with self.assertRaisesRegex(ValueError, 'parameters changed'):
            backend.advance(Tensor(np.array([.25], np.float32)), 1)
        np.testing.assert_array_equal(backend.state.nodes.activity.data[0], previous)
        for invalid in (True, 0, 1.5):
            with self.assertRaises(ValueError):
                backend.advance(Tensor(np.array([.25], np.float32)), invalid)

    def test_decoder_units_and_downward_image_axis_are_explicit(self):
        flow = np.repeat(np.array([[2.], [-3.]]), 721, axis=1)
        result = summarize_flow(flow, centers())
        np.testing.assert_allclose(result['mean_decoder_xy'], [2., -3.])
        np.testing.assert_allclose(result['nominal_velocity_px_s'], np.array([2., 3.])*436*24/169)
        self.assertAlmostEqual(result['rms_decoder_magnitude'], np.sqrt(13.))
        self.assertAlmostEqual(result['mean_decoder_magnitude'], np.sqrt(13.))
        boundary = centers()
        boundary[:100] = [0, 0]
        flow[:, :100] = 99999
        self.assertEqual(summarize_flow(flow, boundary)['supported_receptors'], 621)
        self.assertEqual(summarize_flow(flow, boundary)['mean_decoder_xy'], [2., -3.])
        with self.assertRaises(ValueError):
            summarize_flow(np.full((2, 721), np.nan), centers())

    @patch('experiments.mantis_motion._MotionBackend', Backend)
    def test_old_image_is_held_before_current_capture_and_response_is_explicit(self):
        experiment = RawMotionExperiment('test-only')
        first = experiment.step(image(30), 0.)
        second = experiment.step(image(90), .25)
        self.assertEqual(experiment.backend.calls, [(30, 1), (30, 12), (90, 1)])
        self.assertAlmostEqual(first['response_time_s'], .02)
        self.assertAlmostEqual(second['stimulus_time_s'], .26)
        self.assertAlmostEqual(second['response_time_s'], .28)
        self.assertGreaterEqual(second['stimulus_time_s'], second['capture_time_s'])
        self.assertEqual(experiment.backend.conventional_calls, [(30, 90, .25)])
        self.assertFalse(second['valid'])
        self.assertFalse(second['control_authority'])
        self.assertNotIn('speed_scale', second)
        self.assertEqual(set(second['populations']), set(POPULATIONS))
        json.dumps(second, allow_nan=False)

    @patch('experiments.mantis_motion._MotionBackend', Backend)
    def test_warmup_and_large_gap_reset_never_bridge_missing_camera_time(self):
        experiment = RawMotionExperiment('test-only')
        experiment.step(image(10), 2.)
        self.assertFalse(experiment.step(image(20), 2.4)['valid'])
        self.assertTrue(experiment.step(image(30), 2.5)['valid'])
        resets = experiment.backend.resets
        result = experiment.step(image(70), 3.2)
        self.assertEqual(result['status'], 'gap-reset')
        self.assertTrue(result['gap_reset'])
        self.assertFalse(result['valid'])
        self.assertIsNone(result['conventional'])
        self.assertEqual(experiment.backend.resets, resets+1)
        self.assertEqual(experiment.backend.calls, [(70, 1)])
        self.assertEqual(result['elapsed_since_reset_s'], 0.)
        self.assertAlmostEqual(result['response_time_s'], 3.22)

    @patch('experiments.mantis_motion._MotionBackend', Backend)
    def test_invalid_inputs_do_not_advance_recurrence(self):
        experiment = RawMotionExperiment('test-only')
        experiment.step(image(), 0.)
        for bad in (True, float('nan'), float('inf'), -.1, 0., .01):
            with self.subTest(time=bad), self.assertRaises(ValueError):
                experiment.step(image(), bad)
        for bad_image in (np.zeros((391, 391)), np.zeros((391, 391, 3)), np.zeros((10, 10, 3), np.uint8)):
            with self.assertRaises(ValueError):
                experiment.step(bad_image, .1)
        self.assertEqual(experiment.backend.calls, [(50, 1)])
        self.assertEqual(experiment.previous_time, 0.)

    @patch('experiments.mantis_motion._MotionBackend', Backend)
    def test_independent_instances_never_share_recurrence(self):
        first, second = RawMotionExperiment('a'), RawMotionExperiment('b')
        first.step(image(40), 0.)
        second.step(image(70), 0.)
        first.step(image(50), .1)
        self.assertEqual(second.backend.calls, [(70, 1)])
        first.reset()
        self.assertEqual(first.backend.calls, [])
        self.assertEqual(second.backend.calls, [(70, 1)])
        self.assertIsNone(first.previous_time)

    def test_frozen_benchmark_separates_seeds_and_never_fits(self):
        definition = benchmark_definition()
        self.assertFalse(definition['fitting'])
        self.assertFalse(definition['parameter_search'])
        self.assertTrue(set(definition['train_seeds']).isdisjoint(definition['heldout_seeds']))
        self.assertEqual(len(definition['cases']), 12)
        for split in ('train', 'heldout'):
            self.assertEqual({case['motion'] for case in definition['cases'] if case['split'] == split},
                             {'stationary', 'left', 'right', 'up', 'down', 'diagonal'})

    def test_texture_translation_has_known_direction_and_baseline_scale(self):
        first = translated_texture(1701, [24., -18.], 0.)[:, :, 0]
        second = translated_texture(1701, [24., -18.], .1)[:, :, 0]
        fields = calculate_pixel_flow(first, second, .1)
        # Undo target normalization directly before retinal filtering.
        velocity = np.median(fields[:, 30:-30, 30:-30], axis=(1, 2))*436*24
        np.testing.assert_allclose(velocity, [24., 18.], atol=1.)
        still = calculate_pixel_flow(first, first, .1)
        self.assertLess(float(abs(still[:, 30:-30, 30:-30]).max()), 1e-4)
        np.testing.assert_array_equal(translated_texture(1701, [0., 0.], 0.),
                                      translated_texture(1701, [0., 0.], 1.2))
        with self.assertRaises(ValueError):
            translated_texture(1701, [999., 0.], 1.)

    def test_scoring_does_not_hide_failed_directions_or_zero_motion(self):
        truth = [[10., 0.], [10., 0.], [0., 0.]]
        good = score_vectors(truth, truth)
        self.assertEqual(good['vector_rmse_px_s'], 0.)
        self.assertEqual(good['mean_direction_cosine'], 1.)
        zero = score_vectors(np.zeros((3, 2)), truth)
        self.assertEqual(zero['direction_coverage'], 0.)
        self.assertIsNone(zero['mean_direction_cosine'])
        wrong = score_vectors([[-10., 0.], [-10., 0.], [1., 1.]], truth)
        self.assertEqual(wrong['mean_direction_cosine'], -1.)
        self.assertGreater(wrong['vector_rmse_px_s'], zero['vector_rmse_px_s'])
        with self.assertRaises(ValueError):
            score_vectors([[float('nan'), 0]], [[0, 0]])

    @patch('experiments.mantis_motion._MotionBackend', Backend)
    def test_benchmark_exports_compact_receipts_and_honest_negative_result(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp).resolve()/'benchmark'
            result = run_benchmark('test-only', output)
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(result['control_authority'])
            self.assertFalse(result['neural_outperformed_both_heldout_baselines'])
            self.assertEqual(set(path.suffix for path in output.iterdir()), {'.json', '.jsonl'})
            self.assertLess(sum(path.stat().st_size for path in output.iterdir()), 2*1024**2)
            self.assertEqual(len(result['cases']), 12)
            self.assertEqual(result['splits']['heldout']['neural']['rows'], 48)
            self.assertIn('did not outperform', result['conclusion'])
            before = (output/'summary.json').read_bytes()
            with self.assertRaises(FileExistsError):
                run_benchmark('test-only', output)
            self.assertEqual((output/'summary.json').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
