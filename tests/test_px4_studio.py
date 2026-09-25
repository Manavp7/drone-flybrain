"""Bounded Studio transport checks; no PX4, model inference or graphics runs."""
import json
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import numpy as np

from experiments.mantis_studio import StudioManager, write_json, px4_selection_intent_is_published
from experiments.mantis_studio_config import validate_config
from experiments.px4_studio import StudioBridge, run_session


class ConfigurationTests(unittest.TestCase):
    def test_preview_intent_is_bounded_and_does_not_require_latest_frame(self):
        state = dict(selection=dict(sequence=20), selection_history=[
            dict(sequence=17, capture_time_s=98., selectable_tracks=[2]),
            dict(sequence=20, capture_time_s=99.8, selectable_tracks=[2])])
        chosen = dict(sequence=17, track_id=2)
        self.assertTrue(px4_selection_intent_is_published(state, chosen, 100.))
        self.assertFalse(px4_selection_intent_is_published(state, chosen, 101.01))
        self.assertFalse(px4_selection_intent_is_published(state, chosen, 97.))
        self.assertFalse(px4_selection_intent_is_published(state, dict(chosen, track_id=3), 100.))
        self.assertFalse(px4_selection_intent_is_published(state, dict(chosen, sequence=18), 100.))
        self.assertFalse(px4_selection_intent_is_published(
            dict(selection_history=state['selection_history']*5), chosen, 100.))

    def test_existing_default_and_explicit_px4_default(self):
        self.assertEqual(validate_config({})['backend'], 'mujoco')
        config = validate_config(dict(backend='px4_sih'))
        self.assertEqual(config['scenario'], 'stationary')
        self.assertEqual(config['target_speed'], 0.)
        self.assertFalse(config['low_noise_sensors'])
        self.assertEqual(config['recording'], 'compact')

    def test_unsupported_px4_settings_are_rejected_not_ignored(self):
        for setting in [dict(scenario='crossing'), dict(method='direct_yolo'),
                        dict(follow_distance_m=4), dict(target_speed=.1), dict(detours=True),
                        dict(motion_mode='observe'), dict(recording='full-research')]:
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                validate_config(dict(backend='px4_sih', **setting))
        config = validate_config(dict(backend='px4_sih'))
        with self.assertRaises(ValueError):
            validate_config(dict(target_speed=0), base=config, live=True)
        self.assertEqual(validate_config({}, base=config, live=True), config)

    def test_diagnostic_noise_requires_boolean_and_px4_backend(self):
        for value in [1, '1', None]:
            with self.assertRaises(ValueError):
                validate_config(dict(backend='px4_sih', low_noise_sensors=value))
        with self.assertRaises(ValueError):
            validate_config(dict(low_noise_sensors=True))
        self.assertTrue(validate_config(dict(backend='px4_sih', low_noise_sensors=True))['low_noise_sensors'])


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.manager = StudioManager(Path(self.temp.name).resolve()/'runs', 32)

    def tearDown(self):
        self.temp.cleanup()

    def start(self, **config):
        child = Mock()
        child.poll.return_value = None
        with patch('experiments.mantis_studio.subprocess.Popen', return_value=child) as spawn, \
                patch('experiments.mantis_studio.threading.Thread.start'):
            self.manager.start(dict(backend='px4_sih', max_recording_mb=16, **config))
        self.assertEqual(spawn.call_args.args[0][4], 'experiments.px4_studio')
        return child

    def test_px4_dispatch_reuses_bounded_manager_and_labels_backend(self):
        self.start()
        state = self.manager.snapshot()['state']
        self.assertEqual(state['backend'], 'px4_sih')
        self.assertTrue(state['simulation_only'])
        self.assertFalse(state['config']['low_noise_sensors'])
        with self.assertRaises(RuntimeError):
            self.manager.start(dict(backend='px4_sih', max_recording_mb=16))

    def test_pause_settings_and_stale_observed_selection_are_rejected(self):
        self.start()
        for mutation in [dict(operation='pause'), dict(settings=dict(target_speed=0.))]:
            with self.assertRaises(ValueError):
                self.manager.command(dict(run_id=self.manager.active, **mutation))
        path = self.manager.run_folder(self.manager.active)/'state.json'
        write_json(path, dict(selection=dict(sequence=5), selection_history=[
            dict(sequence=3, capture_time_s=time.monotonic(), selectable_tracks=[1])]))
        with self.assertRaises(ValueError):
            self.manager.command(dict(run_id=self.manager.active, selection=dict(track_id=1, sequence=2)))
        self.manager.command(dict(run_id=self.manager.active, selection=dict(track_id=1, sequence=3)))
        self.assertEqual(self.manager.control['selection']['sequence'], 3)
        self.manager.command(dict(run_id=self.manager.active, settings={}))
        self.assertEqual(self.manager.control['settings'], {})

    def test_shutdown_allows_landing_and_never_kills_wrapper(self):
        child = self.start()
        self.manager.close()
        child.wait.assert_called_once_with(timeout=120)
        child.terminate.assert_not_called()
        child.kill.assert_not_called()
        self.assertEqual(self.manager.control['operation'], 'stop')

    def test_shutdown_timeout_preserves_owned_cleanup(self):
        child = self.start()
        child.wait.side_effect = subprocess.TimeoutExpired('owned-studio-child', 120)
        with self.assertRaisesRegex(RuntimeError, 'landing/cleanup'):
            self.manager.close()
        child.terminate.assert_not_called()
        child.kill.assert_not_called()


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name).resolve()
        self.config = validate_config(dict(backend='px4_sih'))
        write_json(self.folder/'config.json', self.config)
        self.controls = dict(revision=0, operation='run', settings={}, selection=None)
        write_json(self.folder/'control.json', self.controls)
        self.bridge = StudioBridge(self.folder, self.config)

    def tearDown(self):
        if not self.bridge.closed:
            self.bridge.finish(dict(error=None))
        self.temp.cleanup()

    def test_no_automatic_selection_and_signal_stop_is_latched(self):
        self.assertIsNone(self.bridge.control()['selection'])
        self.bridge.request_stop(signal.SIGTERM, None)
        self.assertEqual(self.bridge.control()['operation'], 'stop')
        write_json(self.folder/'control.json', dict(self.controls, revision=1))
        self.assertEqual(self.bridge.control()['operation'], 'stop')

    def test_control_rechecks_shape_revision_and_no_pause(self):
        for change in [dict(operation='pause'), dict(revision=True),
                       dict(selection=dict(revision=0, track_id=1, sequence=-1)),
                       dict(settings=dict(target_speed=.1))]:
            write_json(self.folder/'control.json', dict(self.controls, **change))
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.bridge.control()
        write_json(self.folder/'control.json', dict(self.controls, revision=3))
        self.bridge.control()
        write_json(self.folder/'control.json', dict(self.controls, revision=2))
        with self.assertRaisesRegex(ValueError, 'regressed'):
            self.bridge.control()

    def test_frame_and_selection_sequence_must_match(self):
        frame = np.zeros((2, 2, 3), np.uint8)
        with self.assertRaises(ValueError):
            self.bridge.publish(dict(frame=dict(sequence=3), selection=dict(sequence=2)), frame, frame)
        with self.assertRaises(ValueError):
            self.bridge.publish({}, frame, None)

    def test_preview_cache_keeps_original_times_and_live_loss_status(self):
        image = np.zeros((2, 2, 3), np.uint8)
        cv = Mock(COLOR_RGB2BGR=1, IMWRITE_JPEG_QUALITY=2)
        cv.cvtColor.side_effect = lambda value, mode: value
        cv.imencode.return_value = (True, np.frombuffer(b'jpeg', np.uint8))
        captured = time.monotonic()-1.
        with patch.object(self.bridge, '_cv2', cv):
            for sequence in range(10):
                snapshot = dict(frame=dict(sequence=sequence, capture_time_s=captured+.001*sequence),
                    selection=dict(sequence=sequence, track_id=1, held=False,
                                   people=[dict(track_id=1, selectable=True)]))
                self.bridge._write_images(snapshot, image, image)
            self.assertEqual([p['sequence'] for p in self.bridge.selection_history], list(range(2, 10)))
            with self.assertRaisesRegex(ValueError, 'timestamp changed'):
                self.bridge._write_images(dict(snapshot,
                    frame=dict(sequence=9, capture_time_s=captured+.1)), image, image)
        self.bridge.publish(dict(phase='holding', selection=dict(
            track_id=1, held=True, sequence=None, people=[], reason='selected_person_lost')))
        self.bridge.finish(dict(error=None))
        state = json.loads((self.folder/'state.json').read_text())
        self.assertTrue(state['selection']['held'])
        self.assertIsNone(state['selection']['sequence'])
        self.assertEqual(state['frame_selection']['sequence'], 9)
        self.assertEqual(state['selection_history'][0]['capture_time_s'], captured+.002)

    def test_slow_preview_drops_without_blocking_controls(self):
        entered, resume = threading.Event(), threading.Event()
        original = write_json
        def delayed_write(path, value):
            if path.name == 'state.json':
                entered.set()
                if not resume.wait(2):
                    raise RuntimeError('Test preview timed out')
            original(path, value)
        with patch('experiments.px4_studio.write_json', side_effect=delayed_write):
            try:
                self.assertTrue(self.bridge.publish(dict(phase='hovering')))
                self.assertTrue(entered.wait(1))
                self.assertTrue(self.bridge.publish(dict(phase='awaiting_selection')))
                self.assertFalse(self.bridge.publish(dict(phase='following')))
                self.assertEqual(self.bridge.control()['operation'], 'run')
            finally:
                resume.set()
                self.bridge.finish(dict(error=None))
        self.assertEqual(self.bridge.dropped, 1)
        self.assertEqual(json.loads((self.folder/'state.json').read_text())['phase'], 'completed')

    def test_recording_error_cannot_become_completed_status(self):
        self.bridge.recording_folder.mkdir()
        write_json(self.bridge.recording_folder/'summary.json', dict(status='error', error='encoder failed'))
        with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
            self.bridge.finish(dict(error=None))
        state = json.loads((self.folder/'state.json').read_text())
        self.assertEqual(state['phase'], 'error')

    def test_wrapper_uses_new_output_and_explicit_none_selection(self):
        self.bridge.finish(dict(error=None))
        def fake_runner(folder, **kwargs):
            self.assertEqual(folder, self.folder/'flight')
            self.assertIsNone(kwargs['selected_track'])
            self.assertFalse(kwargs['low_noise_sensors'])
            self.assertEqual(kwargs['estimator_profile'], 'sih_velocity_settled')
            self.assertEqual(kwargs['studio'].recording_folder, self.folder/'capture')
            return dict(error=None, actual_px4=False)
        result = run_session(self.folder, runner=fake_runner)
        self.assertFalse(result['actual_px4'])


class EncoderStartupTests(unittest.TestCase):
    def test_codec_initializes_before_preview_thread_starts(self):
        order = []
        cv = Mock(COLOR_RGB2BGR=1, IMWRITE_JPEG_QUALITY=2)
        cv.cvtColor.side_effect = lambda image, mode: image
        def encode(*args):
            order.append('encode')
            return True, np.frombuffer(b'jpeg', np.uint8)
        cv.imencode.side_effect = encode
        child = Mock()
        child.start.side_effect = lambda: order.append('thread_start')
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('sys.modules', cv2=cv), \
                patch('experiments.px4_studio.threading.Thread', return_value=child):
            bridge = StudioBridge(Path(directory).resolve(), dict(backend='px4_sih'))
        self.assertEqual(order, ['encode', 'thread_start'])
        self.assertIs(bridge._cv2, cv)
        sample = cv.cvtColor.call_args.args[0]
        self.assertEqual(sample.shape, (8, 8, 3))
        self.assertEqual(sample.dtype, np.uint8)

    def test_codec_failure_prevents_preview_thread_creation(self):
        cv = Mock(COLOR_RGB2BGR=1, IMWRITE_JPEG_QUALITY=2)
        cv.imencode.return_value = (False, None)
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict('sys.modules', cv2=cv), \
                patch('experiments.px4_studio.threading.Thread') as thread:
            with self.assertRaisesRegex(RuntimeError, 'initialize Studio preview encoder'):
                StudioBridge(Path(directory).resolve(), dict(backend='px4_sih'))
            thread.assert_not_called()


if __name__ == '__main__':
    unittest.main()
