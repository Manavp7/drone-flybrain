"""Control validation, bounded local transport and causal experimental braking."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock
from http.server import ThreadingHTTPServer

from experiments.mantis_studio_config import validate_config
from experiments.mantis_studio import StudioManager, make_handler
from experiments.mantis_session import motion_brake_scale, selectable_preview_is_fresh


class ConfigTests(unittest.TestCase):
    def test_slow_first_inference_refreshes_preview_instead_of_deadlocking_selection(self):
        selection = dict(capture_time_s=0., people=[dict(selectable=True)])
        self.assertFalse(selectable_preview_is_fresh(selection, 1.2))
        selection['capture_time_s'] = 1.2
        self.assertTrue(selectable_preview_is_fresh(selection, 1.4))
        self.assertFalse(selectable_preview_is_fresh(dict(selection, people=[]), 1.4))

    def test_defaults_are_compact_and_motion_off(self):
        config = validate_config({})
        self.assertEqual(config['recording'], 'compact')
        self.assertEqual(config['motion_mode'], 'off')
        self.assertLessEqual(config['max_recording_mb'], 64)

    def test_nonfinite_booleans_unknown_fields_and_oversized_settings_rejected(self):
        for change in [{'duration_s': True}, {'duration_s': float('nan')}, {'follow_distance_m': float('inf')},
                       {'duration_s': 999}, {'max_recording_mb': 500}, {'max_recording_mb': 16.5},
                       {'scenario': 'exec'}, {'detours': 1}, {'path': '/tmp'}, {'method': None}]:
            with self.subTest(change=change), self.assertRaises((ValueError, TypeError)):
                validate_config(change)

    def test_live_changes_cannot_replace_models_or_recording_budget(self):
        for change in [{'method': 'direct_yolo'}, {'recording': 'full-research'}, {'max_recording_mb': 128}]:
            with self.assertRaises(ValueError):
                validate_config(change, live=True)
        self.assertEqual(validate_config({'follow_distance_m': 4.}, live=True)['follow_distance_m'], 4.)


class BrakeTests(unittest.TestCase):
    def result(self, magnitude=.03):
        return dict(valid=True, capture_time_s=1., response_time_s=1.02, available_time_s=1.2,
                    neural=dict(rms_decoder_magnitude=magnitude))

    def test_no_future_motion_output_can_affect_earlier_motor_ticks(self):
        r = self.result()
        self.assertEqual(motion_brake_scale(r, 1.19), 0.)
        self.assertGreater(motion_brake_scale(r, 1.2), 0.)

    def test_unknown_stale_nonfinite_and_invalid_clocks_hold(self):
        for field in ['capture_time_s', 'response_time_s', 'available_time_s']:
            for invalid in [float('nan'), float('inf'), True, '1', None]:
                r = self.result(); r[field] = invalid
                self.assertEqual(motion_brake_scale(r, 1.3), 0.)
        self.assertEqual(motion_brake_scale(self.result(), float('nan')), 0.)
        self.assertEqual(motion_brake_scale(self.result(), 2.), 0.)
        self.assertEqual(motion_brake_scale(dict(self.result(), valid=False), 1.3), 0.)

    def test_reduction_is_bounded_and_monotonic(self):
        values = [motion_brake_scale(self.result(m), 1.3) for m in [0., .01, .04, .1, 10.]]
        self.assertTrue(all(0 <= x <= 1 for x in values))
        self.assertEqual(values, sorted(values, reverse=True))


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.manager = StudioManager(self.root/'runs', 32)
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(self.manager))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.port = self.server.server_port

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()
        self.temp.cleanup()

    def request(self, method, path, value=None, headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=5)
        h = {'Content-Type': 'application/json', 'X-Mantis-Token': self.manager.token}
        h.update(headers or {})
        if path == '/api/control' and isinstance(value, dict):
            value = dict(run_id=self.manager.active, **value)
        body = json.dumps(value).encode() if value is not None else None
        connection.request(method, path, body=body, headers=h)
        response = connection.getresponse()
        data, status, response_headers = response.read(), response.status, dict(response.getheaders())
        connection.close()
        return status, data, response_headers

    def test_page_status_and_host_rebinding(self):
        status, data, _ = self.request('GET', '/')
        self.assertEqual(status, 200); self.assertIn(self.manager.token.encode(), data)
        self.assertNotIn(b'__MANTIS_TOKEN__', data)
        self.assertEqual(self.request('GET', '/api/state')[0], 200)
        self.assertEqual(self.request('GET', '/', headers={'Host': 'evil.example'})[0], 403)

    def test_cross_origin_or_no_token_cannot_launch(self):
        self.assertEqual(self.request('POST', '/api/start', {}, {'Origin': 'https://evil.example'})[0], 403)
        self.assertEqual(self.request('POST', '/api/start', {}, {'X-Mantis-Token': ''})[0], 403)
        self.assertFalse(self.manager.running())

    def test_storage_reservation_blocks_launch_without_removing_saved_run(self):
        saved = self.manager.root/'keep.txt'; saved.write_text('original')
        with patch('experiments.mantis_studio.subprocess.Popen') as spawn:
            self.assertEqual(self.request('POST', '/api/start', {'max_recording_mb': 64})[0], 409)
            spawn.assert_not_called()
        self.assertEqual(saved.read_text(), 'original')

    def test_no_arbitrary_path_serving_or_symlink_following(self):
        for path in ['/media/../../secret', '/media/run-20260917T000000-12345678/../../secret', '/api/frame?id=../../secret']:
            self.assertEqual(self.request('GET', path)[0], 400)
        identity = 'run-20260917T000000-12345678'
        folder = self.manager.root/identity; folder.mkdir()
        (folder/'capture').symlink_to(self.root)
        (self.root/'summary.json').write_text('private')
        self.assertEqual(self.request('GET', f'/media/{identity}/summary.json')[0], 400)

    def test_recorded_video_byte_ranges_support_seeking(self):
        identity = 'run-20260917T000000-12345678'
        folder = self.manager.root/identity/'capture'; folder.mkdir(parents=True)
        (folder/'camera.mp4').write_bytes(b'0123456789')
        status, body, headers = self.request('GET', f'/media/{identity}/camera.mp4', headers={'Range': 'bytes=3-6'})
        self.assertEqual((status, body), (206, b'3456'))
        self.assertEqual(headers['Content-Range'], 'bytes 3-6/10')
        status, body, headers = self.request('GET', f'/media/{identity}/camera.mp4', headers={'Range': 'bytes=-4'})
        self.assertEqual((status, body, headers['Content-Range']), (206, b'6789', 'bytes 6-9/10'))
        status, body, headers = self.request('GET', f'/media/{identity}/camera.mp4', headers={'Range': 'bytes=10-'})
        self.assertEqual((status, body, headers['Content-Range']), (416, b'', 'bytes */10'))
        status, body, headers = self.request('HEAD', f'/media/{identity}/camera.mp4')
        self.assertEqual((status, body, headers['Content-Length']), (200, b'', '10'))

    def test_control_stale_click_and_immutable_fields_rejected(self):
        identity = 'run-20260917T000000-12345678'
        folder = self.manager.root/identity; folder.mkdir()
        (folder/'state.json').write_text(json.dumps(dict(selection=dict(sequence=10))))
        (folder/'config.json').write_text(json.dumps(validate_config({})))
        self.manager.active = identity
        self.manager.process = MagicMock(); self.manager.process.poll.return_value = None
        self.manager.control = dict(revision=0, operation='run', settings={}, selection=None)
        self.assertEqual(self.request('POST', '/api/control', {'selection': {'sequence': 9, 'track_id': 1}})[0], 400)
        self.assertEqual(self.request('POST', '/api/control', {'settings': {'recording': 'full-research'}})[0], 400)
        self.assertEqual(self.request('POST', '/api/control', {'selection': {'sequence': 10, 'track_id': 1}})[0], 200)
        self.assertEqual(json.loads((folder/'control.json').read_text())['selection']['track_id'], 1)
        with self.assertRaisesRegex(ValueError, 'Run changed'):
            self.manager.command(dict(run_id='run-20260917T000001-12345678', operation='stop'))
        self.assertEqual(self.request('POST', '/api/control', {'operation': 'stop'})[0], 200)
        for mutation in [{'operation': 'pause'}, {'operation': 'run'}, {'settings': {'target_speed': .1}},
                         {'selection': {'clear': True}}]:
            self.assertEqual(self.request('POST', '/api/control', mutation)[0], 409)
        self.assertEqual(json.loads((folder/'control.json').read_text())['operation'], 'stop')
        self.assertEqual(self.request('POST', '/api/control', {'operation': 'stop'})[0], 200)


if __name__ == '__main__':
    unittest.main()
