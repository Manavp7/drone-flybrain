"""Recorded replay must not turn future or unavailable evidence into live data."""
from copy import deepcopy
import json
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mantis_report import (build_timeline, contained_path, latest_available,
    load_recordings, normalize_comparisons, render_frame, script_json, validate_records,
    export_report, ReplayRequestHandler, ACTOR_CREDIT, ACTOR_SOURCE, ACTOR_LICENSE)


def fixture():
    spec = dict(name='walk', kind='normal', duration_s=.5, method='mantis_neural')

    def state(t):
        return dict(time_s=t, position=[t, 0., 1.1], velocity=[.2, 0., 0.], motor_forces=[1.9]*4)

    ticks = [dict(time_s=i*.1, state_before=state(i*.1), state_after=state((i+1)*.1),
                  request=dict(forward_speed=.2, reason='neural_bearing'),
                  guardian=dict(forward_speed=.2, reason='clear'), applied_command_sequence=0,
                  truth_after=dict(contact_count=0)) for i in range(5)]
    rows = [dict(sequence=0, capture_time_s=0., completed_time_s=.2,
                 overview_capture_time_s=0., inference_wall_s=.15,
                 observation=dict(valid=True, bbox_xyxy=[2., 2., 8., 9.], track_id=1),
                 candidate=dict(valid=True), frame_file='frames/000000.npz',
                 neural_features=[.1, -.2, .3, -.4, .5, -.6, .7, -.8]),
            dict(sequence=1, capture_time_s=.2, completed_time_s=.6,
                 overview_capture_time_s=.2, inference_wall_s=.4,
                 observation=dict(valid=True, bbox_xyxy=[2., 2., 8., 9.], track_id=9),
                 candidate=None, frame_file='frames/000001.npz',
                 neural_features=[99.]*8, discarded_after_episode=True)]
    episode = dict(status='completed', physics_ticks=5, observations=2, whole_wall_s=1.)
    return spec, ticks, rows, episode


def save_fixture(root, methods=('mantis_neural',)):
    spec, ticks, rows, episode = fixture()
    plan = {key: value for key, value in spec.items() if key != 'method'}
    (root/'definition.json').write_text(json.dumps(dict(specs=[plan], methods=list(methods))))
    for index, method in enumerate(methods):
        folder = root/f'walk__{method}'
        (folder/'frames').mkdir(parents=True)
        saved = dict(spec, method=method)
        (folder/'spec.json').write_text(json.dumps(saved))
        (folder/'episode.json').write_text(json.dumps(episode))
        (folder/'ticks.jsonl').write_text('\n'.join(json.dumps(row) for row in ticks))
        own_rows = deepcopy(rows)
        own_rows[0]['observation']['track_id'] = index+1
        (folder/'observations.jsonl').write_text('\n'.join(json.dumps(row) for row in own_rows))
        for sequence in range(2):
            np.savez_compressed(folder/'frames'/f'{sequence:06d}.npz',
                                rgb=np.full((12, 12, 3), 80+index*30, dtype=np.uint8),
                                overview=np.full((12, 12, 3), 100+index*30, dtype=np.uint8))
    return root


def comparison_fixture():
    methods = ('mantis_neural', 'direct_yolo', 'alpha_beta')
    score = dict(recorded_count=4, truth_available_count=4, truth_unavailable_count=0,
                 shared_eligible_count=3, matched_count=2, shared_missing_reasons={'gap': 1},
                 score_clock='capture_time_s', completion_time_accuracy_evaluated=False,
                 methods={m: dict(matched_capture_rmse_rad=.1*(i+1), truth_prediction_count=i+2,
                                  full_truth_coverage=(i+2)/4, eligible_truth_coverage=min((i+2)/3, 1.))
                          for i, m in enumerate(methods)})
    result = dict(rows=[{}]*4, timing=dict(per_method_wall_s={m: .01*(i+1) for i, m in enumerate(methods)},
                  detector_wall_s=.04, neural_reset_wall_s=.02))
    return dict(case='walk', variant='noise', seed=123, source_method='mantis_neural', score=score, result=result)


class MantisReplayTests(unittest.TestCase):
    def test_future_detection_and_features_not_visible(self):
        spec, ticks, rows, _ = fixture()
        timeline = build_timeline(spec, ticks, rows)
        self.assertIsNone(timeline[1]['sequence'])
        self.assertIsNone(timeline[1]['features'])
        self.assertFalse(timeline[1]['target_valid'])
        self.assertEqual(timeline[2]['sequence'], 0)
        self.assertEqual(timeline[2]['features'], rows[0]['neural_features'])
        self.assertEqual(timeline[2]['capture_time_s'], 0.)
        self.assertEqual(timeline[2]['completed_time_s'], .2)

    def test_discarded_final_observation_never_appears(self):
        spec, ticks, rows, _ = fixture()
        self.assertEqual(latest_available(rows, 100)['sequence'], 0)
        timeline = build_timeline(spec, ticks, rows)
        self.assertEqual(timeline[-1]['target_id'], 1)
        self.assertNotIn(99., timeline[-1]['features'])

    def test_telemetry_continues_while_camera_is_held(self):
        spec, ticks, rows, _ = fixture()
        timeline = build_timeline(spec, ticks, rows)
        self.assertEqual(timeline[4]['sequence'], timeline[2]['sequence'])
        self.assertEqual(timeline[4]['capture_time_s'], 0.)
        self.assertEqual(timeline[4]['state_time_s'], .4)
        self.assertEqual(timeline[4]['position'][0], .4)
        self.assertEqual(timeline[-1]['state_time_s'], .5)

    def test_renderer_does_not_open_future_frame(self):
        _, _, rows, _ = fixture()
        with TemporaryDirectory() as tmp:
            before = render_frame(Path(tmp), rows, .1)
            self.assertEqual(before.shape, (640, 1280, 3))
            with self.assertRaises(FileNotFoundError):
                render_frame(Path(tmp), rows, .2)

    def test_end_telemetry_keeps_last_encoded_frame_evidence(self):
        spec, ticks, rows, _ = fixture()
        rows[1].update(completed_time_s=.5, discarded_after_episode=False)
        timeline = build_timeline(spec, ticks, rows)
        self.assertEqual(timeline[-1]['state_time_s'], .5)
        self.assertEqual(timeline[-1]['sequence'], 0)

    def test_missing_neural_features_remain_absent(self):
        spec, ticks, rows, _ = fixture()
        rows[0]['neural_features'] = None
        self.assertIsNone(build_timeline(spec, ticks, rows)[2]['features'])

    def test_invalid_feature_shape_rejected(self):
        spec, ticks, rows, _ = fixture()
        rows[0]['neural_features'] = [0.]*7
        with self.assertRaises(ValueError):
            build_timeline(spec, ticks, rows)

    def test_receipt_and_clock_gaps_rejected(self):
        spec, ticks, rows, episode = fixture()
        validate_records(spec, ticks, rows, episode)
        episode['observations'] = 1
        with self.assertRaisesRegex(ValueError, 'receipt'):
            validate_records(spec, ticks, rows, episode)
        episode['observations'] = 2
        rows[1]['discarded_after_episode'] = False
        with self.assertRaisesRegex(ValueError, 'discarded'):
            validate_records(spec, ticks, rows, episode)

    def test_future_overview_rejected(self):
        spec, ticks, rows, episode = fixture()
        rows[0]['overview_capture_time_s'] = .3
        with self.assertRaisesRegex(ValueError, 'Overview'):
            validate_records(spec, ticks, rows, episode)

    def test_script_end_tags_are_inert_and_round_trip(self):
        source = {'name': '</script><script>alert(1)</script>&\u2028\u2029'}
        encoded = script_json(source)
        self.assertNotIn('<', encoded)
        self.assertNotIn('&', encoded)
        self.assertEqual(json.loads(encoded), source)
        with self.assertRaises(ValueError):
            script_json({'bad': float('nan')})

    def test_frame_cannot_escape_episode_by_path_or_symlink(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root/'episode'
            folder.mkdir()
            (folder/'escape').symlink_to(root, target_is_directory=True)
            for name in ('../outside.npz', '/tmp/outside.npz', 'escape/outside.npz'):
                with self.subTest(path=name), self.assertRaises(ValueError):
                    contained_path(folder, name)
            self.assertEqual(contained_path(folder, 'frames/000000.npz'), (folder/'frames/000000.npz').resolve())

    def test_each_controller_selects_its_own_recording(self):
        with TemporaryDirectory() as tmp:
            root = save_fixture(Path(tmp), ('mantis_neural', 'direct_yolo', 'alpha_beta'))
            payload, recordings = load_recordings(root)
            self.assertEqual(len(recordings), 3)
            saved = payload['cases'][0]['methods']
            self.assertEqual(saved['mantis_neural']['timeline'][2]['target_id'], 1)
            self.assertEqual(saved['direct_yolo']['timeline'][2]['target_id'], 2)
            self.assertEqual(saved['alpha_beta']['video'], 'walk__alpha_beta.mp4')
            self.assertFalse(payload['real_time'])

    def test_missing_planned_controller_refuses_partial_export(self):
        with TemporaryDirectory() as tmp:
            root = save_fixture(Path(tmp))
            definition = json.loads((root/'definition.json').read_text())
            definition['methods'].append('direct_yolo')
            (root/'definition.json').write_text(json.dumps(definition))
            with self.assertRaises(FileNotFoundError):
                load_recordings(root)

    def test_mislabelled_controller_refused(self):
        with TemporaryDirectory() as tmp:
            root = save_fixture(Path(tmp))
            path = root/'walk__mantis_neural/spec.json'
            spec = json.loads(path.read_text())
            spec['method'] = 'direct_yolo'
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, 'controller'):
                load_recordings(root)

    def test_export_preserves_actor_credit_and_offline_copy(self):
        with TemporaryDirectory() as tmp:
            root = save_fixture(Path(tmp))
            with patch('experiments.mantis_report.encode_video', return_value={'file': 'fixture.mp4'}):
                page = export_report(root)
            self.assertEqual((page.parent/'CREDITS.txt').read_text(), ACTOR_CREDIT)
            html = page.read_text()
            for value in (ACTOR_SOURCE, ACTOR_LICENSE, 'Cesium Man © 2017 Cesium', 'CREDITS.txt',
                          'Z-up', '1.8 m', '19 animated joints', 'plain clothing materials'):
                self.assertIn(value, html)
            self.assertIn('brown skin and dark shoes', ACTOR_CREDIT)

    def test_existing_export_is_never_overwritten(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'report').mkdir()
            marker = root/'report/index.html'
            marker.write_text('existing user report')
            with self.assertRaises(FileExistsError):
                export_report(root)
            self.assertEqual(marker.read_text(), 'existing user report')

    def test_comparison_preserves_matched_subset_and_missing_coverage(self):
        normalized = normalize_comparisons([comparison_fixture()])[0]
        self.assertEqual(normalized['matched_count'], 2)
        self.assertEqual(normalized['truth_available_count'], 4)
        self.assertEqual(normalized['source_label'], 'Mantis Neural')
        neural = normalized['metrics'][0]
        self.assertEqual(neural['full_truth_coverage'], .5)
        self.assertAlmostEqual(neural['rmse_deg'], np.rad2deg(.1))
        self.assertEqual(neural['estimator_ms_per_observation'], 2.5)
        self.assertEqual(normalized['detector_ms_per_observation'], 10.)
        self.assertFalse(normalized['completion_time_accuracy_evaluated'])

    def test_empty_matched_subset_is_not_zero_error(self):
        entry = comparison_fixture()
        entry['score']['matched_count'] = 0
        for score in entry['score']['methods'].values():
            score['matched_capture_rmse_rad'] = None
        normalized = normalize_comparisons([entry])[0]
        self.assertIsNone(normalized['metrics'][0]['rmse_deg'])
        entry['score']['recorded_count'] = 5
        with self.assertRaises(ValueError):
            normalize_comparisons([entry])


class RangeServerTests(unittest.TestCase):
    def request(self, directory, range_header=None, method='GET'):
        class Connection:
            def __init__(self, request):
                self.request = BytesIO(request)
                self.output = BytesIO()
            def makefile(self, *args, **kwargs):
                return self.request
            def sendall(self, data):
                self.output.write(data)

        class QuietHandler(ReplayRequestHandler):
            def log_message(self, *args):
                pass

        header = '' if range_header is None else f'Range: {range_header}\r\n'
        connection = Connection(f'{method} /clip.mp4 HTTP/1.0\r\nHost: localhost\r\n{header}\r\n'.encode())
        QuietHandler(connection, ('127.0.0.1', 0), object(), directory=directory)
        headers, body = connection.output.getvalue().split(b'\r\n\r\n', 1)
        lines = headers.decode().split('\r\n')
        return int(lines[0].split()[1]), dict(line.split(': ', 1) for line in lines[1:]), body

    def test_full_and_partial_media_responses_allow_seeking(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp)/'clip.mp4').write_bytes(b'0123456789')
            status, headers, body = self.request(tmp)
            self.assertEqual((status, body), (200, b'0123456789'))
            self.assertEqual(headers['Accept-Ranges'], 'bytes')
            status, headers, body = self.request(tmp, 'bytes=2-5')
            self.assertEqual((status, body), (206, b'2345'))
            self.assertEqual(headers['Content-Range'], 'bytes 2-5/10')
            self.assertEqual(headers['Content-Length'], '4')

    def test_suffix_open_range_and_head(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp)/'clip.mp4').write_bytes(b'0123456789')
            self.assertEqual(self.request(tmp, 'bytes=-3')[2], b'789')
            self.assertEqual(self.request(tmp, 'bytes=7-')[2], b'789')
            self.assertEqual(self.request(tmp, 'bytes=8-99')[2], b'89')
            status, headers, body = self.request(tmp, 'bytes=2-5', 'HEAD')
            self.assertEqual((status, body), (206, b''))
            self.assertEqual(headers['Content-Length'], '4')

    def test_unsatisfiable_ranges_do_not_return_wrong_bytes(self):
        with TemporaryDirectory() as tmp:
            (Path(tmp)/'clip.mp4').write_bytes(b'0123456789')
            for value in ('bytes=10-', 'bytes=7-2', 'bytes=-0', 'bytes=0-1,4-5', 'nonsense'):
                with self.subTest(range=value):
                    status, headers, body = self.request(tmp, value)
                    self.assertEqual((status, body), (416, b''))
                    self.assertEqual(headers['Content-Range'], 'bytes */10')


if __name__ == '__main__':
    unittest.main()
