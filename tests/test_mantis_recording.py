"""Recording should stay bounded, causal and independent of past experiments."""
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from experiments.mantis_recording import SessionRecorder


class MemoryEncoder:
    instances = []

    def __init__(self, path, shape, fps, budget, executable):
        self.frames = 0
        self.error = None
        self.images = []
        self.closed = False
        self.path = path
        self.instances.append(self)

    def append(self, image):
        self.images.append(image.copy())
        self.frames += 1

    def close(self):
        self.closed = True


def image(value=50, shape=(32, 34, 3)):
    return np.full(shape, value, dtype=np.uint8)


def tree_bytes(folder):
    return sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())


class SessionRecordingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        MemoryEncoder.instances = []

    def recorder(self, **kwargs):
        return SessionRecorder(self.root/'run', **kwargs)

    def test_validates_settings_without_creating_output(self):
        for config in (dict(profile='research'), dict(max_bytes=True), dict(max_bytes=10),
                       dict(max_bytes=131072.), dict(fps=True), dict(fps=0),
                       dict(fps=float('nan')), dict(fps=float('inf')),
                       dict(provenance={'bad': float('nan')}), dict(provenance=False)):
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.recorder(**config)
        self.assertFalse((self.root/'run').exists())

    def test_existing_run_and_symlinks_are_preserved(self):
        saved = self.root/'run'
        saved.mkdir()
        (saved/'unique-evidence.bin').write_bytes(b'old raw experiment')
        with self.assertRaises(FileExistsError):
            self.recorder()
        self.assertEqual((saved/'unique-evidence.bin').read_bytes(), b'old raw experiment')
        alias = self.root/'alias'
        alias.symlink_to(saved, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlinks'):
            SessionRecorder(alias/'child')
        self.assertFalse((saved/'child').exists())
        with self.assertRaisesRegex(ValueError, 'traversal'):
            SessionRecorder(self.root/'missing'/'..'/'escaped')

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_timestamp_hold_never_displays_future_image(self):
        recorder = self.recorder()
        recorder.append(.25, image(50), image(90), {'target': 1})
        recorder.append(.45, image(150), image(190), {'target': 2})
        receipt = recorder.finish({'duration_s': .7})
        self.assertEqual([int(frame[0, 0, 0]) for frame in MemoryEncoder.instances[0].images],
                         [0, 0, 0, 50, 50, 150, 150])
        self.assertEqual(receipt['videos']['camera']['duration_s'], .7)
        self.assertTrue(all(encoder.closed for encoder in MemoryEncoder.instances))
        rows = [json.loads(line) for line in (recorder.folder/'telemetry.jsonl').read_text().splitlines()]
        self.assertEqual([row['sim_time_s'] for row in rows], [.25, .45])
        self.assertEqual([row['sequence'] for row in rows], [0, 1])

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_explicit_end_does_not_add_future_interval(self):
        recorder = self.recorder()
        recorder.append(0., image(20), image(), {})
        recorder.append(.1, image(70), image(), {})
        recorder.append(.2, image(100), image(), {})
        receipt = recorder.finish({'duration_s': .2})
        self.assertEqual([int(frame[0, 0, 0]) for frame in MemoryEncoder.instances[0].images], [20, 70])
        self.assertEqual(receipt['videos']['camera']['duration_s'], .2)
        self.assertEqual(recorder.finish({'ignored': 'second finalization'}), receipt)
        self.assertEqual(receipt['bytes'], tree_bytes(recorder.folder))
        with self.assertRaisesRegex(RuntimeError, 'finalized'):
            recorder.append(.3, image(), image(), {})

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_compact_discards_raw_but_research_captures_it(self):
        compact = self.recorder()
        compact.append(0., image(), image(), {'features': [.1, .2]}, raw={'bad': object()})
        compact.finish()
        self.assertFalse((compact.folder/'raw').exists())
        research = SessionRecorder(self.root/'research', profile='full-research')
        research.append(0., image(), image(100), {}, raw={'neural_activity': np.arange(20.)})
        receipt = research.finish()
        self.assertEqual(receipt['raw_frames'], 1)
        self.assertTrue(receipt['raw_capture_complete'])
        with np.load(research.folder/'raw'/'000000.npz', allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['neural_activity'], np.arange(20.))
            np.testing.assert_array_equal(saved['rgb'], image())
            np.testing.assert_array_equal(saved['overview'], image(100))
        self.assertEqual(json.loads((research.folder/'provenance.json').read_text())['profile'], 'full-research')
        self.assertIn('Cesium', (research.folder/'CREDITS.txt').read_text())
        self.assertEqual(receipt['bytes'], tree_bytes(research.folder))

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_nonfinite_or_repeated_time_closes_resources(self):
        for index, bad in enumerate((float('nan'), float('inf'), True, 0., -.1)):
            with self.subTest(time=bad):
                recorder = SessionRecorder(self.root/f'run{index}')
                recorder.append(0., image(), image(), {})
                with self.assertRaises(ValueError):
                    recorder.append(bad, image(), image(), {})
                self.assertTrue(recorder.finished)
                self.assertEqual(recorder.receipt['status'], 'error')
                self.assertTrue(all(encoder.closed for encoder in recorder.encoders.values()))

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_invalid_frame_and_json_never_leave_encoders_running(self):
        for index, values in enumerate(((image().astype(float), image(), {}),
                                       (image(), image(), {'bad': float('nan')}),
                                       (image(), image(), ['not', 'an', 'object']))):
            with self.subTest(index=index):
                recorder = SessionRecorder(self.root/f'invalid{index}')
                recorder.append(0., image(), image(), {})
                with self.assertRaises(ValueError):
                    recorder.append(.1, *values)
                self.assertTrue(recorder.finished)
                self.assertEqual(recorder.receipt['status'], 'error')

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_invalid_summary_finalizes_error_receipt(self):
        recorder = self.recorder()
        recorder.append(.4, image(), image(), {})
        with self.assertRaisesRegex(ValueError, 'precede'):
            recorder.finish({'duration_s': .1})
        self.assertTrue(recorder.finished)
        self.assertEqual(recorder.receipt['status'], 'error')
        self.assertTrue(all(encoder.closed for encoder in recorder.encoders.values()))

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_quota_covers_telemetry_and_finalization_without_deleting_old_runs(self):
        old = self.root/'historical'
        old.mkdir()
        (old/'frozen.npz').write_bytes(b'preserve this')
        recorder = self.recorder(max_bytes=128*1024)
        for step in range(50):
            if not recorder.append(step/10, image(), image(), {'detail': 'x'*8000}):
                break
        receipt = recorder.finish({'duration_s': step/10, 'status': 'stopped'})
        self.assertEqual(receipt['status'], 'budget-exhausted')
        self.assertFalse(receipt['complete'])
        self.assertFalse(receipt['raw_capture_complete'])
        self.assertLessEqual(tree_bytes(recorder.folder), 128*1024)
        self.assertEqual(receipt['bytes'], tree_bytes(recorder.folder))
        self.assertEqual((old/'frozen.npz').read_bytes(), b'preserve this')
        self.assertTrue((recorder.folder/'summary.json').is_file())

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_research_raw_capture_is_also_budgeted(self):
        recorder = self.recorder(profile='full-research', max_bytes=128*1024)
        rng = np.random.default_rng(3)
        noise = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        for step in range(20):
            if not recorder.append(step/10, noise, noise, {}, raw={'neural': rng.normal(size=3000)}):
                break
        receipt = recorder.finish()
        self.assertEqual(receipt['status'], 'budget-exhausted')
        self.assertFalse(receipt['raw_capture_complete'])
        self.assertLessEqual(tree_bytes(recorder.folder), 128*1024)
        self.assertEqual(receipt['bytes'], tree_bytes(recorder.folder))
        for path in (recorder.folder/'raw').glob('*.npz'):
            with np.load(path, allow_pickle=False) as saved:
                self.assertIn('neural', saved.files)

    @patch('experiments.mantis_recording._Encoder', MemoryEncoder)
    def test_context_exit_marks_interrupted_and_records_exception(self):
        with self.recorder() as recorder:
            recorder.append(0., image(), image(), {})
        self.assertEqual(recorder.receipt['status'], 'interrupted')
        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
            with SessionRecorder(self.root/'error') as failed:
                failed.append(0., image(), image(), {})
                raise RuntimeError('caller failed')
        self.assertEqual(failed.receipt['status'], 'error')
        self.assertIn('caller failed', failed.receipt['error'])

    def test_error_summary_never_claims_completed(self):
        recorder = self.recorder()
        receipt = recorder.finish({'status': 'error', 'reason': 'caller failed before first capture'})
        self.assertEqual(receipt['status'], 'error')
        self.assertFalse(receipt['complete'])
        self.assertEqual(receipt['bytes'], tree_bytes(recorder.folder))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe required')
    def test_real_h264_video_is_playable_and_records_causal_frames(self):
        recorder = self.recorder()
        recorder.append(.2, image(50), image(90), {'position': [0, 0, 1]})
        recorder.append(.4, image(150), image(190), {'position': [0, .1, 1]})
        receipt = recorder.finish({'duration_s': .6})
        self.assertEqual(receipt['status'], 'completed')
        for name in ('camera', 'overview'):
            path = recorder.folder/f'{name}.mp4'
            info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
                                                      '-show_format', '-of', 'json', str(path)]))
            self.assertEqual(info['streams'][0]['codec_name'], 'h264')
            self.assertAlmostEqual(float(info['format']['duration']), .6, places=4)
            decoded = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', str(path),
                                              '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'])
            frames = np.frombuffer(decoded, np.uint8).reshape((-1, 32, 34, 3))
            self.assertEqual(len(frames), 6)
            expected = [0, 0, 50, 50, 150, 150] if name == 'camera' else [0, 0, 90, 90, 190, 190]
            np.testing.assert_allclose(frames.mean(axis=(1, 2, 3)), expected, atol=3)
        self.assertEqual(receipt['bytes'], tree_bytes(recorder.folder))

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'ffmpeg and ffprobe required')
    def test_real_encoder_budget_retains_decodable_complete_fragments(self):
        recorder = self.recorder(max_bytes=256*1024)
        rng = np.random.default_rng(5)
        for step in range(150):
            rgb = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
            if not recorder.append(step/10, rgb, rgb, {'step': step}):
                break
        receipt = recorder.finish({'duration_s': step/10})
        self.assertEqual(receipt['status'], 'budget-exhausted')
        self.assertLessEqual(tree_bytes(recorder.folder), 256*1024)
        self.assertEqual(receipt['bytes'], tree_bytes(recorder.folder))
        self.assertTrue(receipt['videos'])
        for video in receipt['videos'].values():
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i',
                            str(recorder.folder/video['path']), '-f', 'null', '-'], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertGreater(video['frames'], 0)


if __name__ == '__main__':
    unittest.main()
