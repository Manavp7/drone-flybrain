"""CLI receipts must distinguish successful empty scenes from failed inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from perception.__main__ import main, offline


class OfflineReceiptTests(unittest.TestCase):
    def run_image(self, status, executed):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'source.jpg'
            source.write_bytes(b'input bytes for receipt hash')
            args = SimpleNamespace(command='image', input=source, output=root/'run',
                                   model=Path('yolox_s.onnx'), sha256='a'*64,
                                   input_size=640, output_format='raw_yolox',
                                   confidence=.5, max_age_s=30., tile_size=960)
            detector = SimpleNamespace(inference_count=7 if executed else 0)
            result = {'status':status, 'inference_executed':executed, 'detections':[]}
            fake_cv = SimpleNamespace(IMREAD_COLOR=1,
                                      imread=lambda *a: np.zeros((20,20,3), np.uint8),
                                      imwrite=lambda *a: True)
            pipeline = SimpleNamespace(process=lambda sample: result.copy())
            with patch.dict('sys.modules', {'cv2':fake_cv}), \
                 patch('perception.__main__.make_detector', return_value=detector), \
                 patch('perception.__main__.PerceptionPipeline', return_value=pipeline), \
                 patch('perception.__main__.annotate', side_effect=lambda rgb, r: rgb), \
                 contextlib.redirect_stdout(io.StringIO()):
                code = offline(args)
            return code, json.loads((args.output/'summary.json').read_text())

    def test_valid_empty_scene_is_success_and_records_actual_settings(self):
        code, receipt = self.run_image('ok', True)
        self.assertEqual(code, 0)
        self.assertEqual(receipt['status'], 'completed')
        self.assertEqual(receipt['retained_detections'], 0)
        self.assertEqual(receipt['detector_forward_passes'], 7)
        self.assertEqual(receipt['confidence_threshold'], .5)
        self.assertEqual(receipt['model_filename'], 'yolox_s.onnx')
        self.assertEqual(receipt['tile_size'], 960)
        self.assertTrue(receipt['appearance_tracking'])

    def test_detector_failure_cannot_be_reported_as_completed_success(self):
        code, receipt = self.run_image('detector_error', False)
        self.assertEqual(code, 2)
        self.assertEqual(receipt['status'], 'completed_with_errors')
        self.assertEqual(receipt['frame_status_counts'], {'detector_error':1})
        self.assertFalse(receipt['actual_object_detection_run'])

    def test_missed_deadline_is_error_even_when_forward_pass_executed(self):
        code, receipt = self.run_image('rejected', True)
        self.assertEqual(code, 2)
        self.assertEqual(receipt['status'], 'completed_with_errors')
        self.assertEqual(receipt['inference_calls_completed'], 1)

    def test_cli_default_and_explicit_legacy_threshold(self):
        base = ['image','--input','frame.jpg','--model','model.onnx',
                '--sha256','a'*64,'--output','unused']
        with patch('perception.__main__.offline', return_value=0) as call:
            self.assertEqual(main(base), 0)
            self.assertEqual(call.call_args.args[0].confidence, .5)
            self.assertEqual(main(base+['--confidence','.35','--tile-size','960','--no-appearance-tracking']), 0)
            self.assertEqual(call.call_args.args[0].confidence, .35)
            self.assertEqual(call.call_args.args[0].tile_size, 960)
            self.assertTrue(call.call_args.args[0].no_appearance_tracking)


if __name__ == '__main__':
    unittest.main()
