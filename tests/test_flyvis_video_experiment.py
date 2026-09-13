"""Synthetic numerical/video fixtures test contracts, never neural accuracy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.video_experiment import aggregate_activity, decode_clip, source_timestamps, validate_options


class InputContracts(unittest.TestCase):
    def test_timestamp_corruption_rejected(self):
        for value in ([], [0], [0, 0], [1, 0], [-.1, 0], [0, float("nan")],
                      [0, float("inf")], [0, True], [0, "0.1"], {"pts": [0, 1]}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                source_timestamps(value)

    def test_irregular_valid_timestamps_preserved(self):
        pts = source_timestamps([0., .031, .067, .125])
        np.testing.assert_array_equal(pts, [0., .031, .067, .125])
        self.assertEqual(pts.dtype, np.float64)

    def test_resource_and_timing_bounds(self):
        base = dict(start_s=0., duration_s=2., sample_fps=15., max_width=256, dt=.02)
        validate_options(**base)
        for key, value in (("start_s", -1), ("duration_s", 11), ("duration_s", float("nan")),
                           ("sample_fps", 0), ("max_width", 0), ("max_width", True), ("dt", .1)):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_options(**(base | {key: value}))


class AggregateContracts(unittest.TestCase):
    def test_neuron_labels_group_correct_columns(self):
        activity = np.array([[1., 8., 3.], [5., 6., 9.]], dtype=np.float32)
        labels, actual = aggregate_activity(activity, np.array([b"T4", b"T5", b"T4"]), [.02, .04])
        np.testing.assert_array_equal(labels, ["T4", "T5", "T4"])
        np.testing.assert_array_equal(actual["neuron_count"], [2, 1])
        np.testing.assert_array_equal(actual["mean"], [[2, 8], [7, 6]])
        np.testing.assert_array_equal(actual["std"], [[1, 0], [2, 0]])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "activity.npz"
            np.savez_compressed(path, **actual)
            with np.load(path, allow_pickle=False) as reopened:
                np.testing.assert_array_equal(reopened["cell_types"], ["T4", "T5"])

    def test_mismatched_or_nonfinite_outputs_rejected(self):
        values = np.ones((2, 3), dtype=np.float32)
        for activity, labels, times in ((values, ["T4"], [.02, .04]),
                                      (values, ["T4"] * 3, [.02]),
                                      (values, ["T4"] * 3, [.02, .02]),
                                      (values * np.nan, ["T4"] * 3, [.02, .04]),
                                      (values, [1, 2, 3], [.02, .04])):
            with self.assertRaises(ValueError):
                aggregate_activity(activity, labels, times)


class DecodeTimingContracts(unittest.TestCase):
    def fake_capture(self, pts):
        import cv2
        class Capture:
            index = -1
            released = False
            def isOpened(self): return True
            def read(self):
                self.index += 1
                return (True, np.full((12, 24, 3), self.index, np.uint8)) if self.index < len(pts) else (False, None)
            def get(self, prop):
                return 30. if prop == cv2.CAP_PROP_FPS else pts[self.index] * 1000
            def release(self): self.released = True
        return Capture()

    def test_vfr_timing_and_aspect_preserved(self):
        capture = self.fake_capture([0., .03, .072, .1, .14, .17, .21])
        with patch("cv2.VideoCapture", return_value=capture):
            clip, times, metadata = decode_clip(Path("fixture"), duration_s=.2, sample_fps=15., max_width=12)
        np.testing.assert_allclose(times, [0., .072, .14])
        self.assertEqual(clip.shape, (3, 6, 12))
        self.assertEqual(clip.dtype, np.float32)
        self.assertEqual(metadata["selected_source_indices"], [0, 2, 4])
        self.assertTrue(capture.released)

    def test_unusable_opencv_clock_never_replaced_by_nominal_fps(self):
        capture = self.fake_capture([0., 0., 0.])
        with patch("cv2.VideoCapture", return_value=capture), self.assertRaisesRegex(ValueError, "PTS"):
            decode_clip(Path("fixture"))
        self.assertTrue(capture.released)

    def test_external_pts_count_mismatch_rejected(self):
        capture = self.fake_capture([0., .1, .2])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pts.json"
            path.write_text(json.dumps([0., .1, .2, .3]))
            with patch("cv2.VideoCapture", return_value=capture), self.assertRaisesRegex(ValueError, "count"):
                decode_clip(Path("fixture"), timestamps_path=path)


if __name__ == "__main__":
    unittest.main()
