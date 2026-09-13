"""Clothing-gate behavior and pipeline integration; no identity-accuracy claim."""
import unittest
from unittest.mock import patch

import numpy as np

from perception.detector import Detection
from perception.pipeline import CameraSample, PerceptionPipeline, ShortTermTracker, clothing_histogram


def person(box=(10., 10., 90., 90.), confidence=.9, class_id=0, label="person"):
    return Detection(box, confidence, class_id, label)


def solid(color, shape=(100, 150, 3)):
    frame = np.empty(shape, np.uint8)
    frame[:] = color
    return frame


class AppearanceTrackerTests(unittest.TestCase):
    def test_crossing_different_clothing_keeps_ids_instead_of_following_position(self):
        left, right = (0., 0., 100., 100.), (50., 0., 150., 100.)
        def crossing_frame(left_color, right_color):
            frame = solid((80, 80, 80))
            frame[25:60, 20:70] = left_color
            frame[25:60, 80:130] = right_color
            return frame
        red, blue = (255, 0, 0), (0, 0, 255)
        old, new = [person(left, .9), person(right, .8)], [person(right, .9), person(left, .8)]
        tracker = ShortTermTracker()
        first = tracker.update(old, 0., crossing_frame(red, blue))
        second = tracker.update(new, .1, crossing_frame(blue, red))
        self.assertEqual([item["track_id"] for item in first], [1, 2])
        self.assertEqual([item["track_id"] for item in second], [1, 2])
        self.assertEqual([item["bbox_xyxy"] for item in second], [list(right), list(left)])
        legacy = ShortTermTracker(appearance_threshold=None)
        legacy.update(old, 0.)
        self.assertEqual([item["track_id"] for item in legacy.update(new, .1)], [2, 1])

    def test_same_clothing_motion_retains_real_box_and_score(self):
        tracker = ShortTermTracker()
        image = solid((230, 30, 160))
        old = tracker.update([person()], 0., image)[0]
        new_box = (15., 12., 95., 92.)
        new = tracker.update([person(new_box, .7)], .1, image)[0]
        self.assertEqual(new["track_id"], old["track_id"])
        self.assertEqual(new["bbox_xyxy"], list(new_box))
        self.assertEqual(new["confidence"], .7)
        self.assertEqual(new["label"], "person")

    def test_disjoint_clothing_prevents_same_box_reassignment(self):
        tracker = ShortTermTracker()
        first = tracker.update([person()], 0., solid((255, 0, 0)))[0]
        second = tracker.update([person()], .1, solid((0, 0, 255)))[0]
        self.assertNotEqual(first["track_id"], second["track_id"])

    def test_missing_observation_emits_nothing_and_expiry_never_recycles(self):
        tracker = ShortTermTracker()
        image = solid((255, 0, 0))
        self.assertEqual(tracker.update([person()], 0., image)[0]["track_id"], 1)
        self.assertEqual(tracker.update([], .2, image), [])
        self.assertEqual(tracker.update([person()], .3, image)[0]["track_id"], 1)
        self.assertEqual(tracker.update([], .81, image), [])
        self.assertEqual(tracker.tracks, {})
        self.assertEqual(tracker.update([person()], .9, image)[0]["track_id"], 2)

    def test_reset_clears_histograms_but_keeps_monotonic_ids(self):
        tracker = ShortTermTracker()
        tracker.update([person()], 0., solid((255, 0, 0)))
        self.assertEqual(tracker.tracks[1]["appearance"].shape, (64,))
        tracker.reset()
        self.assertEqual(tracker.tracks, {})
        self.assertIsNone(tracker.last_time)
        self.assertEqual(tracker.update([person()], 0., solid((0, 0, 255)))[0]["track_id"], 2)

    def test_nonperson_color_change_keeps_iou_behavior_and_different_class_cannot_match(self):
        tracker = ShortTermTracker()
        chair = person(class_id=56, label="chair")
        first = tracker.update([chair], 0., solid((255, 0, 0)))[0]
        second = tracker.update([chair], .1, solid((0, 0, 255)))[0]
        self.assertEqual(first["track_id"], second["track_id"])
        self.assertIsNone(tracker.tracks[first["track_id"]]["appearance"])
        third = tracker.update([person()], .2, solid((0, 0, 255)))[0]
        self.assertNotEqual(third["track_id"], second["track_id"])

    def test_missing_rgb_uses_legacy_path_then_cannot_bridge_stale_appearance(self):
        tracker = ShortTermTracker()
        ids = [tracker.update([person()], 0., solid((255, 0, 0)))[0]["track_id"],
               tracker.update([person()], .1)[0]["track_id"],
               tracker.update([person()], .2, solid((0, 0, 255)))[0]["track_id"]]
        self.assertEqual(ids, [1, 1, 2])
        legacy = ShortTermTracker()
        self.assertEqual([legacy.update([person()], time)[0]["track_id"] for time in (0., .1)], [1, 1])

    def test_disabled_threshold_ignores_clothing_even_with_rgb(self):
        tracker = ShortTermTracker(appearance_threshold=None)
        self.assertEqual(tracker.update([person()], 0., solid((255, 0, 0)))[0]["track_id"], 1)
        self.assertEqual(tracker.update([person()], .1, solid((0, 0, 255)))[0]["track_id"], 1)

    def test_tiny_and_image_edge_boxes_produce_finite_bounded_features(self):
        for shape, box in (((1, 1, 3), (0., 0., 1., 1.)), ((100, 150, 3), (0., 0., .01, .01)),
                           ((100, 150, 3), (140., 80., 150., 100.))):
            with self.subTest(shape=shape, box=box):
                image = solid((0, 255, 0), shape)
                histogram = clothing_histogram(image, box)
                self.assertEqual(histogram.shape, (64,))
                self.assertTrue(np.isfinite(histogram).all())
                self.assertAlmostEqual(histogram.sum(), 1.)
                tracker = ShortTermTracker()
                self.assertEqual(tracker.update([person(box)], 0., image)[0]["track_id"], 1)
                self.assertEqual(tracker.update([person(box)], .1, image)[0]["track_id"], 1)

    def test_feature_pixel_work_and_active_histograms_are_bounded(self):
        image = solid((50, 60, 70), (500, 1000, 3))
        real_bincount = np.bincount
        sampled = []
        def checked_bincount(values, **kwargs):
            sampled.append(values.size)
            return real_bincount(values, **kwargs)
        tracker = ShortTermTracker(max_tracks=2)
        detections = [person((0., 0., 1000., 500.), .9 - i * .1) for i in range(3)]
        with patch("perception.pipeline.np.bincount", side_effect=checked_bincount):
            tracker.update(detections, 0., image)
        self.assertEqual(sampled, [1024, 1024])
        self.assertEqual(len(tracker.tracks), 2)
        self.assertEqual(sum(old["appearance"].nbytes for old in tracker.tracks.values()), 2 * 64 * 8)
        for old in tracker.tracks.values():
            self.assertEqual(set(old), {"bbox", "class_id", "time", "appearance"})

    def test_invalid_rgb_or_box_fails_before_mutating_track_state(self):
        tracker = ShortTermTracker()
        tracker.update([person()], 0., solid((255, 0, 0)))
        invalid_images = ["missing", np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32),
                          np.zeros((0, 10, 3), np.uint8), np.zeros((4097, 4097, 3), np.uint8)]
        for image in invalid_images:
            with self.subTest(image_type=type(image)), self.assertRaises(ValueError):
                tracker.update([person()], .1, image)
            self.assertEqual(tracker.last_time, 0.)
        for item in (person((-1., 0., 30., 30.)), person((0., 0., 200., 30.)),
                     person((0., 0., float("nan"), 30.)), "wrong type"):
            with self.subTest(item=item), self.assertRaises(ValueError):
                tracker.update([item], .1, solid((255, 0, 0)))
            self.assertEqual(tracker.last_time, 0.)
        with self.assertRaises(ValueError):
            tracker.update([person()] * 1001, .1, solid((255, 0, 0)))

    def test_invalid_appearance_threshold_rejected(self):
        for value in (True, np.bool_(False), 0, -.1, 1.1, float("nan"), float("inf"), ".65", 1j):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ShortTermTracker(appearance_threshold=value)


class AppearancePipelineTests(unittest.TestCase):
    class Clock:
        now = 0.
        def __call__(self):
            return self.now

    class Detector:
        def detect(self, _):
            return [person()]

    def sample(self, color, sequence, clock):
        return CameraSample(solid(color), float(sequence), clock.now, sequence, "test", "sim", "optical", 0.)

    def test_default_pipeline_uses_rgb_and_optout_retains_geometry(self):
        for enabled, expected in ((True, [1, 2]), (False, [1, 1])):
            with self.subTest(enabled=enabled):
                clock = self.Clock()
                pipeline = PerceptionPipeline(self.Detector(), clock=clock, appearance_tracking=enabled)
                first = pipeline.process(self.sample((255, 0, 0), 0, clock))
                # Capture timestamps remain within the existing 0.5 s expiry.
                second_sample = self.sample((0, 0, 255), 1, clock)
                second_sample = CameraSample(**{**vars(second_sample), "capture_time_s": .1})
                second = pipeline.process(second_sample)
                self.assertEqual([first["detections"][0]["track_id"], second["detections"][0]["track_id"]], expected)
                self.assertEqual(first["status"], "ok")
                self.assertEqual(second["status"], "ok")
                self.assertFalse(second["control_authority"])

    def test_appearance_feature_cost_is_inside_complete_pipeline_deadline(self):
        clock = self.Clock()
        pipeline = PerceptionPipeline(self.Detector(), clock=clock)
        first = pipeline.process(self.sample((255, 0, 0), 0, clock))
        self.assertEqual(first["status"], "ok")
        self.assertTrue(pipeline.tracker.tracks)
        original = clothing_histogram
        def slow_appearance(image, box):
            clock.now += .4
            return original(image, box)
        with patch("perception.pipeline.clothing_histogram", side_effect=slow_appearance):
            result = pipeline.process(self.sample((255, 0, 0), 1, clock))
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["reason"], "processing_deadline_missed")
        self.assertEqual(result["detections"], [])
        self.assertEqual(result["depth_advisory"], {"state": "unknown"})
        self.assertGreaterEqual(result["processing_ms"], 400.)
        self.assertEqual(pipeline.tracker.tracks, {})
        self.assertTrue(result["inference_executed"])

    def test_appearance_pipeline_option_requires_actual_bool(self):
        for value in (0, 1, "false", None, np.bool_(True)):
            with self.subTest(value=value), self.assertRaises(ValueError):
                PerceptionPipeline(self.Detector(), appearance_tracking=value)


if __name__ == "__main__":
    unittest.main()
