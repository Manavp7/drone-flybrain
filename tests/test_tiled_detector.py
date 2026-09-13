"""Crop geometry and failure contracts; fake outputs do not measure accuracy."""
import unittest

import numpy as np

from perception.detector import COCO_CLASSES, Detection, DetectorError, ModelContractError
from perception.pipeline import CameraSample, PerceptionPipeline
from perception.tiled_detector import TiledDetector, _tile_windows


def detection(box=(10., 10., 30., 40.), confidence=.8, class_id=0):
    return Detection(box, confidence, class_id, COCO_CLASSES[class_id])


class ScriptedDetector:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.images = []
        self.inference_count = 0

    def detect(self, image_rgb):
        self.images.append(image_rgb.copy())
        result = self.outputs.pop(0)
        if isinstance(result, Exception):
            raise result
        self.inference_count += 1
        return result


class TileGeometryTests(unittest.TestCase):
    def test_regular_grid_and_last_window_clamped_without_duplicate(self):
        self.assertEqual(_tile_windows(250, 100, 100, 75, 4), [
            (0, 0, 100, 100), (75, 0, 175, 100), (150, 0, 250, 100),
        ])
        self.assertEqual(_tile_windows(270, 100, 100, 75, 4), [
            (0, 0, 100, 100), (75, 0, 175, 100),
            (150, 0, 250, 100), (170, 0, 270, 100),
        ])

    def test_two_dimensional_grid_covers_edges(self):
        self.assertEqual(_tile_windows(170, 170, 100, 75, 4), [
            (0, 0, 100, 100), (70, 0, 170, 100),
            (0, 70, 100, 170), (70, 70, 170, 170),
        ])

    def test_smaller_dimension_is_clipped_and_small_frame_needs_no_crops(self):
        self.assertEqual(_tile_windows(170, 50, 100, 75, 2), [
            (0, 0, 100, 50), (70, 0, 170, 50),
        ])
        self.assertEqual(_tile_windows(100, 50, 100, 75, 1), [])

    def test_tile_budget_rejected_before_first_inference(self):
        backend = ScriptedDetector([])
        detector = TiledDetector(backend, tile_size=100, max_tiles=3)
        with self.assertRaisesRegex(ValueError, "requires 4 crops"):
            detector.detect(np.zeros((170, 170, 3), dtype=np.uint8))
        self.assertEqual(backend.images, [])
        self.assertEqual(detector.inference_count, 0)

    def test_extreme_grid_rejected_before_allocating_windows(self):
        # The input shape is allowed, but this grid would contain 16M tiles.
        with self.assertRaisesRegex(ValueError, "exceeding max_tiles"):
            _tile_windows(4096, 4096, 1, 1, 16)


class TiledInferenceTests(unittest.TestCase):
    def test_small_image_runs_full_pass_once_and_retains_global_edge(self):
        item = detection((0., 0., 70., 50.))
        backend = ScriptedDetector([[item]])
        result = TiledDetector(backend, tile_size=100).detect(np.zeros((50, 70, 3), dtype=np.uint8))
        self.assertEqual(result, [item])
        self.assertEqual(backend.inference_count, 1)

    def test_passes_receive_exact_rgb_pixels_and_actual_count_is_proxied(self):
        image = np.arange(100 * 170 * 3, dtype=np.uint32).reshape(100, 170, 3).astype(np.uint8)
        original = image.copy()
        backend = ScriptedDetector([[], [], []])
        wrapped = TiledDetector(backend, tile_size=100)
        self.assertEqual(wrapped.detect(image), [])
        np.testing.assert_array_equal(backend.images[0], image)
        np.testing.assert_array_equal(backend.images[1], image[:, :100])
        np.testing.assert_array_equal(backend.images[2], image[:, 70:])
        np.testing.assert_array_equal(image, original)
        self.assertEqual(wrapped.inference_count, 3)

    def test_internal_seam_fragment_rejected_and_overlap_maps_complete_box(self):
        backend = ScriptedDetector([
            [], [detection((80., 10., 100., 50.))],
            [detection((10., 10., 50., 50.))],
        ])
        result = TiledDetector(backend, tile_size=100).detect(np.zeros((100, 170, 3), dtype=np.uint8))
        self.assertEqual(result, [detection((80., 10., 120., 50.))])

    def test_internal_edges_use_margin_on_all_four_sides(self):
        backend = ScriptedDetector([
            [],
            [detection((10., 10., 97., 50.)), detection((10., 10., 50., 97.))],
            [detection((3., 10., 50., 50.))],
            [detection((10., 3., 50., 50.))],
            [detection((4., 4., 40., 40.))],
        ])
        result = TiledDetector(backend, tile_size=100).detect(np.zeros((170, 170, 3), dtype=np.uint8))
        self.assertEqual(result, [detection((74., 74., 110., 110.))])

    def test_external_image_edges_remain_allowed_for_every_corner(self):
        backend = ScriptedDetector([
            [], [detection((0., 0., 30., 30.))],
            [detection((70., 0., 100., 30.))],
            [detection((0., 70., 30., 100.))],
            [detection((70., 70., 100., 100.))],
        ])
        result = TiledDetector(backend, tile_size=100).detect(np.zeros((170, 170, 3), dtype=np.uint8))
        self.assertEqual([item.bbox_xyxy for item in result], [
            (0., 0., 30., 30.), (140., 0., 170., 30.),
            (0., 140., 30., 170.), (140., 140., 170., 170.),
        ])

    def test_full_frame_large_object_survives_when_every_crop_truncates_it(self):
        original = detection((20., 10., 150., 90.))
        backend = ScriptedDetector([
            [original], [detection((20., 10., 100., 90.))],
            [detection((0., 10., 80., 90.))],
        ])
        result = TiledDetector(backend, tile_size=100).detect(np.zeros((100, 170, 3), dtype=np.uint8))
        self.assertEqual(result, [original])

    def test_duplicate_same_class_uses_best_score_but_other_class_is_preserved(self):
        backend = ScriptedDetector([
            [detection((80., 10., 120., 50.), .7)], [],
            [detection((10., 10., 50., 50.), .9), detection((10., 10., 50., 50.), .8, 14)],
        ])
        result = TiledDetector(backend, tile_size=100).detect(np.zeros((100, 170, 3), dtype=np.uint8))
        self.assertEqual(result, [
            detection((80., 10., 120., 50.), .9), detection((80., 10., 120., 50.), .8, 14),
        ])

    def test_final_output_limit_applies_after_cross_pass_merge(self):
        backend = ScriptedDetector([
            [detection((0., 0., 10., 10.), .7)],
            [detection((20., 20., 30., 30.), .9)],
            [detection((20., 20., 30., 30.), .8)],
        ])
        result = TiledDetector(backend, tile_size=100, max_detections=2).detect(np.zeros((100, 170, 3), dtype=np.uint8))
        self.assertEqual([item.confidence for item in result], [.9, .8])
        self.assertEqual(backend.inference_count, 3)


class TiledFailureTests(unittest.TestCase):
    def test_invalid_configuration_rejected_before_backend(self):
        for options in (
            {"tile_size": 0}, {"tile_size": True}, {"tile_size": 100.5}, {"tile_size": 4097},
            {"overlap": 1.}, {"overlap": -.1}, {"overlap": float("nan")}, {"overlap": ".25"},
            {"max_tiles": 0}, {"max_tiles": 65}, {"max_tiles": False},
            {"edge_margin_px": -1}, {"edge_margin_px": 481}, {"edge_margin_px": 2.5},
            {"nms_iou": 0}, {"nms_iou": float("inf")}, {"nms_iou": True},
            {"max_detections": 0}, {"max_detections": 1001},
        ):
            backend = ScriptedDetector([])
            with self.subTest(options=options), self.assertRaises(ValueError):
                TiledDetector(backend, **options)
            self.assertEqual(backend.images, [])
        with self.assertRaises(ValueError):
            TiledDetector(object())

    def test_invalid_input_rejected_before_backend(self):
        backend = ScriptedDetector([])
        for image in ("missing", np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32),
                      np.zeros((0, 10, 3), np.uint8), np.zeros((4097, 4097, 3), np.uint8)):
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                TiledDetector(backend).detect(image)
        self.assertEqual(backend.images, [])

    def test_bad_detection_fields_error_even_if_internal_edge_would_discard(self):
        invalid = [
            Detection((-1., 10., 30., 40.), .8, 0, "person"),
            Detection((0., 10., 101., 40.), .8, 0, "person"),
            Detection((0., 10., float("nan"), 40.), .8, 0, "person"),
            Detection((0., 10., float("inf"), 40.), .8, 0, "person"),
            Detection((0., 10., 0., 40.), .8, 0, "person"),
            Detection(("0", 10., 30., 40.), .8, 0, "person"),
            Detection((False, 10., 30., 40.), .8, 0, "person"),
            Detection([0., 10., 30., 40.], .8, 0, "person"),
            Detection((0., 10., 30.), .8, 0, "person"),
            Detection((0., 10., 30., 40.), float("nan"), 0, "person"),
            Detection((0., 10., 30., 40.), 1.1, 0, "person"),
            Detection((0., 10., 30., 40.), True, 0, "person"),
            Detection((0., 10., 30., 40.), .8, -1, "person"),
            Detection((0., 10., 30., 40.), .8, 80, "person"),
            Detection((0., 10., 30., 40.), .8, 0., "person"),
            Detection((0., 10., 30., 40.), .8, True, "person"),
            Detection((0., 10., 30., 40.), .8, 0, "bird"),
            Detection((0., 10., 30., 40.), .8, 0, None),
            "not a detection",
        ]
        for item in invalid:
            # Bad item lies on the second crop's internal left boundary. A
            # filter-first implementation could silently lose invalid values.
            backend = ScriptedDetector([[detection()], [], [item]])
            with self.subTest(item=item), self.assertRaises(ModelContractError):
                TiledDetector(backend, tile_size=100).detect(np.zeros((100, 170, 3), np.uint8))

    def test_invalid_full_pass_is_rejected_before_crops(self):
        backend = ScriptedDetector([[Detection((0., 0., 200., 20.), .9, 0, "person")]])
        with self.assertRaises(ModelContractError):
            TiledDetector(backend, tile_size=100).detect(np.zeros((100, 170, 3), np.uint8))
        self.assertEqual(len(backend.images), 1)

    def test_backend_failure_returns_no_partial_full_frame_result(self):
        backend = ScriptedDetector([[detection()], DetectorError("backend failed")])
        with self.assertRaisesRegex(DetectorError, "backend failed"):
            TiledDetector(backend, tile_size=100).detect(np.zeros((100, 170, 3), np.uint8))
        self.assertEqual(backend.inference_count, 1)

    def test_collection_and_per_pass_budgets_reject_without_iteration(self):
        for output in (None, {"detections": []}, iter([detection()]), [detection()] * 1001):
            backend = ScriptedDetector([output])
            with self.subTest(output_type=type(output)), self.assertRaises(ModelContractError):
                TiledDetector(backend).detect(np.zeros((50, 50, 3), np.uint8))

    def test_combined_budget_counts_raw_candidates_including_rejected_seams(self):
        backend = ScriptedDetector([[detection((1., 1., 2., 2.))] * 1000] * 17)
        with self.assertRaisesRegex(ModelContractError, "combined budget"):
            TiledDetector(backend, tile_size=20, overlap=0, max_tiles=16).detect(np.zeros((80, 80, 3), np.uint8))
        self.assertEqual(backend.inference_count, 11)

    def test_tiled_cost_obeys_pipeline_deadline_and_clears_existing_track(self):
        class Clock:
            now = 0.
            def __call__(self):
                return self.now

        clock = Clock()

        class TimedDetector:
            inference_count = 0
            latency = 0.
            def detect(self, image_rgb):
                clock.now += self.latency
                self.inference_count += 1
                return [detection()]

        backend = TimedDetector()
        wrapped = TiledDetector(backend, tile_size=100)
        pipeline = PerceptionPipeline(wrapped, max_frame_age_s=.3, clock=clock)
        def sample(sequence, image):
            return CameraSample(image, float(sequence), clock.now, sequence, "test", "sim", "optical", 0.)

        first = pipeline.process(sample(0, np.zeros((50, 50, 3), np.uint8)))
        self.assertEqual(first["status"], "ok")
        self.assertTrue(pipeline.tracker.tracks)
        # Each pass individually fits .3 s, but their combined .36 s must not.
        backend.latency = .12
        rejected = pipeline.process(sample(1, np.zeros((100, 170, 3), np.uint8)))
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["reason"], "inference_deadline_missed")
        self.assertTrue(rejected["inference_executed"])
        self.assertEqual(rejected["detections"], [])
        self.assertEqual(pipeline.tracker.tracks, {})
        self.assertEqual(wrapped.inference_count, 4)
        self.assertFalse(rejected["control_authority"])


if __name__ == "__main__":
    unittest.main()
