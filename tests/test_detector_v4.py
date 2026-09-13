"""Numerical contract tests. Stub backend tests do not measure neural accuracy."""
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from perception.detector import (
    COCO_CLASSES, DetectorError, Letterbox, ModelContractError, YOLOXDetector,
    class_aware_nms, decode_yolox, postprocess_yolox, preprocess_rgb,
    resize_bilinear_numpy,
)


def decoded_rows(*specifications):
    rows = np.zeros((len(specifications), 85), dtype=np.float32)
    for index, (xywh, objectness, class_id, class_probability) in enumerate(specifications):
        rows[index, :4] = xywh
        rows[index, 4] = objectness
        rows[index, 5 + class_id] = class_probability
    return rows


def identity_transform(size=640):
    return Letterbox(size, size, size, 1., size, size)


class LetterboxTests(unittest.TestCase):
    def test_rgb_to_bgr_top_left_padding_without_normalization(self):
        rgb = np.empty((2, 4, 3), dtype=np.uint8)
        rgb[:] = [255, 31, 7]
        blob, transform = preprocess_rgb(rgb, 416)
        self.assertEqual(blob.shape, (1, 3, 416, 416))
        self.assertEqual(blob.dtype, np.float32)
        self.assertTrue(blob.flags.c_contiguous)
        self.assertEqual(transform.ratio, 104)
        self.assertEqual((transform.resized_height, transform.resized_width), (208, 416))
        np.testing.assert_array_equal(blob[0, :, 0, 0], [7, 31, 255])
        np.testing.assert_array_equal(blob[0, :, 207, 415], [7, 31, 255])
        np.testing.assert_array_equal(blob[0, :, 208, 0], [114, 114, 114])
        np.testing.assert_array_equal(rgb[0, 0], [255, 31, 7])

    def test_resize_half_pixel_reference_has_correct_midpoint_and_edges(self):
        image = np.repeat(np.array([[[0], [100]]], dtype=np.uint8), 3, axis=2)
        actual = resize_bilinear_numpy(image, 4, 1)
        np.testing.assert_array_equal(actual[0, :, 0], [0, 25, 75, 100])

    def test_letterbox_inverse_restores_original_coordinates(self):
        _, transform = preprocess_rgb(np.zeros((300, 800, 3), dtype=np.uint8), 640)
        # Original box x=100..300, y=50..150. Scale .8 to model pixels.
        output = decoded_rows(((160, 80, 160, 80), .9, 2, .8))
        found = postprocess_yolox(output, transform, output_format="decoded_cxcywh")
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0].bbox_xyxy, (100, 50, 300, 150))
        self.assertAlmostEqual(found[0].confidence, .72, places=6)
        self.assertEqual(found[0].label, "car")

    def test_fractional_resize_matches_official_floor_dimensions_and_ratio(self):
        _, transform = preprocess_rgb(np.zeros((333, 1000, 3), dtype=np.uint8), 416)
        self.assertEqual(transform.resized_height, 138)
        self.assertEqual(transform.ratio, .416)
        output = decoded_rows(((104, 41.6, 83.2, 41.6), 1, 0, .9))
        found = postprocess_yolox(output, transform, output_format="decoded_cxcywh")
        np.testing.assert_allclose(found[0].bbox_xyxy, (150, 50, 350, 150), atol=1e-4)

    def test_invalid_camera_shapes_and_dtype_fail_explicitly(self):
        for image in (
            np.zeros((10, 10, 3), dtype=np.float32), np.zeros((10, 10), dtype=np.uint8),
            np.zeros((0, 10, 3), dtype=np.uint8), np.zeros((10, 10, 4), dtype=np.uint8),
            np.zeros((1, 10000, 3), dtype=np.uint8), "camera missing",
        ):
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                preprocess_rgb(image)


class DecodeTests(unittest.TestCase):
    def test_all_three_raw_strides_and_row_order_640_and_416(self):
        for size in (416, 640):
            with self.subTest(size=size):
                counts = [(size // stride) ** 2 for stride in (8, 16, 32)]
                output = np.zeros((1, sum(counts), 85), dtype=np.float32)
                output[0, 1, :4] = [.5, .25, np.log(2), np.log(3)]
                before = output.copy()
                decoded = decode_yolox(output, size)
                np.testing.assert_allclose(decoded[1, :4], [12, 2, 16, 24], rtol=1e-6)
                np.testing.assert_allclose(decoded[size // 8, :4], [0, 8, 8, 8])
                np.testing.assert_allclose(decoded[counts[0], :4], [0, 0, 16, 16])
                np.testing.assert_allclose(decoded[sum(counts[:2]), :4], [0, 0, 32, 32])
                np.testing.assert_array_equal(output, before)

    def test_decoded_contract_never_decodes_again(self):
        output = decoded_rows(((320, 200, 40, 90), .9, 1, .9))
        np.testing.assert_array_equal(decode_yolox(output, 640, "decoded_cxcywh"), output)
        with self.assertRaisesRegex(ModelContractError, "anchor count"):
            decode_yolox(output, 640, "raw_yolox")

    def test_wrong_batch_transposition_class_count_and_format_rejected(self):
        for output in (np.zeros((2, 8400, 85)), np.zeros((1, 85, 8400)), np.zeros((3, 84)), np.zeros((8401, 85)), np.zeros((85,)), np.zeros((2, 85), dtype=complex)):
            with self.subTest(shape=output.shape), self.assertRaises(ModelContractError):
                decode_yolox(output, output_format="decoded_cxcywh")
        with self.assertRaises(ModelContractError):
            decode_yolox(np.zeros((0, 85)), output_format="auto")
        with self.assertRaises(ValueError):
            decode_yolox(np.zeros((0, 85)), input_size=608)

    def test_nonfinite_geometry_probabilities_and_negative_width_rejected(self):
        for column, value in ((0, np.nan), (2, np.inf), (4, 1.1), (5, -.1), (2, -1), (0, 1e10)):
            output = decoded_rows(((10, 10, 4, 4), .9, 2, .8))
            output[0, column] = value
            with self.subTest(column=column, value=value), self.assertRaises(ModelContractError):
                decode_yolox(output, output_format="decoded_cxcywh")

    def test_raw_exponential_overflow_is_error_not_clamped_detection(self):
        output = np.zeros((8400, 85))
        output[0, 2] = 1000
        with self.assertRaisesRegex(ModelContractError, "overflows"):
            decode_yolox(output)


class PostprocessTests(unittest.TestCase):
    def test_objectness_times_class_not_class_confidence_alone(self):
        output = decoded_rows(((50, 50, 20, 20), .2, 0, .99), ((100, 100, 20, 20), .8, 7, .9))
        found = postprocess_yolox(output, identity_transform(), output_format="decoded_cxcywh", confidence=.35)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].class_id, 7)
        self.assertEqual(found[0].label, "truck")
        self.assertAlmostEqual(found[0].confidence, .72, places=6)

    def test_same_class_suppressed_different_class_preserved(self):
        output = decoded_rows(
            ((100, 100, 80, 80), 1, 0, .95),
            ((101, 101, 80, 80), 1, 0, .9),
            ((100, 100, 80, 80), 1, 2, .85),
        )
        found = postprocess_yolox(output, identity_transform(), output_format="decoded_cxcywh")
        self.assertEqual([d.class_id for d in found], [0, 2])

    def test_padding_boxes_removed_and_image_edges_clipped(self):
        transform = Letterbox(200, 400, 640, 1.6, 320, 640)
        output = decoded_rows(((10, 10, 80, 80), 1, 0, .9), ((100, 500, 20, 20), 1, 2, .9))
        found = postprocess_yolox(output, transform, output_format="decoded_cxcywh")
        self.assertEqual(len(found), 1)
        np.testing.assert_allclose(found[0].bbox_xyxy, [0, 0, 31.25, 31.25])

    def test_no_candidates_is_valid_empty_result(self):
        for output in (np.zeros((0, 85)), decoded_rows(((20, 20, 10, 10), .1, 0, .2))):
            found = postprocess_yolox(output, identity_transform(), output_format="decoded_cxcywh")
            self.assertEqual(found, [])
        # Raw output remains fixed-size even when there are no confident objects.
        self.assertEqual(postprocess_yolox(np.zeros((8400, 85)), identity_transform()), [])

    def test_nms_candidate_and_final_output_budgets_are_enforced(self):
        boxes = np.array([[i * 20, 0, i * 20 + 10, 10] for i in range(12)], dtype=float)
        scores = np.linspace(.5, .99, 12)
        classes = np.zeros(12, dtype=int)
        self.assertEqual(class_aware_nms(boxes, scores, classes, max_candidates=4, max_detections=2), [11, 10])
        self.assertEqual(class_aware_nms(boxes, scores, classes, max_candidates=3, max_detections=9), [11, 10, 9])

    def test_nms_uses_continuous_box_area_and_stable_ties(self):
        boxes = np.array([[0, 0, 1, 1], [1, 0, 2, 1], [0, 0, 1, 1]], dtype=float)
        self.assertEqual(class_aware_nms(boxes, np.array([.9, .9, .9]), np.zeros(3)), [0, 1])

    def test_thresholds_invalid_limits_and_malformed_nms_rejected(self):
        for kwargs in ({"confidence": float("nan")}, {"nms_iou": -.1}, {"max_candidates": 0}, {"max_detections": 2.5}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                postprocess_yolox(np.zeros((0, 85)), identity_transform(), output_format="decoded_cxcywh", **kwargs)
        with self.assertRaises(ModelContractError):
            class_aware_nms(np.array([[1, 1, 0, 0.]]), np.array([.9]), np.array([0]))
        self.assertEqual(len(COCO_CLASSES), 80)
        self.assertEqual(COCO_CLASSES[-1], "toothbrush")


class BackendContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "unit-test-stub.onnx"
        self.model_bytes = b"test-only placeholder, never claimed to be an ONNX model"
        self.path.write_bytes(self.model_bytes)
        self.digest = hashlib.sha256(self.model_bytes).hexdigest()

    def test_missing_model_is_explicit_and_never_imports_backend(self):
        with patch("perception.detector.importlib.import_module") as backend:
            with self.assertRaisesRegex(FileNotFoundError, "no model is downloaded"):
                YOLOXDetector(self.path.with_name("absent.onnx"), expected_sha256=self.digest)
            backend.assert_not_called()

    def test_hash_mismatch_is_rejected_before_model_load(self):
        with patch("perception.detector.importlib.import_module") as backend:
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                YOLOXDetector(self.path, expected_sha256="0" * 64)
            backend.assert_not_called()
        with self.assertRaisesRegex(ValueError, "64-digit"):
            YOLOXDetector(self.path, expected_sha256="")

    def test_missing_opencv_never_becomes_empty_success(self):
        with patch("perception.detector.importlib.import_module", side_effect=ImportError("cv2 unavailable")):
            with self.assertRaisesRegex(DetectorError, "OpenCV DNN is unavailable"):
                YOLOXDetector(self.path, expected_sha256=self.digest)

    def stub_cv2(self, output, output_names=("output",), forward_error=None):
        # Exercising wiring only; there is no learned inference in this stub.
        state = {}
        def forward(name):
            state["forward_name"] = name
            if forward_error:
                raise forward_error
            return output
        net = SimpleNamespace(
            setPreferableBackend=lambda value: state.update(backend=value),
            setPreferableTarget=lambda value: state.update(target=value),
            getUnconnectedOutLayersNames=lambda: output_names,
            setInput=lambda blob: state.update(blob=blob), forward=forward,
        )
        def read_net(buffer):
            state["loaded_bytes"] = buffer.tobytes()
            return net
        cv2 = SimpleNamespace(
            dnn=SimpleNamespace(readNetFromONNX=read_net, DNN_BACKEND_OPENCV=3, DNN_TARGET_CPU=0),
            INTER_LINEAR=1,
            resize=lambda image, dims, interpolation: resize_bilinear_numpy(image, dims[0], dims[1]),
        )
        return cv2, state

    def test_backend_receives_hashed_bytes_bgr_and_declared_output_name(self):
        output = decoded_rows(((80, 40, 40, 40), .9, 0, .9))[None]
        cv2, state = self.stub_cv2(output)
        with patch("perception.detector.importlib.import_module", return_value=cv2):
            detector = YOLOXDetector(self.path, expected_sha256=self.digest, output_format="decoded_cxcywh")
        image = np.zeros((320, 640, 3), dtype=np.uint8)
        image[:, :, 0] = 255
        detections = detector.detect(image)
        self.assertEqual(state["loaded_bytes"], self.model_bytes)
        self.assertEqual(state["forward_name"], "output")
        np.testing.assert_array_equal(state["blob"][0, :, 0, 0], [0, 0, 255])
        self.assertEqual(state["backend"], 3)
        self.assertEqual(state["target"], 0)
        self.assertEqual(detector.model_sha256, self.digest)
        self.assertEqual(detector.inference_count, 1)
        self.assertEqual(len(detections), 1)

    def test_multioutput_export_requires_explicit_adapter(self):
        cv2, _ = self.stub_cv2(np.zeros((1, 8400, 85)), output_names=("head8", "head16", "head32"))
        with patch("perception.detector.importlib.import_module", return_value=cv2):
            with self.assertRaisesRegex(ModelContractError, "multi-head"):
                YOLOXDetector(self.path, expected_sha256=self.digest)

    def test_forward_exception_is_not_an_empty_detection_list(self):
        cv2, _ = self.stub_cv2(None, forward_error=RuntimeError("unsupported ONNX operator"))
        with patch("perception.detector.importlib.import_module", return_value=cv2):
            detector = YOLOXDetector(self.path, expected_sha256=self.digest)
        with self.assertRaisesRegex(DetectorError, "inference failed"):
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(detector.inference_count, 0)

    def test_corrupt_backend_tensor_is_not_counted_as_success(self):
        output = np.zeros((1, 8400, 85))
        output[0, 0, 5] = np.nan
        cv2, _ = self.stub_cv2(output)
        with patch("perception.detector.importlib.import_module", return_value=cv2):
            detector = YOLOXDetector(self.path, expected_sha256=self.digest)
        with self.assertRaises(ModelContractError):
            detector.detect(np.zeros((32, 32, 3), dtype=np.uint8))
        self.assertEqual(detector.inference_count, 0)


if __name__ == "__main__":
    unittest.main()
