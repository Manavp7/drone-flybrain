"""Optional YOLOX ONNX object detection. This module has no flight authority.

Input is uint8 RGB HWC; ONNX input is float32 BGR NCHW, without normalization,
with an aspect-preserving resize and top-left placement on a 114-valued canvas.
This is the official ONNX demo preprocessing, NOT ValTransform(legacy=True).
Only 80-class COCO, batch-one, P5 exports at 416 or 640 are supported. Raw
YOLOX outputs require grid decoding; decoded_cxcywh exports must opt in.

Primary implementation contracts:
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/onnx_inference.py
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/demo_utils.py
https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/datasets/coco_classes.py
https://docs.opencv.org/4.13.0/d6/d0f/group__dnn.html
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import re
import threading
from typing import Callable, Protocol

import numpy as np


COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)
OUTPUT_FORMATS = ("raw_yolox", "decoded_cxcywh")
MAX_IMAGE_PIXELS = 4096 * 4096
MAX_MODEL_BYTES = 512 * 1024 * 1024


class DetectorError(RuntimeError):
    """An unavailable model/backend or a failed inference, never an empty result."""


class ModelContractError(ValueError):
    """The supplied model output does not meet the declared YOLOX contract."""


@dataclass(frozen=True, slots=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    label: str


class Detector(Protocol):
    def detect(self, image_rgb: np.ndarray) -> list[Detection]: ...


@dataclass(frozen=True, slots=True)
class Letterbox:
    original_height: int
    original_width: int
    input_size: int
    ratio: float
    resized_height: int
    resized_width: int


def _input_size(value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or value not in (416, 640):
        raise ValueError("Only square YOLOX P5 input_size 416 or 640 is supported")
    return int(value)


def _threshold(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number in (0, 1]")
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError(f"{name} must be a finite number in (0, 1]")
    return value


def _limit(value: int, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer between 1 and {maximum}")
    return int(value)


def validate_rgb_image(image_rgb: np.ndarray) -> None:
    if not isinstance(image_rgb, np.ndarray) or image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be a uint8 NumPy RGB HWC array")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or min(image_rgb.shape[:2]) < 1:
        raise ValueError("image_rgb must have nonempty shape (height, width, 3)")
    if image_rgb.shape[0] * image_rgb.shape[1] > MAX_IMAGE_PIXELS:
        raise ValueError("image_rgb exceeds the bounded 16-megapixel input budget")


def resize_bilinear_numpy(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Dependency-free half-pixel bilinear reference for helper tests.

    This is not claimed bit-identical to OpenCV's fixed-point rounding.
    YOLOXDetector always injects OpenCV INTER_LINEAR for actual inference.
    """
    source_height, source_width = image.shape[:2]
    xs = np.clip((np.arange(width) + .5) * source_width / width - .5, 0, source_width - 1)
    ys = np.clip((np.arange(height) + .5) * source_height / height - .5, 0, source_height - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1, y1 = np.minimum(x0 + 1, source_width - 1), np.minimum(y0 + 1, source_height - 1)
    wx, wy = (xs - x0)[None, :, None], (ys - y0)[:, None, None]
    top = image[y0[:, None], x0[None, :]] * (1 - wx) + image[y0[:, None], x1[None, :]] * wx
    bottom = image[y1[:, None], x0[None, :]] * (1 - wx) + image[y1[:, None], x1[None, :]] * wx
    return np.clip(np.rint(top * (1 - wy) + bottom * wy), 0, 255).astype(np.uint8)


def preprocess_rgb(
    image_rgb: np.ndarray,
    input_size: int = 640,
    *,
    resize: Callable[[np.ndarray, int, int], np.ndarray] = resize_bilinear_numpy,
) -> tuple[np.ndarray, Letterbox]:
    """Return a BGR NCHW float32 tensor and its exact inverse transform metadata."""
    validate_rgb_image(image_rgb)
    size = _input_size(input_size)
    height, width = image_rgb.shape[:2]
    ratio = min(size / height, size / width)
    resized_height, resized_width = int(height * ratio), int(width * ratio)
    if min(resized_height, resized_width) < 1:
        raise ValueError("Image aspect ratio produces an empty resized dimension")
    resized = resize(np.ascontiguousarray(image_rgb[:, :, ::-1]), resized_width, resized_height)
    if resized.shape != (resized_height, resized_width, 3) or resized.dtype != np.uint8:
        raise ValueError("resize must return the requested uint8 HWC BGR image")
    padded = np.full((size, size, 3), 114, dtype=np.uint8)
    padded[:resized_height, :resized_width] = resized
    blob = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=np.float32)
    return blob, Letterbox(height, width, size, ratio, resized_height, resized_width)


def decode_yolox(output: np.ndarray, input_size: int = 640, output_format: str = "raw_yolox") -> np.ndarray:
    """Validate then decode to Nx85 cx,cy,w,h,objectness,80 class probabilities.

    No automatic output-format inference, transposition, sigmoid, or extra NMS
    is hidden here. A mismatched export fails explicitly.
    """
    size = _input_size(input_size)
    if output_format not in OUTPUT_FORMATS:
        raise ModelContractError(f"output_format must be one of {OUTPUT_FORMATS}")
    if not isinstance(output, np.ndarray) or output.dtype.kind not in "fiu":
        raise ModelContractError("Model must produce a real numeric NumPy tensor")
    if output.ndim == 3 and output.shape[0] == 1:
        output = output[0]
    expected = sum((size // stride) ** 2 for stride in (8, 16, 32))
    if output.ndim != 2 or output.shape[1] != 85:
        raise ModelContractError("Expected shape (1, N, 85) or (N, 85), COCO80 batch one")
    if len(output) > expected or (output_format == "raw_yolox" and len(output) != expected):
        raise ModelContractError(f"{output_format} has invalid anchor count {len(output)}; expected {expected} for raw output")
    if not np.isfinite(output).all():
        raise ModelContractError("Model output contains NaN or infinity")
    if np.any(output[:, 4:] < 0) or np.any(output[:, 4:] > 1):
        raise ModelContractError("Objectness/class channels must be probabilities in [0, 1], not logits")
    decoded = output.astype(np.float64, copy=True)
    if output_format == "raw_yolox":
        grids, scales = [], []
        for stride in (8, 16, 32):
            axis = np.arange(size // stride)
            xx, yy = np.meshgrid(axis, axis)
            grids.append(np.column_stack((xx.ravel(), yy.ravel())))
            scales.append(np.full((len(axis) ** 2, 1), stride))
        scale = np.concatenate(scales)
        try:
            with np.errstate(over="raise", invalid="raise"):
                decoded[:, :2] = (decoded[:, :2] + np.concatenate(grids)) * scale
                decoded[:, 2:4] = np.exp(decoded[:, 2:4]) * scale
        except FloatingPointError as exc:
            raise ModelContractError("Raw YOLOX geometry overflows during decoding") from exc
    if not np.isfinite(decoded).all() or np.any(np.abs(decoded[:, :4]) > 1_000_000):
        raise ModelContractError("Decoded geometry is nonfinite or exceeds the pixel-coordinate contract")
    if np.any(decoded[:, 2:4] < 0):
        raise ModelContractError("Decoded box width and height cannot be negative")
    return decoded


def class_aware_nms(
    boxes: np.ndarray, scores: np.ndarray, class_ids: np.ndarray,
    iou_threshold: float = .45, max_detections: int = 100, max_candidates: int = 3000,
) -> list[int]:
    """Stable highest-score-first NMS with continuous xyxy coordinates."""
    threshold = _threshold(iou_threshold, "iou_threshold")
    count = _limit(max_detections, "max_detections", 1000)
    budget = _limit(max_candidates, "max_candidates", 10000)
    boxes, scores, class_ids = np.asarray(boxes), np.asarray(scores), np.asarray(class_ids)
    if boxes.shape != (len(scores), 4) or scores.ndim != 1 or class_ids.shape != scores.shape:
        raise ModelContractError("NMS expects matching Nx4 boxes, N scores and N class IDs")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or not np.isfinite(class_ids).all():
        raise ModelContractError("NMS inputs must be finite")
    if np.any(boxes[:, 2:] <= boxes[:, :2]) or np.any(scores < 0) or np.any(scores > 1):
        raise ModelContractError("NMS requires positive-area boxes and probability scores")
    if np.any(class_ids < 0) or np.any(class_ids != class_ids.astype(np.int64)):
        raise ModelContractError("NMS class IDs must be nonnegative integers")
    order = np.argsort(-scores, kind="stable")[:budget]
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = []
    while len(order) and len(keep) < count:
        current = int(order[0])
        keep.append(current)
        rest = order[1:]
        low = np.maximum(boxes[current, :2], boxes[rest, :2])
        high = np.minimum(boxes[current, 2:], boxes[rest, 2:])
        overlap = np.prod(np.maximum(0, high - low), axis=1)
        union = area[current] + area[rest] - overlap
        iou = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=float), where=union > 0)
        order = rest[(class_ids[rest] != class_ids[current]) | (iou <= threshold)]
    return keep


def postprocess_yolox(
    output: np.ndarray, transform: Letterbox, *, output_format: str = "raw_yolox",
    confidence: float = .35, nms_iou: float = .45,
    max_detections: int = 100, max_candidates: int = 3000,
) -> list[Detection]:
    """Select one best COCO class per anchor, unletterbox, clip, and apply NMS."""
    confidence = _threshold(confidence, "confidence")
    _threshold(nms_iou, "nms_iou")
    _limit(max_detections, "max_detections", 1000)
    _limit(max_candidates, "max_candidates", 10000)
    if not np.isfinite(transform.ratio) or transform.ratio <= 0 or min(transform.original_height, transform.original_width) < 1:
        raise ValueError("Invalid letterbox inverse transform")
    decoded = decode_yolox(output, transform.input_size, output_format)
    if len(decoded) == 0:
        return []
    classes = np.argmax(decoded[:, 5:], axis=1)
    scores = decoded[:, 4] * decoded[np.arange(len(decoded)), 5 + classes]
    selected = scores >= confidence
    boxes = decoded[selected, :4]
    scores, classes = scores[selected], classes[selected]
    xyxy = np.column_stack((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:] / 2)) / transform.ratio
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, transform.original_width)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, transform.original_height)
    valid = np.all(xyxy[:, 2:] > xyxy[:, :2], axis=1)
    xyxy, scores, classes = xyxy[valid], scores[valid], classes[valid]
    keep = class_aware_nms(xyxy, scores, classes, nms_iou, max_detections, max_candidates)
    return [Detection(tuple(float(x) for x in xyxy[i]), float(scores[i]), int(classes[i]), COCO_CLASSES[int(classes[i])]) for i in keep]


class YOLOXDetector:
    """A real OpenCV DNN backend; absent weights/dependencies are hard errors.

    The caller supplies a separately verified SHA256 and an export contract.
    The digest identifies bytes; it does not prove accuracy or model provenance.
    Importing this module does not import OpenCV or download anything.
    """
    def __init__(
        self, model_path: str | Path, input_size: int = 640,
        confidence: float = .35, nms_iou: float = .45, *,
        expected_sha256: str, output_format: str = "raw_yolox",
        max_detections: int = 100, max_candidates: int = 3000,
    ) -> None:
        self.input_size = _input_size(input_size)
        self.confidence = _threshold(confidence, "confidence")
        self.nms_iou = _threshold(nms_iou, "nms_iou")
        self.max_detections = _limit(max_detections, "max_detections", 1000)
        self.max_candidates = _limit(max_candidates, "max_candidates", 10000)
        if output_format not in OUTPUT_FORMATS:
            raise ModelContractError(f"output_format must be one of {OUTPUT_FORMATS}")
        self.output_format = output_format
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha256):
            raise ValueError("expected_sha256 must be a separately verified 64-digit hexadecimal digest")
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}; no model is downloaded automatically")
        if self.model_path.suffix.lower() != ".onnx":
            raise ValueError("model_path must identify a .onnx file")
        # Load/hash the same immutable byte buffer passed to OpenCV; path mutation
        # between checksum and readNetFromONNX cannot silently change the model.
        with self.model_path.open("rb") as stream:
            model_bytes = stream.read(MAX_MODEL_BYTES + 1)
        if not model_bytes or len(model_bytes) > MAX_MODEL_BYTES:
            raise ValueError("ONNX model must be nonempty and no larger than 512 MiB")
        self.model_sha256 = hashlib.sha256(model_bytes).hexdigest()
        if self.model_sha256 != expected_sha256.lower():
            raise ValueError("ONNX model SHA256 mismatch; backend was not loaded")
        try:
            self._cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise DetectorError("OpenCV DNN is unavailable; install opencv-python-headless to run ONNX inference") from exc
        try:
            self._net = self._cv2.dnn.readNetFromONNX(np.frombuffer(model_bytes, dtype=np.uint8))
            self._net.setPreferableBackend(self._cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(self._cv2.dnn.DNN_TARGET_CPU)
            self._output_names = tuple(self._net.getUnconnectedOutLayersNames())
            if len(self._output_names) != 1:
                raise ModelContractError("Expected one concatenated YOLOX output tensor; multi-head exports are unsupported")
        except ModelContractError:
            raise
        except Exception as exc:
            raise DetectorError("OpenCV could not load this ONNX export") from exc
        self._lock = threading.Lock()
        self.inference_count = 0

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        blob, transform = preprocess_rgb(
            image_rgb, self.input_size,
            resize=lambda image, width, height: self._cv2.resize(image, (width, height), interpolation=self._cv2.INTER_LINEAR),
        )
        try:
            with self._lock:
                self._net.setInput(blob)
                output = self._net.forward(self._output_names[0])
                # Copy before releasing the mutable network to another caller.
                if not isinstance(output, np.ndarray):
                    raise ModelContractError("OpenCV forward did not return one NumPy output tensor")
                output = output.copy()
        except ModelContractError:
            raise
        except Exception as exc:
            raise DetectorError("OpenCV ONNX inference failed; no detections were fabricated") from exc
        detections = postprocess_yolox(
            output, transform, output_format=self.output_format,
            confidence=self.confidence, nms_iou=self.nms_iou,
            max_detections=self.max_detections, max_candidates=self.max_candidates,
        )
        self.inference_count += 1
        return detections
