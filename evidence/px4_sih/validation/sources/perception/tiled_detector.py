"""Bounded full-frame and overlapping-crop inference for the COCO detector.

Cropping preserves more source detail for small objects. It does not improve a
model's training distribution or guarantee correct classes. Every output here
comes from the wrapped detector; class IDs, labels and scores are not rewritten.
This module imports neither OpenCV nor a model backend and has no flight role.
"""
from __future__ import annotations

from numbers import Real

import numpy as np

from .detector import (
    COCO_CLASSES, MAX_IMAGE_PIXELS, Detection, Detector, ModelContractError,
    class_aware_nms, validate_rgb_image,
)


MAX_TILES = 64
MAX_PASS_DETECTIONS = 1000
MAX_COMBINED_CANDIDATES = 10000


def _integer(value: int, name: str, minimum: int, maximum: int) -> int:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{name} must be an integer between {minimum} and {maximum}")
    return int(value)


def _real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real number")
    return result


def _tile_windows(
    width: int, height: int, tile_size: int, stride: int, max_tiles: int,
) -> list[tuple[int, int, int, int]]:
    """Return bounded xyxy crops; the last window on each axis is clamped.

    Count before allocating the grid, so tiny tiles on a large image fail before
    either a large allocation or a backend call. The extra full-frame inference
    is not included in max_tiles. Images fitting one tile need only that pass.
    """
    crop_width, crop_height = min(tile_size, width), min(tile_size, height)
    if (crop_width, crop_height) == (width, height):
        return []
    nx = (width - crop_width + stride - 1) // stride + 1
    ny = (height - crop_height + stride - 1) // stride + 1
    if nx * ny > max_tiles:
        raise ValueError(
            f"Image requires {nx * ny} crops, exceeding max_tiles={max_tiles}; "
            "increase tile_size or the explicit tile budget"
        )
    xs = [min(index * stride, width - crop_width) for index in range(nx)]
    ys = [min(index * stride, height - crop_height) for index in range(ny)]
    return [(x, y, x + crop_width, y + crop_height) for y in ys for x in xs]


def _validated_detections(result: object, width: int, height: int) -> list[Detection]:
    """Reject malformed output before it can disappear behind seam filtering."""
    if not isinstance(result, (list, tuple)):
        raise ModelContractError("Wrapped detector must return a bounded list or tuple of Detection objects")
    if len(result) > MAX_PASS_DETECTIONS:
        raise ModelContractError(f"Wrapped detector exceeds the {MAX_PASS_DETECTIONS}-detection per-pass budget")
    validated = []
    for detection in result:
        if not isinstance(detection, Detection):
            raise ModelContractError("Wrapped detector returned a non-Detection object")
        box = detection.bbox_xyxy
        if not isinstance(box, tuple) or len(box) != 4:
            raise ModelContractError("Detection bbox_xyxy must be a four-number tuple")
        try:
            coords = tuple(_real(value, "Detection coordinate") for value in box)
            confidence = _real(detection.confidence, "Detection confidence")
        except ValueError as exc:
            raise ModelContractError(str(exc)) from exc
        x0, y0, x1, y1 = coords
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ModelContractError("Detection box must have positive area and stay within the supplied image or crop")
        if not 0 <= confidence <= 1:
            raise ModelContractError("Detection confidence must be in [0, 1]")
        class_id = detection.class_id
        if (
            isinstance(class_id, (bool, np.bool_))
            or not isinstance(class_id, (int, np.integer))
            or not 0 <= class_id < len(COCO_CLASSES)
        ):
            raise ModelContractError("Detection class_id must be a COCO80 integer in [0, 79]")
        if not isinstance(detection.label, str) or detection.label != COCO_CLASSES[int(class_id)]:
            raise ModelContractError("Detection label must match its COCO80 class_id")
        validated.append(Detection(coords, confidence, int(class_id), detection.label))
    return validated


class TiledDetector:
    """Add square crops to one full-frame COCO pass, then class-aware NMS.

    Crops are rectangular only when an image dimension is smaller than the tile.
    Internal-edge detections are discarded because they may be truncated; image
    edges are allowed. Full-frame detections remain available for large objects
    or objects split across crops. A backend error fails the complete call rather
    than returning partial or fabricated observations.
    """

    def __init__(
        self, detector: Detector, tile_size: int = 960, overlap: float = .25,
        max_tiles: int = 16, edge_margin_px: int = 3, nms_iou: float = .45,
        max_detections: int = 100,
    ) -> None:
        if not callable(getattr(detector, "detect", None)):
            raise ValueError("detector must provide a callable detect method")
        self.detector = detector
        self.tile_size = _integer(tile_size, "tile_size", 1, int(np.sqrt(MAX_IMAGE_PIXELS)))
        self.overlap = _real(overlap, "overlap")
        if not 0 <= self.overlap < 1:
            raise ValueError("overlap must be in [0, 1)")
        self.max_tiles = _integer(max_tiles, "max_tiles", 1, MAX_TILES)
        self.edge_margin_px = _integer(edge_margin_px, "edge_margin_px", 0, self.tile_size // 2)
        self.nms_iou = _real(nms_iou, "nms_iou")
        if not 0 < self.nms_iou <= 1:
            raise ValueError("nms_iou must be in (0, 1]")
        self.max_detections = _integer(max_detections, "max_detections", 1, 1000)
        self.stride = max(1, int(self.tile_size * (1 - self.overlap)))

    @property
    def inference_count(self) -> int:
        """Expose successfully validated backend passes, not image-call count."""
        return self.detector.inference_count

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        validate_rgb_image(image_rgb)
        height, width = image_rgb.shape[:2]
        windows = _tile_windows(width, height, self.tile_size, self.stride, self.max_tiles)
        # All configuration/grid checks finish before the first backend call.
        combined = _validated_detections(self.detector.detect(image_rgb), width, height)
        raw_candidate_count = len(combined)
        for x0, y0, x1, y1 in windows:
            crop = np.ascontiguousarray(image_rgb[y0:y1, x0:x1])
            detections = _validated_detections(self.detector.detect(crop), x1 - x0, y1 - y0)
            raw_candidate_count += len(detections)
            if raw_candidate_count > MAX_COMBINED_CANDIDATES:
                raise ModelContractError(f"Wrapped detector exceeds the {MAX_COMBINED_CANDIDATES}-candidate combined budget")
            for detection in detections:
                left, top, right, bottom = detection.bbox_xyxy
                margin = self.edge_margin_px
                if (
                    (x0 > 0 and left <= margin)
                    or (y0 > 0 and top <= margin)
                    or (x1 < width and x1 - x0 - right <= margin)
                    or (y1 < height and y1 - y0 - bottom <= margin)
                ):
                    continue
                combined.append(Detection(
                    (left + x0, top + y0, right + x0, bottom + y0),
                    detection.confidence, detection.class_id, detection.label,
                ))
        if not combined:
            return []
        keep = class_aware_nms(
            np.array([item.bbox_xyxy for item in combined], dtype=float),
            np.array([item.confidence for item in combined], dtype=float),
            np.array([item.class_id for item in combined], dtype=int),
            self.nms_iou, self.max_detections, MAX_COMBINED_CANDIDATES,
        )
        return [combined[index] for index in keep]
