"""Studio-only, same-frame person retry; all freshness checks remain downstream.

YOLO thresholds, class labels and its per-call NMS are untouched. An empty-person
primary result permits one horizontal-flip inference. Only genuine retry person
boxes are added, since all other classes already have a primary observation.
No images, boxes, tracks or result authority survive between calls.
"""
from __future__ import annotations

import math
import time
from typing import Callable

import numpy as np

from experiments.flight_contracts import MAX_OBSERVATION_AGE_S
from perception.detector import COCO_CLASSES, Detection, Detector, ModelContractError, validate_rgb_image


def _validate_detections(detections, height: int, width: int) -> list[Detection]:
    """Reject malformed output before it can trigger or populate a retry."""
    if not isinstance(detections, (list, tuple)) or len(detections) > 1000:
        raise ModelContractError('Studio detector requires at most 1000 detections')
    for detection in detections:
        if not isinstance(detection, Detection):
            raise ModelContractError('Studio detector requires Detection objects')
        try:
            box, class_id = detection.bbox_xyxy, detection.class_id
            valid = (
                len(box) == 4 and all(math.isfinite(value) for value in box)
                and 0 <= box[0] < box[2] <= width
                and 0 <= box[1] < box[3] <= height
                and not isinstance(detection.confidence, (bool, np.bool_))
                and math.isfinite(detection.confidence)
                and 0 <= detection.confidence <= 1
                and isinstance(class_id, (int, np.integer))
                and not isinstance(class_id, (bool, np.bool_))
                and 0 <= class_id < len(COCO_CLASSES)
                and detection.label == COCO_CLASSES[class_id]
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            raise ModelContractError('Studio detector returned invalid geometry, score or COCO class')
    return list(detections)


class StudioPersonRetryDetector:
    """Sequential detector wrapper with at most two actual backend calls.

    ``last_receipt`` is invocation-local metadata, never a detection cache. Its
    ``backend_calls`` counts attempted calls, including a call that raises;
    ``backend_calls_completed`` counts returned, contract-valid calls. Errors
    propagate without falling back to an earlier result. ``used_retry`` means
    retry persons entered the raw return, not that the pipeline accepted them.

    The local elapsed bound only prevents starting an already-late retry. The
    surrounding MantisPerceptionPipeline must still reject total capture age
    above .65 s, including both calls, transforms, validation and tracking.
    """
    def __init__(self, backend: Detector, *,
                 max_detection_elapsed_s: float = MAX_OBSERVATION_AGE_S,
                 clock: Callable[[], float] = time.monotonic):
        if not callable(getattr(backend, 'detect', None)) or not callable(clock):
            raise ValueError('Studio retry requires a detector and monotonic clock')
        if (isinstance(max_detection_elapsed_s, (bool, np.bool_))
                or not isinstance(max_detection_elapsed_s, (int, float, np.integer, np.floating))
                or not math.isfinite(max_detection_elapsed_s)
                or not 0 < max_detection_elapsed_s <= MAX_OBSERVATION_AGE_S):
            raise ValueError('Studio retry elapsed bound must be in (0, .65] seconds')
        self.backend = backend
        self.max_detection_elapsed_s = float(max_detection_elapsed_s)
        self.clock = clock
        self.backend_call_count = 0
        self.last_receipt = None

    def _time(self, previous=None):
        value = self.clock()
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not math.isfinite(value) or value < 0
                or (previous is not None and value < previous)):
            raise ValueError('Studio retry clock must be finite, nonnegative and monotonic')
        return float(value)

    def _detect(self, image, phase):
        receipt = self.last_receipt
        receipt['backend_calls'] += 1
        self.backend_call_count += 1
        receipt['outcome'] = phase + '_error'
        # Inference and validation errors intentionally propagate. In particular,
        # a retry exception cannot become a successful empty/primary result.
        result = _validate_detections(self.backend.detect(image), *image.shape[:2])
        receipt['backend_calls_completed'] += 1
        return result

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        self.last_receipt = {
            'backend_calls': 0, 'backend_calls_completed': 0,
            'retry_attempted': False, 'primary_person_count': None,
            'retry_person_count': None, 'used_retry': False,
            'transform': 'horizontal_flip_same_frame',
            'elapsed_s': None, 'outcome': 'invalid_input',
        }
        validate_rgb_image(image_rgb)
        started = self._time()
        # Separate input arrays also stop a mutating backend from changing the
        # source image or contaminating the mirrored view with its first call.
        primary = self._detect(image_rgb.copy(order='C'), 'primary')
        checked = self._time(started)
        receipt = self.last_receipt
        receipt['elapsed_s'] = checked - started
        receipt['primary_person_count'] = sum(d.class_id == 0 for d in primary)
        if receipt['primary_person_count']:
            receipt['outcome'] = 'primary_person_found'
            return primary
        if checked - started >= self.max_detection_elapsed_s:
            receipt['outcome'] = 'retry_skipped_elapsed_budget'
            return primary
        flipped = image_rgb[:, ::-1, :].copy(order='C')
        checked = self._time(checked)
        receipt['elapsed_s'] = checked - started
        if checked - started >= self.max_detection_elapsed_s:
            receipt['outcome'] = 'retry_skipped_elapsed_budget'
            return primary
        receipt['retry_attempted'] = True
        retry = self._detect(flipped, 'retry')
        width = image_rgb.shape[1]
        persons = [Detection((float(width-d.bbox_xyxy[2]), float(d.bbox_xyxy[1]),
                              float(width-d.bbox_xyxy[0]), float(d.bbox_xyxy[3])),
                             d.confidence, d.class_id, d.label)
                   for d in retry if d.class_id == 0]
        # Coordinates denote continuous pixel edges, so use W-x, never W-1-x.
        result = _validate_detections(primary + persons, *image_rgb.shape[:2])
        finished = self._time(checked)
        receipt.update(retry_person_count=len(persons), used_retry=bool(persons),
                       elapsed_s=finished-started,
                       outcome='retry_person_found' if persons else 'retry_no_person')
        return result
