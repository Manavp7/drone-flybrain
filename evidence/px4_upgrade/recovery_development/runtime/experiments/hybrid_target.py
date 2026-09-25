"""Causal, persistent YOLO target selection for an explicitly clocked simulation.

This adapter supplies image geometry, never inferred metric depth or identity.
``now_s`` must use the same clock as ``capture_time_s``. A file replay using media
time must be labelled as replay by its caller; it does not establish live latency.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


def _image_hw(image_hw):
    if (not isinstance(image_hw, (tuple, list)) or len(image_hw) != 2
            or any(not isinstance(v, Integral) or isinstance(v, (bool, np.bool_)) or v < 2 for v in image_hw)):
        raise ValueError("image_hw must contain two positive image dimensions >= 2")
    return int(image_hw[0]), int(image_hw[1])


class TargetBridge:
    """Choose once and retain that temporary track ID, including through loss.

    A missing target can recover only with the same ID in the same stream. A
    changed stream latches loss; construct a new bridge to select explicitly.
    No class or box is invented, and another visible person cannot replace loss.
    """

    def __init__(self, track_id=None):
        if track_id is not None and (not isinstance(track_id, Integral)
                or isinstance(track_id, (bool, np.bool_)) or track_id < 1):
            raise ValueError("track_id must be a positive integer or None")
        self.track_id = None if track_id is None else int(track_id)
        self.stream = None
        self.last_sequence = self.last_capture = None
        self.stream_changed = False

    def _invalid(self, reason):
        return {"valid": False, "reason": reason, "track_id": self.track_id,
                "bbox_xyxy": None, "center_normalized": None,
                "height_fraction": None, "area_fraction": None,
                "depth_m": None, "control_authority": False}

    def update(self, result, now_s, image_hw, max_age_s=.25):
        height, width = _image_hw(image_hw)
        if not _finite(now_s) or now_s < 0 or not _finite(max_age_s) or max_age_s <= 0:
            raise ValueError("now_s and max_age_s must be finite and nonnegative/positive")
        if not isinstance(result, dict):
            return self._invalid("invalid_result")
        if self.stream_changed:
            return self._invalid("stream_changed")
        stream = tuple(result.get(k) for k in ("stream_id", "clock_domain", "frame_id"))
        if any(not isinstance(v, str) or not v for v in stream):
            return self._invalid("missing_stream_identity")
        if self.stream is not None and stream != self.stream:
            self.stream_changed = True
            return self._invalid("stream_changed")
        sequence, timestamp = result.get("sequence"), result.get("capture_time_s")
        if (not isinstance(sequence, Integral) or isinstance(sequence, (bool, np.bool_)) or sequence < 0
                or not _finite(timestamp) or timestamp < 0):
            return self._invalid("invalid_frame_identity")
        if self.last_sequence is not None and (sequence <= self.last_sequence or timestamp <= self.last_capture):
            return self._invalid("duplicate_or_reordered_frame")
        age = float(now_s - timestamp)
        if age < -1e-9:
            return self._invalid("future_capture_time")
        if age > max_age_s + 1e-9:
            return self._invalid("stale_target_frame")
        if result.get("status") != "ok" or result.get("inference_executed") is not True:
            return self._invalid("perception_unavailable")
        detections = result.get("detections")
        if not isinstance(detections, list) or len(detections) > 1000:
            return self._invalid("invalid_detection_collection")
        # Validate before changing stream/selection history. Reject ambiguous IDs
        # rather than choosing one of two conflicting observations of a target.
        ids = set()
        people = []
        for detection in detections:
            if not isinstance(detection, dict):
                return self._invalid("invalid_detection")
            tid = detection.get("track_id")
            if (not isinstance(tid, Integral) or isinstance(tid, (bool, np.bool_)) or tid < 1 or tid in ids):
                return self._invalid("invalid_or_duplicate_track_id")
            ids.add(tid)
            box, confidence = detection.get("bbox_xyxy"), detection.get("confidence")
            if (not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite(x) for x in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height
                    or not _finite(confidence) or not 0 <= confidence <= 1):
                return self._invalid("invalid_detection")
            if detection.get("class_id") == 0 and detection.get("label") == "person":
                people.append(detection)
        self.stream = stream
        self.last_sequence, self.last_capture = int(sequence), float(timestamp)
        if self.track_id is None and people:
            self.track_id = int(min(people, key=lambda d: (-d["confidence"], d["track_id"]))["track_id"])
        selected = next((d for d in people if d["track_id"] == self.track_id), None)
        if selected is None:
            return self._invalid("target_not_observed" if self.track_id is not None else "no_person_observed")
        x0, y0, x1, y1 = map(float, selected["bbox_xyxy"])
        return {"valid": True, "reason": "observed", "track_id": self.track_id,
                "bbox_xyxy": [x0, y0, x1, y1],
                "center_normalized": [(x0+x1)/width - 1., (y0+y1)/height - 1.],
                "height_fraction": (y1-y0)/height, "area_fraction": (x1-x0)*(y1-y0)/(width*height),
                "confidence": float(selected["confidence"]), "sequence": int(sequence),
                "capture_time_s": float(timestamp), "age_s": max(0., age),
                "depth_m": None, "control_authority": False}


def salience_mask(observation, image_hw, output_size=391):
    """Dark YOLO rectangle on a grey square, with isotropic letterbox mapping.

    This is an engineered visual cue supplied to Flyvis, not a raw camera image.
    The output uses actual valid detector geometry only. Loss produces plain grey.
    """
    height, width = _image_hw(image_hw)
    if not isinstance(output_size, Integral) or isinstance(output_size, (bool, np.bool_)) or output_size < 2:
        raise ValueError("output_size must be an integer >= 2")
    output_size = int(output_size)
    mask = np.full((output_size, output_size), .5, np.float32)
    if not isinstance(observation, dict) or observation.get("valid") is not True:
        return mask
    box = observation.get("bbox_xyxy")
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or not all(_finite(x) for x in box)
            or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
        raise ValueError("valid target requires an on-image bbox_xyxy")
    scale = output_size / max(height, width)
    offset_x, offset_y = (output_size-width*scale)/2, (output_size-height*scale)/2
    x0, y0 = math.floor(offset_x+box[0]*scale), math.floor(offset_y+box[1]*scale)
    x1, y1 = math.ceil(offset_x+box[2]*scale), math.ceil(offset_y+box[3]*scale)
    mask[max(0,y0):min(output_size,y1), max(0,x0):min(output_size,x1)] = .1
    return mask
