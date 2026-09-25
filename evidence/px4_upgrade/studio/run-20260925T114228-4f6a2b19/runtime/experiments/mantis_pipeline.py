"""Keep observed association memory across a verified inference timeout.

Deadline rejection still returns no detections and grants no authority. Only the
small state from before that attempt is eligible for later reassociation; a new
actual detection must pass the existing class, overlap and clothing gates. This
does not establish personal identity or extend result/command freshness.
"""
from __future__ import annotations

import math

import numpy as np

from perception.detector import Detection
from perception.pipeline import PerceptionPipeline


def _valid_detection_return(detections, image):
    """A timeout precedes the base validator, so inspect its return separately."""
    if not isinstance(detections, (list, tuple)) or len(detections) > 1000:
        return False
    height, width = image.shape[:2]
    for detection in detections:
        if not isinstance(detection, Detection):
            return False
        try:
            box = detection.bbox_xyxy
            valid = (len(box) == 4 and all(math.isfinite(v) for v in box)
                     and 0 <= box[0] < box[2] <= width
                     and 0 <= box[1] < box[3] <= height
                     and math.isfinite(detection.confidence)
                     and 0 <= detection.confidence <= 1
                     and isinstance(detection.class_id, (int, np.integer))
                     and not isinstance(detection.class_id, (bool, np.bool_))
                     and detection.class_id >= 0
                     and isinstance(detection.label, str) and bool(detection.label))
        except (TypeError, ValueError, OverflowError):
            return False
        if not valid:
            return False
    return True


class _ObservedDetector:
    """Delegate inference once; retain only validity, never its image or boxes."""
    def __init__(self, detector):
        self.detector = detector
        self.valid_return = False

    def detect(self, image):
        detections = self.detector.detect(image)
        self.valid_return = _valid_detection_return(detections, image)
        return detections


class MantisPerceptionPipeline(PerceptionPipeline):
    """Sequential PerceptionPipeline API with bounded timeout-only retention.

    The source stream and sequence/capture watermarks are never rolled back.
    Invalid input, clock anomalies, detector errors and stream/order changes
    retain the original clearing behavior. Stale-at-entry samples also clear.
    """
    def _snapshot_memory(self):
        tracker = self.tracker
        if len(tracker.tracks) > tracker.max_tracks:
            raise ValueError('tracker memory exceeds its configured bound')
        tracks, anchors = {}, {}
        for track_id, track in tracker.tracks.items():
            appearance = track['appearance']
            if appearance is not None:
                if not isinstance(appearance, np.ndarray) or appearance.shape != (64,):
                    raise ValueError('unexpected bounded clothing descriptor')
                appearance = appearance.copy()
            tracks[track_id] = dict(bbox=tuple(track['bbox']), class_id=track['class_id'],
                                    time=track['time'], appearance=appearance)
            anchor = getattr(tracker, 'anchors', {}).get(track_id)
            if anchor is not None:
                corners = anchor['corners']
                if not isinstance(corners, np.ndarray) or corners.shape != (4, 3):
                    raise ValueError('unexpected bounded observed anchor')
                anchors[track_id] = dict(corners=corners.copy(),
                                        capture_time_s=anchor['capture_time_s'])
        return tracks, anchors, tracker.last_time, tracker.next_id

    def process(self, sample):
        original_clock, original_detector = self.clock, self.detector
        observed_detector = _ObservedDetector(original_detector)
        readings = []

        def checked_clock():
            value = original_clock()
            previous = getattr(self, '_last_clock_read_s', None)
            if (not math.isfinite(value) or value < 0
                    or (previous is not None and value < previous)):
                raise ValueError('perception clock must be finite and monotonic')
            self._last_clock_read_s = value
            readings.append(value)
            return value

        try:
            # Validate before considering a saved state. The base validates too;
            # this adds no clock read or detector execution.
            sample.validate()
            established_forward_stream = (
                self.stream == (sample.stream_id, sample.clock_domain, sample.frame_id)
                and self.last_sequence is not None and self.last_capture is not None
                and sample.sequence > self.last_sequence
                and sample.capture_time_s > self.last_capture)
            snapshot = self._snapshot_memory() if established_forward_stream else None
            self.clock, self.detector = checked_clock, observed_detector
            result = super().process(sample)
        except Exception:
            # Exceptions must not leave a partially updated tracking state.
            self.tracker.reset()
            raise
        finally:
            self.clock, self.detector = original_clock, original_detector

        receipt = dict(preserved_after_deadline=False, restored_track_count=0,
                       observation_timestamps_renewed=False, control_authority=False)
        result['tracking_memory'] = receipt
        reason = result.get('reason')
        source_age = sample.capture_age_at_receive_s
        verified_deadline = (
            snapshot is not None and observed_detector.valid_return
            and source_age is not None and len(readings) >= 2
            and readings[0] >= sample.received_monotonic_s
            and result.get('status') == 'rejected'
            and reason in ('inference_deadline_missed', 'processing_deadline_missed')
            and result.get('inference_executed') is True
            and result.get('control_authority') is False
            and result.get('detections') == []
            and math.isfinite(result.get('age_at_finish_s', math.nan))
            and result['age_at_finish_s'] > self.max_frame_age_s
            and self.stream == (sample.stream_id, sample.clock_domain, sample.frame_id)
            and self.last_sequence == sample.sequence
            and self.last_capture == sample.capture_time_s)
        if not verified_deadline:
            return result

        tracks, anchors, last_time, next_id = snapshot
        # Count the verified elapsed age as well as capture spacing. A long
        # inference must not keep a descriptor alive beyond its observed life.
        checked_time = sample.capture_time_s + result['age_at_finish_s']
        retained = {track_id: track for track_id, track in tracks.items()
                    if 0 <= checked_time-track['time'] <= self.tracker.max_age_s}
        self.tracker.tracks = retained
        self.tracker.last_time = last_time if retained else None
        self.tracker.next_id = max(next_id, self.tracker.next_id)
        if hasattr(self.tracker, 'anchors'):
            self.tracker.anchors = {key: anchor for key, anchor in anchors.items()
                                    if key in retained}
        # A prior reprojection describes a different capture and is not a new
        # observation. prepared_frame remains the current frame, without a copy.
        if hasattr(self.tracker, 'last_reprojections'):
            self.tracker.last_reprojections = []
        receipt.update(preserved_after_deadline=bool(retained),
                       restored_track_count=len(retained), reason=reason,
                       age_checked_at_capture_time_s=float(checked_time),
                       max_observed_memory_age_s=float(self.tracker.max_age_s))
        return result
