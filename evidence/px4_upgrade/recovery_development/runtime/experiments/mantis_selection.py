"""Explicit, frame-bound person selection from camera observations only.

This is conservative temporary tracking, not person recognition. Ambiguity,
loss or a changed clothing descriptor latches a hold until another explicit
selection. Neither simulator actor IDs nor projected ground truth enter here.
"""
from __future__ import annotations

import math
from numbers import Integral
import numpy as np

from experiments.hybrid_target import TargetBridge, _finite

# Matches MANTIS_TRACK_MEMORY_S without importing the learned vision runtime.
# This bounds private reassociation memory, never observation/command validity.
_MAX_DEADLINE_RECOVERY_S = 3.


def _clothing_descriptor(image, box):
    """Bounded upper-body colour sample, robust to lighting and box jitter.

    A YOLO box can gain/lose the head between frames. Counting hard RGB bins in
    a narrow fixed crop then mistakes changing background/trouser fractions for
    changed clothing. Sample a broader upper torso, omit neutral background from
    the chromatic histogram, and interpolate cyclic hue bins. Neutral clothing
    uses its own soft brightness histogram rather than borrowing a colour cue.
    This remains a weak appearance check, not recognition or segmentation.
    """
    left, top, right, bottom = box
    width, height = right-left, bottom-top
    x0, x1 = math.floor(left+.2*width), math.ceil(left+.8*width)
    y0, y1 = math.floor(top+.15*height), math.ceil(top+.55*height)
    if x1 <= x0 or y1 <= y0:
        return None
    xs = np.linspace(x0, x1-1, min(32, x1-x0), dtype=int)
    ys = np.linspace(y0, y1-1, min(32, y1-y0), dtype=int)
    pixels = image[ys[:, None], xs[None, :]].reshape(-1, 3).astype(float)
    maximum, difference = pixels.max(axis=1), np.ptp(pixels, axis=1)
    chromatic = (difference >= 12.) & (difference/np.maximum(maximum, 1.) >= .18)
    if np.count_nonzero(chromatic) >= max(8, .1*len(pixels)):
        colours, delta = pixels[chromatic], difference[chromatic]
        dominant = np.argmax(colours, axis=1)
        hue = np.where(dominant == 0, (colours[:, 1]-colours[:, 2])/delta,
              np.where(dominant == 1, (colours[:, 2]-colours[:, 0])/delta+2.,
                       (colours[:, 0]-colours[:, 1])/delta+4.)) % 6.
        coordinate, bins, mode = hue*2., 12, 'chromatic'
        lower = np.floor(coordinate).astype(int)
        fraction = coordinate-lower
        histogram = (np.bincount(lower % bins, weights=1.-fraction, minlength=bins)
                     + np.bincount((lower+1) % bins, weights=fraction, minlength=bins))
    else:
        coordinate, bins, mode = pixels.mean(axis=1)*7./255., 8, 'neutral'
        lower = np.floor(coordinate).astype(int)
        fraction = coordinate-lower
        histogram = (np.bincount(lower, weights=1.-fraction, minlength=bins)
                     + np.bincount(np.minimum(lower+1, bins-1), weights=fraction, minlength=bins))
    return dict(histogram=histogram/histogram.sum(), mode=mode)


def _appearance_similarity(first, second):
    if first['mode'] != second['mode']:
        return 0.
    # Bhattacharyya affinity tolerates changing proportions of an existing
    # colour mixture. New colours still need support in the original anchor.
    return float(np.clip(np.sqrt(first['histogram']*second['histogram']).sum(), 0., 1.))


def _overlap_fraction(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0])) * max(
        0., min(a[3], b[3])-max(a[1], b[1]))
    return intersection / min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))


class SelectionGuard:
    """Drop-in TargetBridge update plus explicit select/clear/status methods.

    Call observe_frame(frame) immediately before vision.process(frame). Clicks
    must include the latest detector sequence, preventing a displayed old box
    from silently selecting a new occupant. The original clothing sample is
    fixed for each selection, never adapted to follow a replacement person.
    """
    def __init__(self, *, overlap_threshold=.18, appearance_threshold=.65):
        for value in (overlap_threshold, appearance_threshold):
            if not _finite(value) or not 0 < value <= 1:
                raise ValueError('Selection thresholds must be finite in (0,1]')
        self.overlap_threshold = float(overlap_threshold)
        self.appearance_threshold = float(appearance_threshold)
        self._bridge = TargetBridge(track_id=2**63-1)
        self.track_id = None
        self._image = self._image_time = None
        self._people = []
        self._features = {}
        self._sequence = self._capture = self._now = None
        self._max_age = .25
        self._anchor = None
        self._reason = 'selection_required'
        self._held = False
        self._previous_people = []
        self._last_similarity = None
        self._last_accepted_capture = None
        self._deadline_pending = False
        self._deadline_people = []
        self._deadline_limit_s = _MAX_DEADLINE_RECOVERY_S

    def observe_frame(self, frame):
        frame.validate()
        self._image = frame.rgb.copy()
        self._image_time = float(frame.capture_time_s)

    def _invalid(self, reason):
        observation = self._bridge._invalid(reason)
        observation['track_id'] = self.track_id
        return observation

    def _hold(self, reason):
        self._reason = reason
        self._deadline_pending = False
        self._deadline_people = []
        if self.track_id is not None:
            self._held = True
        return self._invalid(reason)

    def _verified_deadline(self, result, now_s, max_age_s, bridge_reason):
        """Recognize only a complete timeout receipt from MantisPerceptionPipeline.

        A reason string alone is insufficient. Source order/stream, actual
        execution, rejected empty observations, timing consistency and preserved
        pre-attempt association memory must all agree. This is a typed internal
        receipt contract, not authentication of an arbitrary external detector.
        """
        if (self._held or self.track_id is None or self._anchor is None
                or self._last_accepted_capture is None or not isinstance(result, dict)
                or bridge_reason not in ('perception_unavailable', 'stale_target_frame')):
            return False
        reason = result.get('reason')
        memory = result.get('tracking_memory')
        if (result.get('status') != 'rejected'
                or reason not in ('inference_deadline_missed', 'processing_deadline_missed')
                or result.get('inference_executed') is not True
                or result.get('control_authority') is not False
                or not isinstance(result.get('detections'), list) or result['detections']
                or not isinstance(memory, dict)
                or memory.get('preserved_after_deadline') is not True
                or memory.get('observation_timestamps_renewed') is not False
                or memory.get('control_authority') is not False or memory.get('reason') != reason):
            return False
        count = memory.get('restored_track_count')
        if (not isinstance(count, Integral) or isinstance(count, (bool, np.bool_))
                or not 1 <= count <= 256):
            return False
        capture, sequence = result.get('capture_time_s'), result.get('sequence')
        stream = tuple(result.get(k) for k in ('stream_id', 'clock_domain', 'frame_id'))
        if (self._bridge.stream_changed or stream != self._bridge.stream
                or not isinstance(sequence, Integral) or isinstance(sequence, (bool, np.bool_))
                or self._bridge.last_sequence is None or sequence <= self._bridge.last_sequence
                or not _finite(capture) or capture <= self._bridge.last_capture
                or capture > now_s+1e-9):
            return False
        start, finish, milliseconds = (result.get('age_at_start_s'), result.get('age_at_finish_s'),
                                       result.get('processing_ms'))
        checked, limit, received = (memory.get('age_checked_at_capture_time_s'),
                                    memory.get('max_observed_memory_age_s'),
                                    result.get('received_monotonic_s'))
        if (not all(_finite(v) for v in (start, finish, milliseconds, checked, limit, received))
                or not 0 <= start <= max_age_s or finish <= max_age_s or milliseconds <= 0
                or received < 0 or not 0 < limit <= _MAX_DEADLINE_RECOVERY_S
                or not math.isclose((finish-start)*1000., milliseconds, rel_tol=1e-7, abs_tol=1e-5)
                or not math.isclose(capture+finish, checked, rel_tol=0., abs_tol=1e-8)):
            return False
        bound = min(limit, self._deadline_limit_s)
        return 0 <= max(now_s, checked)-self._last_accepted_capture <= bound+1e-9

    def _wait_after_deadline(self, result):
        if not self._deadline_pending:
            self._deadline_people = [dict(person, bbox_xyxy=list(person['bbox_xyxy']))
                                     for person in self._people]
        self._deadline_pending = True
        self._deadline_limit_s = min(self._deadline_limit_s,
                                     result['tracking_memory']['max_observed_memory_age_s'])
        # These are source-order watermarks only. Do not renew the independent
        # last accepted observation time, clothing anchor or person snapshot.
        self._bridge.last_sequence = int(result['sequence'])
        self._bridge.last_capture = float(result['capture_time_s'])
        self._sequence = self._capture = None
        self._people, self._features = [], {}
        self._image = self._image_time = None
        self._reason = 'verified_deadline_waiting_for_fresh_observation'
        return self._invalid(self._reason)

    def _ambiguous(self, person, people=None):
        return any(other['track_id'] != person['track_id'] and
                   _overlap_fraction(person['bbox_xyxy'], other['bbox_xyxy']) >= self.overlap_threshold
                   for other in (self._people if people is None else people))

    def status(self):
        return dict(track_id=self.track_id, reason=self._reason, held=self._held or self._deadline_pending,
                    sequence=self._sequence, capture_time_s=self._capture,
                    selection_required=self.track_id is None or self._held,
                    people=[dict(**person, selectable=(not self._ambiguous(person) and
                            person['track_id'] in self._features)) for person in self._people],
                    appearance_similarity=self._last_similarity,
                    appearance_metric='soft_upper_body_colour_bhattacharyya',
                    recovery_pending=self._deadline_pending,
                    identity_limit='temporary visual track; clothing is not identity')

    def clear(self):
        self.track_id = None
        self._bridge.track_id = 2**63-1
        self._anchor = None
        self._last_similarity = None
        self._held = False
        self._reason = 'selection_required'
        self._last_accepted_capture = None
        self._deadline_pending = False
        self._deadline_people = []
        self._deadline_limit_s = _MAX_DEADLINE_RECOVERY_S
        return self.status()

    def select(self, track_id, sequence, now_s=None):
        if (not isinstance(track_id, Integral) or isinstance(track_id, (bool, np.bool_)) or track_id < 1
                or not isinstance(sequence, Integral) or isinstance(sequence, (bool, np.bool_)) or sequence < 0):
            raise ValueError('Selection requires positive track ID and nonnegative sequence')
        now_s = self._now if now_s is None else now_s
        if not _finite(now_s) or now_s < 0:
            raise ValueError('Selection time must be finite and nonnegative')
        if (self._sequence is None or sequence != self._sequence or self._capture is None
                or now_s < self._capture or now_s-self._capture > self._max_age + 1e-9):
            raise ValueError('Selection frame is stale; select from the latest camera frame')
        if self._bridge.stream_changed:
            raise ValueError('Changed camera stream requires a fresh selection guard')
        selected = next((person for person in self._people if person['track_id'] == track_id), None)
        if selected is None or track_id not in self._features:
            raise ValueError('Selected person must be currently observed with camera appearance')
        if self._ambiguous(selected):
            raise ValueError('People overlap; wait for an unambiguous view before selecting')
        self.track_id = self._bridge.track_id = int(track_id)
        feature = self._features[track_id]
        self._anchor = dict(mode=feature['mode'], histogram=feature['histogram'].copy())
        self._last_similarity = None
        self._held = False
        self._last_accepted_capture = float(self._capture)
        self._deadline_pending = False
        self._deadline_people = []
        self._deadline_limit_s = _MAX_DEADLINE_RECOVERY_S
        self._reason = 'selected_waiting_for_next_frame'
        return self.status()

    def update(self, result, now_s, image_hw, max_age_s=.25):
        observation = self._bridge.update(result, now_s, image_hw, max_age_s)
        self._last_similarity = None
        self._now, self._max_age = float(now_s), float(max_age_s)
        if observation['reason'] not in ('observed', 'target_not_observed'):
            if self._verified_deadline(result, now_s, max_age_s, observation['reason']):
                return self._wait_after_deadline(result)
            self._people, self._features = [], {}
            if self._held:
                return self._invalid(self._reason)
            return self._hold(observation['reason'])
        recovering = self._deadline_pending
        self._previous_people = self._deadline_people if recovering else self._people
        self._sequence = int(result['sequence'])
        self._capture = float(result['capture_time_s'])
        self._people = [dict(track_id=int(d['track_id']), bbox_xyxy=list(d['bbox_xyxy']),
                             confidence=float(d['confidence'])) for d in result['detections']
                        if d['class_id'] == 0 and d['label'] == 'person']
        self._features = {}
        if (self._image is not None and self._image.shape[:2] == tuple(image_hw)
                and abs(self._image_time-self._capture) < 1e-9):
            self._features = {p['track_id']: _clothing_descriptor(self._image, p['bbox_xyxy'])
                              for p in self._people}
            self._features = {tid: f for tid, f in self._features.items() if f is not None}
        if self.track_id is None:
            self._reason = 'selection_required'
            return self._invalid(self._reason)
        if self._held:
            return self._invalid(self._reason)
        if recovering:
            elapsed = result.get('age_at_finish_s')
            if not _finite(elapsed) or not 0 <= elapsed <= max_age_s+1e-9:
                return self._hold('recovery_observation_not_fresh')
            if max(now_s, self._capture+elapsed)-self._last_accepted_capture > self._deadline_limit_s+1e-9:
                return self._hold('deadline_recovery_window_expired')
        if not observation['valid']:
            return self._hold('selected_person_lost')
        selected = next(p for p in self._people if p['track_id'] == self.track_id)
        if self._ambiguous(selected):
            return self._hold('ambiguous_people_overlap')
        if recovering and self._ambiguous(selected, self._deadline_people):
            return self._hold('ambiguous_identity_after_deadline')
        current_ids = {p['track_id'] for p in self._people}
        vanished = [p for p in self._previous_people if p['track_id'] not in current_ids]
        if self._ambiguous(selected, vanished):
            return self._hold('ambiguous_person_occlusion')
        feature = self._features.get(self.track_id)
        if feature is None:
            return self._hold('current_appearance_unavailable')
        similarity = _appearance_similarity(self._anchor, feature)
        self._last_similarity = similarity
        if similarity < self.appearance_threshold:
            return self._hold('selected_appearance_changed')
        if recovering and any(tid != self.track_id and
                _appearance_similarity(self._anchor, other) >= self.appearance_threshold
                for tid, other in self._features.items()):
            return self._hold('ambiguous_clothing_after_deadline')
        self._last_accepted_capture = self._capture
        self._deadline_pending = False
        self._deadline_people = []
        self._deadline_limit_s = _MAX_DEADLINE_RECOVERY_S
        self._reason = 'observed'
        observation['appearance_similarity'] = similarity
        return observation
