"""Camera detections, short-lived tracks and depth advisories; never flight commands."""
from __future__ import annotations
from dataclasses import dataclass
import math
import time
from typing import Callable
import numpy as np
from .detector import Detection, validate_rgb_image


@dataclass(frozen=True)
class CameraSample:
    image_rgb: np.ndarray
    capture_time_s: float
    received_monotonic_s: float
    sequence: int
    stream_id: str
    clock_domain: str
    frame_id: str
    capture_age_at_receive_s: float | None = None
    depth_m: np.ndarray | None = None
    depth_time_s: float | None = None
    camera_intrinsics: tuple[float, float, float, float] | None = None
    registration_verified: bool = False

    def validate(self):
        a = self.image_rgb
        if not isinstance(a, np.ndarray) or a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] != 3:
            raise ValueError('image_rgb must be uint8 HWC with exactly three RGB channels')
        if min(a.shape[:2]) < 2 or a.shape[0] * a.shape[1] > 8_500_000:
            raise ValueError('camera image has invalid dimensions or exceeds 8.5M pixels')
        for value in (self.capture_time_s, self.received_monotonic_s):
            if not math.isfinite(value) or value < 0:
                raise ValueError('camera timestamps must be finite and nonnegative')
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError('sequence must be a nonnegative integer')
        for value in (self.stream_id, self.clock_domain, self.frame_id):
            if not isinstance(value, str) or not 1 <= len(value) <= 256:
                raise ValueError('camera stream, clock and frame identifiers are required')
        if self.capture_age_at_receive_s is not None and (not math.isfinite(self.capture_age_at_receive_s) or self.capture_age_at_receive_s < 0):
            raise ValueError('capture age must be finite and nonnegative or absent')
        if type(self.registration_verified) is not bool:
            raise ValueError('registration_verified must be boolean')
        if self.depth_m is not None:
            if not isinstance(self.depth_m, np.ndarray) or self.depth_m.ndim != 2 or self.depth_m.shape != a.shape[:2] or self.depth_m.dtype.kind != 'f':
                raise ValueError('depth_m requires floating-point optical-Z meters on the RGB pixel grid')
            if self.depth_time_s is None or not math.isfinite(self.depth_time_s) or self.depth_time_s < 0:
                raise ValueError('depth timestamp required when depth is present')
        if self.camera_intrinsics is not None:
            v = self.camera_intrinsics
            if len(v) != 4 or not all(math.isfinite(x) for x in v) or v[0] <= 0 or v[1] <= 0:
                raise ValueError('intrinsics require finite fx,fy,cx,cy and positive focal lengths')
            if not 0 <= v[2] < a.shape[1] or not 0 <= v[3] < a.shape[0]:
                raise ValueError('principal point must be inside the registered image')


def iou(a, b):
    w = max(0., min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0., min(a[3], b[3]) - max(a[1], b[1]))
    intersection = w*h
    denominator = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / denominator if denominator > 0 else 0.


def clothing_histogram(image_rgb, bbox):
    """Coarse clothing color only; at most 32x32 source pixels, no image history.

    Joint RGB histogram, four bins per channel. The central x20-80%, y25-60%
    rectangle targets upper-body clothing without using face embeddings. Similar
    clothing and occlusions remain ambiguous; this cannot establish identity.
    """
    h,w = image_rgb.shape[:2]
    left,top,right,bottom = bbox
    bw,bh = right-left,bottom-top
    x0,x1 = max(0,math.floor(left+.2*bw)),min(w,math.ceil(left+.8*bw))
    y0,y1 = max(0,math.floor(top+.25*bh)),min(h,math.ceil(top+.6*bh))
    if x1 <= x0 or y1 <= y0:
        return None
    ys = np.linspace(y0,y1-1,min(32,y1-y0),dtype=int)
    xs = np.linspace(x0,x1-1,min(32,x1-x0),dtype=int)
    pixels = image_rgb[ys[:,None],xs[None,:]].astype(np.int32)//64
    histogram = np.bincount((pixels[...,0]*16+pixels[...,1]*4+pixels[...,2]).ravel(),minlength=64).astype(float)
    return histogram/histogram.sum()


class ShortTermTracker:
    """Greedy class-aware IoU with optional clothing gate; no identity inference."""
    def __init__(self, minimum_iou=.3, max_age_s=.5, max_tracks=64, appearance_threshold=.65):
        if not 0 < minimum_iou <= 1 or not math.isfinite(max_age_s) or max_age_s <= 0 or type(max_tracks) is not int or not 1 <= max_tracks <= 256:
            raise ValueError('invalid tracker limits')
        if appearance_threshold is not None:
            if (isinstance(appearance_threshold,(bool,np.bool_)) or not isinstance(appearance_threshold,(int,float,np.integer,np.floating))
                or not math.isfinite(appearance_threshold) or not 0 < appearance_threshold <= 1):
                raise ValueError('appearance_threshold must be finite in (0,1] or None')
            appearance_threshold = float(appearance_threshold)
        self.minimum_iou, self.max_age_s, self.max_tracks = minimum_iou, max_age_s, max_tracks
        self.appearance_threshold = appearance_threshold
        self.tracks, self.next_id, self.last_time = {}, 1, None

    def reset(self):
        self.tracks.clear()
        self.last_time = None
        # IDs continue increasing; resets cannot silently reuse an old identity.

    def update(self, detections, timestamp, image_rgb=None):
        if not math.isfinite(timestamp) or timestamp < 0 or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError('tracking requires strictly increasing timestamps')
        if image_rgb is not None:
            validate_rgb_image(image_rgb)
            h,w = image_rgb.shape[:2]
            # Validate all supplied boxes before selecting the bounded work set.
            # Pipeline callers already validate them; direct RGB callers must
            # not turn an off-image/invalid rectangle into an appearance sample.
            if not isinstance(detections,(list,tuple)) or len(detections) > 1000:
                raise ValueError('invalid tracker detection collection')
            for det in detections:
                if not isinstance(det,Detection):
                    raise ValueError('invalid tracker detection contract')
                box = det.bbox_xyxy
                try:
                    valid = (len(box) == 4 and all(math.isfinite(x) for x in box)
                             and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h
                             and math.isfinite(det.confidence) and 0 <= det.confidence <= 1)
                except (TypeError,ValueError):
                    valid = False
                if not valid:
                    raise ValueError('invalid tracker detection contract')
        use_appearance = image_rgb is not None and self.appearance_threshold is not None
        self.last_time = timestamp
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp-v['time'] <= self.max_age_s}
        indexed = sorted(enumerate(detections), key=lambda p: (-p[1].confidence, p[0]))[:self.max_tracks]
        features = {index: clothing_histogram(image_rgb,det.bbox_xyxy) if use_appearance and det.class_id == 0 else None
                    for index,det in indexed}
        candidates = []
        for index, det in indexed:
            for tid, old in self.tracks.items():
                overlap = iou(det.bbox_xyxy, old['bbox']) if det.class_id == old['class_id'] else 0
                if overlap >= self.minimum_iou:
                    similarity = 1.
                    if use_appearance and det.class_id == 0:
                        feature,previous = features[index],old['appearance']
                        similarity = float(np.sqrt(feature*previous).sum()) if feature is not None and previous is not None else 0.
                        if similarity < self.appearance_threshold:
                            continue
                    candidates.append((-overlap*similarity, index, tid))
        assigned, used = {}, set()
        for _, index, tid in sorted(candidates):
            if index not in assigned and tid not in used:
                assigned[index] = tid
                used.add(tid)
        result = []
        for index, det in indexed:
            tid = assigned.get(index)
            if tid is None:
                if len(self.tracks) >= self.max_tracks:
                    evictable = [(v['time'], k) for k, v in self.tracks.items() if k not in used]
                    if not evictable:
                        continue
                    del self.tracks[min(evictable)[1]]
                tid, self.next_id = self.next_id, self.next_id+1
            used.add(tid)
            # Only the last observed histogram is retained, never the frame or
            # predicted boxes. Missing RGB keeps the legacy two-argument path.
            self.tracks[tid] = {'bbox': det.bbox_xyxy, 'class_id': det.class_id, 'time': timestamp,
                                'appearance': features[index]}
            result.append({'track_id': tid, 'class_id': det.class_id, 'label': det.label,
                           'confidence': float(det.confidence), 'bbox_xyxy': [float(x) for x in det.bbox_xyxy]})
        return result


def depth_status(sample, max_skew_s=.04):
    if sample.depth_m is None:
        return 'depth_unavailable'
    if not sample.registration_verified:
        return 'depth_registration_unverified'
    if sample.camera_intrinsics is None:
        return 'camera_calibration_unavailable'
    if abs(sample.capture_time_s-sample.depth_time_s) > max_skew_s + 1e-9:
        return 'rgb_depth_not_synchronized'
    return ''


def surface_measurement(sample, bbox):
    """Central bounding-box depth is a visible-surface estimate, not object extent."""
    if depth_status(sample):
        return None
    height, width = sample.image_rgb.shape[:2]
    x1,y1,x2,y2 = bbox
    x1,x2 = max(0,int(math.floor(x1+.3*(x2-x1)))), min(width,int(math.ceil(x2-.3*(x2-x1))))
    y1,y2 = max(0,int(math.floor(y1+.3*(y2-y1)))), min(height,int(math.ceil(y2-.3*(y2-y1))))
    if x2 <= x1 or y2 <= y1:
        return None
    step = max(1,math.ceil(math.sqrt((y2-y1)*(x2-x1)/4096)))
    region = sample.depth_m[y1:y2:step,x1:x2:step]
    valid = np.isfinite(region) & (region >= .15) & (region <= 80)
    fraction = float(valid.mean())
    if valid.sum() < 4 or fraction < .6:
        return None
    rows,cols = np.where(valid)
    z = region[valid].astype(float)
    fx,fy,cx,cy = sample.camera_intrinsics
    x,y = (cols*step+x1-cx)*z/fx, (rows*step+y1-cy)*z/fy
    q10,q90 = np.quantile(z,[.1,.9])
    return {'surface_optical_z_m': float(np.median(z)),
            'surface_range_m': float(np.median(np.sqrt(x*x+y*y+z*z))),
            'surface_xyz_optical_m': [float(np.median(x)),float(np.median(y)),float(np.median(z))],
            'valid_depth_fraction': fraction, 'depth_spread_p90_p10_m': float(q90-q10),
            'sampling_step_px': step,
            'meaning': 'central_box_visible_surface_only', 'frame_id': sample.frame_id}


def forward_depth_advisory(sample, near_distance_m=3.):
    """Sample a forward image region. This is not collision-free corridor proof."""
    reason = depth_status(sample)
    if reason:
        return {'state': 'unknown', 'reason': reason}
    if not math.isfinite(near_distance_m) or near_distance_m <= 0:
        raise ValueError('near distance must be finite and positive')
    h,w = sample.depth_m.shape
    region = sample.depth_m[int(h*.2):max(int(h*.8),int(h*.2)+1), int(w*.2):max(int(w*.8),int(w*.2)+1)]
    step = max(1,math.ceil(math.sqrt(region.size/16384)))
    region = region[::step,::step]
    valid = np.isfinite(region) & (region >= .15) & (region <= 80)
    fraction = float(valid.mean())
    if valid.sum() < 4 or fraction < .8:
        return {'state': 'unknown', 'reason': 'insufficient_forward_depth', 'valid_depth_fraction': fraction}
    distance = float(np.quantile(region[valid], .1))
    return {'state': 'near_surface' if distance <= near_distance_m else 'no_near_surface_in_sampled_region',
            'p10_optical_z_m': distance, 'threshold_m': near_distance_m,
            'valid_depth_fraction': fraction, 'region_fraction_xyxy': [.2,.2,.8,.8],
            'sampling_step_px': step,
            'flight_corridor_clear': None, 'control_authority': False}


class PerceptionPipeline:
    def __init__(self, detector, *, max_frame_age_s=.3, clock: Callable[[], float]=time.monotonic, appearance_tracking=True):
        if not math.isfinite(max_frame_age_s) or max_frame_age_s <= 0:
            raise ValueError('max_frame_age_s must be positive and finite')
        if type(appearance_tracking) is not bool:
            raise ValueError('appearance_tracking must be boolean')
        self.detector, self.max_frame_age_s, self.clock = detector, max_frame_age_s, clock
        self.appearance_tracking = appearance_tracking
        self.tracker = ShortTermTracker(appearance_threshold=.65 if appearance_tracking else None)
        self.stream = self.last_sequence = self.last_capture = None

    def process(self, sample: CameraSample):
        sample.validate()
        started = self.clock()
        result = {'schema_version': 1, 'sequence': sample.sequence, 'stream_id': sample.stream_id,
                  'clock_domain': sample.clock_domain, 'frame_id': sample.frame_id,
                  'capture_time_s': sample.capture_time_s, 'received_monotonic_s': sample.received_monotonic_s,
                  'status': 'unknown', 'inference_executed': False, 'control_authority': False,
                  'detections': [], 'depth_advisory': {'state': 'unknown'}, 'processing_ms': 0.}
        stream_key = (sample.stream_id,sample.clock_domain,sample.frame_id)
        if stream_key != self.stream:
            self.tracker.reset()
            self.stream, self.last_sequence, self.last_capture = stream_key, None, None
        if self.last_sequence is not None and (sample.sequence <= self.last_sequence or sample.capture_time_s <= self.last_capture):
            self.tracker.reset()
            result.update(status='rejected', reason='duplicate_or_reordered_frame')
            return result
        local_age = started-sample.received_monotonic_s
        source_age = sample.capture_age_at_receive_s
        age = local_age + (source_age or 0.)
        result['age_at_start_s'] = age
        if local_age < -1e-6 or age > self.max_frame_age_s:
            self.tracker.reset()
            result.update(status='rejected', reason='future_receive_time' if local_age < 0 else 'stale_camera_frame')
            return result
        self.last_sequence, self.last_capture = sample.sequence, sample.capture_time_s
        try:
            detections = self.detector.detect(sample.image_rgb)
        except Exception as exc:
            self.tracker.reset()
            result.update(status='detector_error', reason=f'{type(exc).__name__}: {exc}')
            result['processing_ms'] = max(0.,(self.clock()-started)*1000)
            return result
        finished = self.clock()
        elapsed = finished-started
        result.update(inference_executed=True, processing_ms=elapsed*1000, age_at_finish_s=age+elapsed)
        if elapsed < 0 or age+elapsed > self.max_frame_age_s:
            self.tracker.reset()
            result.update(status='rejected', reason='inference_deadline_missed')
            return result
        h,w = sample.image_rgb.shape[:2]
        if not isinstance(detections, (list, tuple)) or len(detections) > 1000:
            self.tracker.reset()
            result.update(status='detector_error', reason='invalid_detection_collection')
            return result
        for det in detections:
            if not isinstance(det, Detection):
                self.tracker.reset()
                result.update(status='detector_error', reason='invalid_detection_contract')
                return result
            box = det.bbox_xyxy
            if (len(box) != 4 or not all(math.isfinite(x) for x in box)
                or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h
                or not math.isfinite(det.confidence) or not 0 <= det.confidence <= 1):
                self.tracker.reset()
                result.update(status='detector_error', reason='invalid_detection_contract')
                return result
        tracks = (self.tracker.update(detections,sample.capture_time_s,sample.image_rgb) if self.appearance_tracking
                  else self.tracker.update(detections,sample.capture_time_s))
        result.update(status='ok' if source_age is not None else 'timing_unverified', detections=tracks,
                      detection_count=len(detections), tracked_detection_count=len(tracks))
        if source_age is None:
            result['depth_advisory'] = {'state': 'unknown','reason': 'capture_age_at_receive_unknown'}
        else:
            result['depth_advisory'] = forward_depth_advisory(sample)
            for track in tracks:
                track['surface_measurement'] = surface_measurement(sample,track['bbox_xyxy'])
        # Tracking and depth work also consume the frame's deadline. A fast
        # network must not make a result fresh after slow downstream processing.
        total_elapsed = self.clock()-started
        result.update(processing_ms=total_elapsed*1000,age_at_finish_s=age+total_elapsed)
        if total_elapsed < 0 or age+total_elapsed > self.max_frame_age_s:
            self.tracker.reset()
            result.update(status='rejected',reason='processing_deadline_missed',detections=[],
                          tracked_detection_count=0,depth_advisory={'state':'unknown'})
        return result
