"""Conservative registered-depth surface measurement for a 3D person box.

Only observed RGB-D calibration, timestamps and a detector box are inputs. The
measurement describes visible foreground inside that box; it cannot establish
person identity or distinguish a foreground occluder from the selected person.
No simulator geometry, semantic relabeling, depth filling or threshold inflation
is used. The original guidance's 0.8 valid-depth / 0.35 m spread limits remain.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.ndimage import label

from experiments.flight_contracts import DEPTH_MAX_AGE_S

# Frozen using the first development walking sequence, before held-out runs.
TORSO_ROI_FRACTION_XYXY = (.25, .18, .75, .58)
MIN_VALID_DEPTH_FRACTION = .8
MIN_FOREGROUND_SUPPORT_FRACTION = .25
MIN_COMPONENT_PIXELS = 32
MIN_COMPONENT_COHERENCE = .8
DEPTH_CLUSTER_GAP_M = .12
MAX_SELECTED_SPREAD_M = .35
NEAR_LAYER_AMBIGUITY_M = .6
MAX_RGB_DEPTH_SKEW_S = .04


def _finite_scalar(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


def diagnose_foreground_surface(frame, bbox, *, now_s=None, depth_time_s=None):
    """Return an auditable diagnostic containing ``surface`` or a rejection.

    ``frame`` follows CameraFrame's synchronized RGB/optical-depth contract.
    Supply ``depth_time_s`` when a source has a separate depth timestamp, and
    ``now_s`` when measuring later than capture, both in the capture clock.
    Omission evaluates the synchronized capture itself; it does not authorize
    later use. The flight command path must still enforce observation expiry.
    """
    result = dict(valid=False, reason='invalid_frame', surface=None)
    try:
        frame.validate()
    except (ValueError, TypeError, AttributeError):
        return result
    if not frame.registration_verified:
        return dict(result, reason='depth_registration_unverified')
    capture = frame.capture_time_s
    now = capture if now_s is None else now_s
    depth_time = capture if depth_time_s is None else depth_time_s
    if not all(_finite_scalar(value) and value >= 0 for value in (capture, now, depth_time)):
        return dict(result, reason='invalid_depth_timing')
    if capture > now + 1e-9 or depth_time > now + 1e-9:
        return dict(result, reason='future_depth_or_capture')
    if max(now-capture, now-depth_time) > DEPTH_MAX_AGE_S + 1e-9:
        return dict(result, reason='stale_depth_or_capture')
    if abs(capture-depth_time) > MAX_RGB_DEPTH_SKEW_S + 1e-9:
        return dict(result, reason='rgb_depth_not_synchronized')
    try:
        box = np.asarray(bbox, dtype=float)
    except (ValueError, TypeError):
        return dict(result, reason='invalid_box')
    if box.shape != (4,) or not np.isfinite(box).all() or min(box[2:]-box[:2]) <= 0:
        return dict(result, reason='invalid_box')
    height, width = frame.depth_m.shape
    bw, bh = box[2:]-box[:2]
    rx1, ry1, rx2, ry2 = TORSO_ROI_FRACTION_XYXY
    left, top = max(0, math.floor(box[0]+rx1*bw)), max(0, math.floor(box[1]+ry1*bh))
    right, bottom = min(width, math.ceil(box[0]+rx2*bw)), min(height, math.ceil(box[1]+ry2*bh))
    if right-left < 4 or bottom-top < 4 or (right-left)*(bottom-top) < 64:
        return dict(result, reason='insufficient_torso_pixels')
    # A deterministic grid bounds large camera frames; support fractions refer
    # to these actual sampled pixels. No invalid or background pixel is filled.
    step = max(1, math.ceil(math.sqrt((right-left)*(bottom-top)/4096)))
    region = frame.depth_m[top:bottom:step, left:right:step]
    valid = np.isfinite(region) & (region >= .3) & (region <= 20.)
    fraction = float(valid.mean())
    result.update(roi_xyxy=[left, top, right, bottom], sampling_step_px=step,
                  valid_depth_fraction=fraction, sampled_pixels=int(region.size))
    if fraction < MIN_VALID_DEPTH_FRACTION or np.count_nonzero(valid) < MIN_COMPONENT_PIXELS:
        return dict(result, reason='insufficient_valid_depth')
    values = np.sort(region[valid].astype(float))
    groups = np.split(values, np.flatnonzero(np.diff(values) > DEPTH_CLUSTER_GAP_M)+1)
    groups = [group for group in groups if len(group)]
    support_min = max(MIN_COMPONENT_PIXELS, math.ceil(MIN_FOREGROUND_SUPPORT_FRACTION*region.size))
    meaningful_min = max(8, math.ceil(.05*region.size))
    candidate = None
    for index, group in enumerate(groups):
        if len(group) < meaningful_min:
            # Isolated tiny near specks cannot replace a supported surface.
            continue
        if len(group) < support_min:
            return dict(result, reason='unsupported_nearer_depth_layer')
        mask = valid & (region >= group[0]) & (region <= group[-1])
        components, count = label(mask, structure=np.ones((3,3), dtype=int))
        areas = np.bincount(components.ravel(), minlength=count+1)
        selected_id = int(np.argmax(areas[1:])+1)
        selected = components == selected_id
        selected_count = int(areas[selected_id])
        coherence = selected_count/len(group)
        if selected_count < support_min or coherence < MIN_COMPONENT_COHERENCE:
            return dict(result, reason='incoherent_nearer_depth_layer')
        candidate = (index, selected, coherence)
        break
    if candidate is None:
        return dict(result, reason='no_supported_foreground_layer')
    index, selected, coherence = candidate
    z = region[selected].astype(float)
    median_z = float(np.median(z))
    # Distinct, substantially supported surfaces at similar distances are not
    # resolved by choosing whichever happens to be fractionally closer.
    for other in groups[index+1:]:
        if len(other) >= support_min and float(np.median(other))-median_z < NEAR_LAYER_AMBIGUITY_M:
            return dict(result, reason='ambiguous_nearby_depth_layers')
    spread = float(np.quantile(z,.9)-np.quantile(z,.1))
    if spread > MAX_SELECTED_SPREAD_M:
        return dict(result, reason='foreground_depth_spread_unsupported',
                    foreground_depth_spread_p90_p10_m=spread)
    rows, columns = np.where(selected)
    fx, fy, cx, cy = frame.intrinsics
    x = (columns*step+left-cx)*z/fx
    y = (rows*step+top-cy)*z/fy
    support = float(np.count_nonzero(selected)/region.size)
    surface = dict(surface_optical_z_m=median_z,
        surface_range_m=float(np.median(np.sqrt(x*x+y*y+z*z))),
        surface_xyz_optical_m=[float(np.median(x)),float(np.median(y)),median_z],
        valid_depth_fraction=fraction, depth_spread_p90_p10_m=spread,
        foreground_support_fraction=support, foreground_component_coherence=float(coherence),
        foreground_pixel_count=int(len(z)), sampled_pixels=int(region.size),
        roi_xyxy=[left,top,right,bottom], sampling_step_px=step,
        meaning='coherent_visible_foreground_in_torso_box_not_identity_or_object_extent',
        reason='supported_coherent_visible_foreground', frame_id='primary_optical',
        capture_time_s=float(capture), depth_time_s=float(depth_time))
    return dict(result, valid=True, reason=surface['reason'], surface=surface)


def coherent_foreground_surface(frame, bbox, *, now_s=None, depth_time_s=None):
    """Return a supported sensor-only surface dictionary, or ``None``."""
    return diagnose_foreground_surface(frame,bbox,now_s=now_s,depth_time_s=depth_time_s)['surface']
