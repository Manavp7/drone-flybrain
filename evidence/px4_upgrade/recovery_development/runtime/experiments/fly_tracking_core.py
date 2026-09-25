"""Pure, offline box propagation from Flyvis flow; no detector or fallback.

Boxes use continuous xyxy coordinates with half-open right/bottom bounds. A
manually initialized, fixed-size box can move only through the supplied flow
mapping. Once evidence is insufficient or a candidate leaves the image, its
last accepted position is retained permanently.
"""
from __future__ import annotations

import numpy as np

IMAGE_SIDE = 391
KERNEL_RADIUS = 6
TRAINING_HEIGHT = 436
TRAINING_FPS = 24
KERNEL_AREA = 169
RIDGE = 1e-8


def _finite_array(value, name):
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numbers")
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _centers(value):
    centers = _finite_array(value, "centers_rc")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_rc must have shape [N,2]")
    return centers


def _box(value):
    box = _finite_array(value, "box_xyxy")
    if box.shape != (4,) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("box_xyxy must have shape [4] and positive width and height")
    return box


def _flow(value, count):
    flow = _finite_array(value, "flow_xy")
    if flow.shape != (2, count):
        raise ValueError("flow_xy must have shape [2,N] matching centers_rc")
    return flow


def supported_indices(centers_rc, box_xyxy, image_side=IMAGE_SIDE, min_support=9):
    """Return interior receptors in the box, or no indices when too few exist.

    The six-pixel margin keeps each 13x13 kernel inside the image. Receptors
    outside the manually selected region are never substituted for support.
    """
    centers = _centers(centers_rc)
    box = _box(box_xyxy)
    side = _positive_integer(image_side, "image_side")
    minimum = _positive_integer(min_support, "min_support")
    if side < 2 * KERNEL_RADIUS + 1:
        raise ValueError("image_side must fit a 13x13 kernel")
    row, col = centers.T
    valid = ((row >= KERNEL_RADIUS) & (row <= side - KERNEL_RADIUS - 1)
             & (col >= KERNEL_RADIUS) & (col <= side - KERNEL_RADIUS - 1)
             & (col >= box[0]) & (col < box[2])
             & (row >= box[1]) & (row < box[3]))
    indices = np.flatnonzero(valid)
    return indices if len(indices) >= minimum else np.empty(0, dtype=np.int64)


def pool_flow(flow_xy, centers_rc, box):
    """Return the componentwise median and indices of at least nine receptors."""
    centers = _centers(centers_rc)
    flow = _flow(flow_xy, len(centers))
    indices = supported_indices(centers, box)
    if len(indices) < 9:
        raise ValueError("At least nine interior receptors must support the box")
    return np.median(flow[:, indices], axis=1), indices


def raw_flow_mapping():
    """Invert the training scale into 391-square pixels/second, without fitting.

    Flyvis targets sum 13x13 samples of [dx,-dy]/436 from 24-FPS Sintel.
    This mapping is a transfer assumption for the local average motion of an
    interior receptor, not calibrated physical velocity or object recognition.
    """
    scale = TRAINING_HEIGHT * TRAINING_FPS / KERNEL_AREA
    return np.asarray([[scale, 0.], [0., -scale], [0., 0.]], dtype=np.float64)


class FlowBoxTracker:
    """Fixed-size flow-only tracker with permanent loss and no reinitialization."""

    def __init__(self, centers_rc, initial_box, mapping, min_support=9):
        self.centers_rc = _centers(centers_rc).copy()
        self.box = _box(initial_box).copy()
        self.mapping = _finite_array(mapping, "mapping").copy()
        self.min_support = _positive_integer(min_support, "min_support")
        if self.mapping.shape != (3, 2):
            raise ValueError("mapping must have shape [3,2]")
        if not self._inside(self.box):
            raise ValueError("initial_box must be inside the 391-square image")
        self.status = "tracking"

    @staticmethod
    def _inside(box):
        return bool(np.all(box[:2] >= 0) and np.all(box[2:] <= IMAGE_SIDE))

    def _result(self, indices=None, pooled=None, velocity=None):
        return {"box_xyxy": self.box.tolist(), "status": self.status,
                "selected_indices": [] if indices is None else indices.tolist(),
                "pooled_flow": None if pooled is None else pooled.tolist(),
                "velocity_xy": None if velocity is None else velocity.tolist()}

    def step(self, flow_xy, dt):
        flow = _flow(flow_xy, len(self.centers_rc))
        elapsed = _finite_array(dt, "dt")
        if elapsed.ndim != 0 or elapsed.item() <= 0:
            raise ValueError("dt must be a finite positive scalar")
        if self.status != "tracking":
            return self._result()
        indices = supported_indices(self.centers_rc, self.box, min_support=self.min_support)
        if len(indices) < self.min_support:
            self.status = "insufficient_support"
            return self._result()
        pooled = np.median(flow[:, indices], axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            velocity = np.r_[pooled, 1.] @ self.mapping
            delta = velocity * float(elapsed)
            candidate = self.box + np.tile(delta, 2)
        if not np.isfinite(velocity).all() or not np.isfinite(candidate).all():
            raise ValueError("Flow mapping or integration produced nonfinite values")
        if not self._inside(candidate):
            self.status = "outside_frame"
        else:
            self.box = candidate
        return self._result(indices, pooled, velocity)


def fit_affine_flow(pooled, velocities):
    """Fit a 2D affine velocity readout from a separately chosen calibration set.

    No temporal alignment, lag selection or evaluation data is consumed here.
    A tiny ridge applies to the two slopes; the intercept is unpenalized.
    Rank-deficient calibration is rejected instead of inventing a mapping.
    """
    inputs = _finite_array(pooled, "pooled")
    targets = _finite_array(velocities, "velocities")
    if inputs.ndim != 2 or inputs.shape[1] != 2 or targets.shape != inputs.shape or len(inputs) < 3:
        raise ValueError("Calibration requires matching [N,2] arrays with N >= 3")
    design = np.column_stack((inputs, np.ones(len(inputs))))
    rank = int(np.linalg.matrix_rank(design))
    if rank != 3:
        raise ValueError("Calibration must independently constrain both slopes and the intercept")
    regularizer = np.sqrt(RIDGE) * np.diag([1., 1., 0.])
    mapping = np.linalg.lstsq(np.vstack((design, regularizer)),
                              np.vstack((targets, np.zeros((3, 2)))), rcond=None)[0]
    residual = design @ mapping - targets
    diagnostics = {"calibration_samples": len(inputs), "design_rank": rank,
                   "design_condition_number": float(np.linalg.cond(design)),
                   "ridge_slopes_only": RIDGE,
                   "calibration_rmse_px_per_s": float(np.sqrt(np.mean(residual**2))),
                   "calibration_max_vector_error_px_per_s": float(np.linalg.norm(residual, axis=1).max())}
    if not np.isfinite(mapping).all() or not all(np.isfinite(v) for v in diagnostics.values()):
        raise ValueError("Calibration produced nonfinite outputs")
    return mapping, diagnostics


def _box_geometry(pred, truth):
    prediction = _finite_array(pred, "pred")
    target = _finite_array(truth, "truth")
    if prediction.ndim != 2 or prediction.shape[1] != 4 or target.shape != prediction.shape or not len(target):
        raise ValueError("Expected matching nonempty [T,4] box arrays")
    for boxes in (prediction, target):
        if np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError("Every box must have positive width and height")
    error = np.linalg.norm((prediction[:, :2] + prediction[:, 2:]
                            - target[:, :2] - target[:, 2:]) / 2, axis=1)
    overlap = np.maximum(0., np.minimum(prediction[:, 2:], target[:, 2:])
                          - np.maximum(prediction[:, :2], target[:, :2]))
    intersection = np.prod(overlap, axis=1)
    union = (np.prod(prediction[:, 2:] - prediction[:, :2], axis=1)
             + np.prod(target[:, 2:] - target[:, :2], axis=1) - intersection)
    iou = intersection / union
    return {"mean_center_error": float(error.mean()), "final_center_error": float(error[-1]),
            "mean_iou": float(iou.mean()), "fraction_iou_ge_0_5": float((iou >= .5).mean())}, iou


def evaluate_boxes(pred, truth):
    """Return retained-box geometry, independent of active/lost tracker status.

    In particular, fraction_iou_ge_0_5 can count a frozen lost box that overlaps
    the target again. Use evaluate_tracking for an active tracking success rate.
    """
    return _box_geometry(pred, truth)[0]


def evaluate_tracking(pred, truth, active_mask):
    """Score every supplied frame; lost rows are always tracking failures.

    Existing fields describe retained-box geometry only. The additional
    fraction_active_iou_ge_0_5 requires both active status and IoU >= 0.5,
    divided by all scored frames. Callers select their evaluation interval
    before this call, including exclusion of manually initialized frames.
    """
    geometry, iou = _box_geometry(pred, truth)
    active = np.asarray(active_mask)
    if active.dtype.kind != "b" or active.ndim != 1 or active.shape != iou.shape:
        raise ValueError("active_mask must be a boolean [T] array matching the boxes")
    return {**geometry,
            "fraction_active_iou_ge_0_5": float((active & (iou >= .5)).mean()),
            "fraction_active": float(active.mean()),
            "scored_frames": len(iou)}


def visible_response_indices(response_times, display_times):
    """Latest response already available at each display time, or -1.

    There is deliberately no positive epsilon: even a slightly future response
    must remain invisible. Empty response arrays are valid and yield only -1.
    """
    responses = _finite_array(response_times, "response_times")
    displays = _finite_array(display_times, "display_times")
    for values in (responses, displays):
        if values.ndim != 1 or np.any(np.diff(values) <= 0):
            raise ValueError("Times must be one-dimensional and strictly increasing")
    return np.searchsorted(responses, displays, side="right").astype(np.int64) - 1
