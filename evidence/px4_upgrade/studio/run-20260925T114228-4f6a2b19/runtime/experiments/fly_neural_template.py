"""Causal spatial-template tracking using supplied receptor features only.

This experiment has no detector, image input, motion fallback, or identity
labels. The caller selects one initial box and supplies a sequence of [C,N]
features from a frozen neural model (or an explicitly named baseline). The
tracker remembers the initial feature pattern and searches locally thereafter.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import Delaunay, QhullError

IMAGE_SIDE = 391
RECEPTOR_MARGIN = 6
MINIMUM_TEMPLATE_POINTS = 9
MINIMUM_TEMPLATE_SUPPORT = 0.8


def _finite(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numbers")
    array = array.astype(np.float64, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _scalar(value, name, low, high, *, low_open=False):
    array = _finite(value, name)
    if array.ndim != 0:
        raise ValueError(f"{name} must be a scalar")
    number = float(array)
    if number > high or number < low or (low_open and number == low):
        raise ValueError(f"{name} is outside the supported range")
    return number


class NeuralTemplateTracker:
    """Remember one spatial feature template and match it within a local disk.

    ``channel_scale`` is a positive [C] divisor, typically a scale frozen from
    training data. No scale is learned from future frames. The interpolation
    uses only receptors whose full 13x13 eye kernel lies inside the image;
    positions outside their convex hull have no evidence.

    ``tracking`` means that a sufficiently correlated candidate was accepted.
    ``uncertain`` (low correlation) and ``insufficient_evidence`` (flat features
    or inadequate support) retain the previous box and allow a later local
    recovery. An unusable initial template cannot be replaced automatically.
    Search radius is a per-observation displacement limit in pixels, not a
    velocity estimate; ``dt`` records the caller's causal observation interval.
    """

    def __init__(self, centers_rc, initial_box, initial_features,
                 channel_scale=None, grid_step=4, search_radius_px=24,
                 minimum_score=0.35, template_update=0.0):
        centers = _finite(centers_rc, "centers_rc")
        if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) < 3:
            raise ValueError("centers_rc must have shape [N,2], with N >= 3")
        self.centers_rc = centers.copy()
        box = _finite(initial_box, "initial_box")
        if (box.shape != (4,) or np.any(box[2:] <= box[:2])
                or np.any(box[:2] < 0) or np.any(box[2:] > IMAGE_SIDE)):
            raise ValueError("initial_box must be a positive xyxy box inside the 391-square image")
        self.box = box.copy()
        if (isinstance(grid_step, (bool, np.bool_))
                or not isinstance(grid_step, (int, np.integer))
                or not 1 <= grid_step <= 32):
            raise ValueError("grid_step must be an integer from 1 through 32")
        self.grid_step = int(grid_step)
        self.search_radius_px = _scalar(search_radius_px, "search_radius_px", 0, 96)
        self.minimum_score = _scalar(minimum_score, "minimum_score", -1, 1)
        self.template_update = _scalar(template_update, "template_update", 0, 1)
        initial = _finite(initial_features, "initial_features")
        if initial.ndim != 2 or initial.shape[1] != len(centers) or initial.shape[0] < 1:
            raise ValueError("initial_features must have shape [C,N], with C >= 1")
        self.feature_shape = initial.shape
        if channel_scale is None:
            self.channel_scale = np.ones(initial.shape[0], dtype=np.float64)
        else:
            scale = _finite(channel_scale, "channel_scale")
            if scale.shape != (initial.shape[0],) or np.any(scale <= 0):
                raise ValueError("channel_scale must be a positive [C] array")
            self.channel_scale = scale.copy()

        valid_receptors = np.all((centers >= RECEPTOR_MARGIN)
                                & (centers <= IMAGE_SIDE - RECEPTOR_MARGIN - 1), axis=1)
        self.receptor_indices = np.flatnonzero(valid_receptors)
        if len(self.receptor_indices) < 3:
            raise ValueError("At least three interior receptors are required")
        interior = centers[self.receptor_indices]
        if len(np.unique(interior, axis=0)) != len(interior):
            raise ValueError("Interior receptor centers must be distinct")
        try:
            triangulation = Delaunay(interior)
        except QhullError as exc:
            raise ValueError("Interior receptor centers must span a two-dimensional region") from exc
        axis = np.arange(0, IMAGE_SIDE, self.grid_step, dtype=np.float64)
        rows, cols = np.meshgrid(axis, axis, indexing="ij")
        self.grid_shape = rows.shape
        points = np.column_stack((rows.ravel(), cols.ravel()))
        simplex = triangulation.find_simplex(points)
        supported = simplex >= 0
        supported_points = points[supported]
        transforms = triangulation.transform[simplex[supported]]
        first_weights = np.einsum("nij,nj->ni", transforms[:, :2, :],
                                  supported_points - transforms[:, 2, :])
        self.weights = np.column_stack((first_weights, 1 - first_weights.sum(axis=1)))
        self.vertices = self.receptor_indices[triangulation.simplices[simplex[supported]]]
        self.grid_indices = np.flatnonzero(supported)
        self.support_mask = supported.reshape(self.grid_shape)

        # Sampling starts at the first fixed-grid location inside the box;
        # the continuous original box retains its exact fractional offset.
        self.anchor_rc = np.ceil(box[[1, 0]] / self.grid_step).astype(np.int64)
        end_rc = np.ceil(box[[3, 2]] / self.grid_step).astype(np.int64)
        self.template_shape = tuple((end_rc - self.anchor_rc).tolist())
        row, col = self.anchor_rc
        height, width = self.template_shape
        self.template_mask = self.support_mask[row:row + height, col:col + width].copy()
        self.template_points = int(self.template_mask.sum())
        self.template_support_fraction = (float(self.template_mask.mean())
                                          if self.template_mask.size else 0.0)
        feature_map = self._feature_map(initial)
        self.template = feature_map[:, row:row + height, col:col + width].copy()
        self.template_usable = (self.template_points >= MINIMUM_TEMPLATE_POINTS
                                and self.template_support_fraction >= MINIMUM_TEMPLATE_SUPPORT)
        self._prepare_template()
        self.status = "tracking" if self.template_usable else "insufficient_evidence"

    def _feature_map(self, value):
        features = _finite(value, "features")
        if features.shape != self.feature_shape:
            raise ValueError("features must retain the initial [C,N] shape")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            scaled = features / self.channel_scale[:, None]
            # Correlation is invariant to a spatially constant channel offset.
            # Remove it in float64 before float32 convolutions so a large neural
            # resting potential cannot erase a small spatial signal numerically.
            # This uses the present observation only and estimates no trajectory.
            scaled = scaled - scaled[:, self.receptor_indices].mean(axis=1, keepdims=True)
            samples = np.einsum("cnk,nk->cn", scaled[:, self.vertices], self.weights)
        if not np.isfinite(samples).all() or np.any(np.abs(samples) > 1e12):
            raise ValueError("Scaled interpolated features exceed the finite numeric range")
        maps = np.zeros((features.shape[0], int(np.prod(self.grid_shape))), dtype=np.float32)
        maps[:, self.grid_indices] = samples
        return maps.reshape((features.shape[0], *self.grid_shape))

    def _prepare_template(self):
        if not self.template_usable:
            self.centered_template = np.zeros_like(self.template)
            self.template_energy = 0.0
            return
        selected = self.template[:, self.template_mask].astype(np.float64)
        means = selected.mean(axis=1)
        centered = (self.template.astype(np.float64) - means[:, None, None]) * self.template_mask
        energy = float(np.sum(centered ** 2))
        raw_energy = float(np.sum(selected ** 2))
        self.template_usable = bool(energy > max(1e-10, raw_energy * 1e-10))
        self.centered_template = centered.astype(np.float32)
        self.template_energy = energy

    def _result(self, dt, score=None, candidate_box=None, delta=None,
                candidate_count=0, reason=None):
        movement = np.zeros(2) if delta is None else np.asarray(delta)
        return {"box_xyxy": self.box.tolist(), "status": self.status,
                "score": score, "displacement_xy": movement.tolist(),
                "candidate_box_xyxy": None if candidate_box is None else candidate_box.tolist(),
                "selected_position_xy": ((self.box[:2] + self.box[2:]) / 2).tolist(),
                "candidate_count": int(candidate_count),
                "template_supported_points": self.template_points,
                "template_support_fraction": self.template_support_fraction,
                "dt_s": dt, "reason": reason}

    def step(self, features, dt=0.02):
        """Consume only the current feature observation and retain target memory."""
        elapsed = _scalar(dt, "dt", 0, np.inf, low_open=True)
        feature_map = self._feature_map(features)  # Validate before changing state.
        if not self.template_usable:
            self.status = "insufficient_evidence"
            return self._result(elapsed, reason="unusable_initial_template")

        radius = int(np.floor(self.search_radius_px / self.grid_step))
        height, width = self.template_shape
        max_anchor = np.asarray(self.grid_shape) - np.asarray(self.template_shape)
        start = np.maximum(0, self.anchor_rc - radius)
        end = np.minimum(max_anchor, self.anchor_rc + radius)
        top, left = start
        bottom, right = end + np.asarray(self.template_shape)
        search = np.ascontiguousarray(feature_map[:, top:bottom, left:right])
        mask = self.template_mask.astype(np.float32)
        search_support = self.support_mask[top:bottom, left:right].astype(np.float32)
        support_counts = cv2.matchTemplate(search_support, mask, cv2.TM_CCORR)
        shape = support_counts.shape
        candidate_rows, candidate_cols = np.indices(shape)
        offsets_rc = np.stack((candidate_rows + top - self.anchor_rc[0],
                               candidate_cols + left - self.anchor_rc[1]), axis=-1)
        offsets_xy = offsets_rc[..., ::-1] * self.grid_step
        candidate_boxes = self.box + np.tile(offsets_xy, (1, 1, 2))
        allowed = ((support_counts >= self.template_points - 0.25)
                   & (np.sum(offsets_xy ** 2, axis=-1) <= self.search_radius_px ** 2 + 1e-9)
                   & np.all(candidate_boxes[..., :2] >= 0, axis=-1)
                   & np.all(candidate_boxes[..., 2:] <= IMAGE_SIDE, axis=-1))
        numerator = np.zeros(shape, dtype=np.float64)
        energy = np.zeros(shape, dtype=np.float64)
        raw_energy = np.zeros(shape, dtype=np.float64)
        for channel, template in zip(search, self.centered_template):
            # Per-channel centering removes fixed response offsets, while the
            # joint norm retains the training-frozen channel weights.
            sums = cv2.matchTemplate(channel, mask, cv2.TM_CCORR).astype(np.float64)
            squares = cv2.matchTemplate(channel ** 2, mask, cv2.TM_CCORR).astype(np.float64)
            numerator += cv2.matchTemplate(channel, template, cv2.TM_CCORR)
            energy += np.maximum(0, squares - sums ** 2 / self.template_points)
            raw_energy += np.maximum(0, squares)
        # Float32 convolutions can leave tiny positive variance for a constant
        # candidate. A relative threshold rejects that cancellation residue.
        informative = energy > np.maximum(1e-9, raw_energy * 2e-6)
        allowed &= informative
        candidate_count = int(allowed.sum())
        if candidate_count == 0:
            self.status = "insufficient_evidence"
            return self._result(elapsed, reason="no_supported_nonflat_candidate")
        scores = np.full(shape, -np.inf, dtype=np.float64)
        scores[allowed] = np.clip(numerator[allowed]
                                  / np.sqrt(self.template_energy * energy[allowed]), -1, 1)
        best_score = float(np.max(scores))
        tied = np.argwhere(allowed & (scores >= best_score - 1e-6))
        # Prefer no displacement, then stable row/column ordering for exact
        # periodic symmetries. A tied pattern never causes arbitrary drift.
        best_row, best_col = min(tied.tolist(), key=lambda rc: (
            int(np.sum(offsets_xy[rc[0], rc[1]] ** 2)), rc[0], rc[1]))
        delta = offsets_xy[best_row, best_col].astype(np.float64)
        candidate = candidate_boxes[best_row, best_col]
        selected_score = float(scores[best_row, best_col])
        if selected_score < self.minimum_score:
            self.status = "uncertain"
            return self._result(elapsed, selected_score, candidate,
                                candidate_count=candidate_count, reason="below_minimum_score")
        self.box = candidate.copy()
        self.anchor_rc = self.anchor_rc + offsets_rc[best_row, best_col]
        self.status = "tracking"
        if self.template_update:
            patch = search[:, best_row:best_row + height, best_col:best_col + width]
            self.template = ((1 - self.template_update) * self.template
                             + self.template_update * patch).astype(np.float32)
            self._prepare_template()
        return self._result(elapsed, selected_score, candidate, delta, candidate_count)
