"""Training-only standardized ridge readouts for local Flyvis motion signals.

This module has no image/detector input, ground-truth tracker updates, or startup
stationarity assumption. Its only tracker motion evidence is the supplied neural
flow in the tracker's own previous box. Training data selection, temporal
alignment, and held-out evaluation belong to the experiment runner.
"""
from __future__ import annotations

import numpy as np

from experiments.fly_tracking_core import IMAGE_SIDE, supported_indices

FEATURE_COUNTS = {"median": 2, "statistics": 14}


def _array(value, name):
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


def _scalar(value, name, positive=False):
    array = _array(value, name)
    if array.ndim != 0 or (array.item() <= 0 if positive else array.item() < 0):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} scalar")
    return float(array)


def _mode(value):
    if not isinstance(value, str) or value not in FEATURE_COUNTS:
        raise ValueError("mode must be 'median' or 'statistics'")
    return value


def _minimum(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 9):
        raise ValueError("min_support must be an integer >= 9")
    return int(value)


def _geometry(centers_rc, box):
    centers = _array(centers_rc, "centers_rc")
    target = _array(box, "box")
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers_rc must have shape [N,2]")
    if target.shape != (4,) or np.any(target[2:] <= target[:2]):
        raise ValueError("box must have shape [4] and positive width and height")
    return centers, target


def _field(value, count):
    field = _array(value, "flow")
    if field.shape != (2, count):
        raise ValueError("flow must have shape [2,N] matching centers_rc")
    return field


def _features(field, centers, box, indices, mode):
    selected = field[:, indices]
    with np.errstate(over="ignore", invalid="ignore"):
        median = np.median(selected, axis=1)
    if not np.isfinite(median).all():
        raise ValueError("Feature extraction produced nonfinite values")
    if mode == "median":
        return median
    mean = np.mean(selected, axis=1)
    mid_x, mid_y = (box[:2] + box[2:]) / 2
    rows, cols = centers[indices].T
    quadrants = []
    for bottom, right in ((False, False), (False, True), (True, False), (True, True)):
        mask = ((rows >= mid_y) == bottom) & ((cols >= mid_x) == right)
        quadrants.append(selected[:, mask].mean(axis=1) if np.any(mask) else mean)
    features = np.concatenate((mean, median, selected.std(axis=1), *quadrants))
    if not np.isfinite(features).all():
        raise ValueError("Feature extraction produced nonfinite values")
    return features


def flow_features(flow, centers_rc, box, mode="median", min_support=9):
    """Extract supported local flow features, with fixed order and no fitting.

    ``median`` yields [median_x, median_y]. ``statistics`` yields channel pairs
    for mean, median, population standard deviation, and quadrant means in
    TL/TR/BL/BR order (14 values). Empty quadrants use the overall mean.
    Thirteen-pixel receptive windows must be wholly inside the image, and at
    least nine receptor centers must be inside the half-open xyxy box.
    """
    selected_mode = _mode(mode)
    minimum = _minimum(min_support)
    centers, target = _geometry(centers_rc, box)
    field = _field(flow, len(centers))
    indices = supported_indices(centers, target, min_support=minimum)
    if len(indices) < minimum:
        raise ValueError(f"At least {minimum} interior receptors must support the box")
    return _features(field, centers, target, indices, selected_mode)


class RidgeReadout:
    """A frozen affine mapping from standardized neural features to pixels/s.

    ``fit`` minimizes sum(normalized_weight * squared_error) plus ``ridge``
    times squared standardized slopes. The weighted target mean is the
    unpenalized intercept. Zero-weight rows affect neither scaling nor domain.
    Constant training features use scale 1 and a fitted slope of zero.
    """

    def __init__(self, mean, scale, coefficients, intercept, training_min,
                 training_max, ridge, diagnostics):
        arrays = {name: _array(value, name).copy() for name, value in (
            ("mean", mean), ("scale", scale), ("coefficients", coefficients),
            ("intercept", intercept), ("training_min", training_min),
            ("training_max", training_max))}
        mean = arrays["mean"]
        if mean.ndim != 1 or not len(mean):
            raise ValueError("mean must be a nonempty feature vector")
        count = len(mean)
        if any(arrays[name].shape != (count,) for name in ("scale", "training_min", "training_max")):
            raise ValueError("Feature metadata must match mean shape")
        if arrays["coefficients"].shape != (count, 2) or arrays["intercept"].shape != (2,):
            raise ValueError("Readout coefficients/intercept must produce 2D velocity")
        if np.any(arrays["scale"] <= 0) or np.any(arrays["training_min"] > arrays["training_max"]):
            raise ValueError("Invalid feature scale or domain")
        if np.any(mean < arrays["training_min"] - 1e-10) or np.any(mean > arrays["training_max"] + 1e-10):
            raise ValueError("Feature mean must lie in its training domain")
        self.ridge = _scalar(ridge, "ridge")
        if not isinstance(diagnostics, dict):
            raise ValueError("diagnostics must be a dictionary")
        expected = {"training_samples", "positive_weight_samples", "weighted_rmse_px_per_s"}
        if set(diagnostics) != expected:
            raise ValueError("Unexpected diagnostic fields")
        for key in ("training_samples", "positive_weight_samples"):
            if (isinstance(diagnostics[key], (bool, np.bool_))
                    or not isinstance(diagnostics[key], (int, np.integer)) or diagnostics[key] < 1):
                raise ValueError(f"{key} must be a positive integer")
        if diagnostics["positive_weight_samples"] > diagnostics["training_samples"]:
            raise ValueError("Positive-weight samples exceed training samples")
        self.diagnostics = {
            "training_samples": int(diagnostics["training_samples"]),
            "positive_weight_samples": int(diagnostics["positive_weight_samples"]),
            "weighted_rmse_px_per_s": _scalar(diagnostics["weighted_rmse_px_per_s"], "weighted_rmse_px_per_s"),
        }
        for name, array in arrays.items():
            array.setflags(write=False)
            setattr(self, name, array)
        self.feature_count = count

    @classmethod
    def fit(cls, X, y, weights=None, ridge=1e-3):
        features, target = _array(X, "X"), _array(y, "y")
        penalty = _scalar(ridge, "ridge")
        if features.ndim != 2 or min(features.shape) < 1 or target.shape != (len(features), 2):
            raise ValueError("Training requires nonempty X[N,D] and y[N,2]")
        weight = np.ones(len(features)) if weights is None else _array(weights, "weights")
        if weight.shape != (len(features),) or np.any(weight < 0) or not np.any(weight > 0):
            raise ValueError("weights must be nonnegative [N] with some positive support")
        # Max rescaling avoids overflow and leaves the normalized objective intact.
        weight = weight / weight.max()
        weight = weight / weight.sum()
        supported = weight > 0
        training_count = len(features)
        features, target, weight = features[supported], target[supported], weight[supported]
        # Drop zero-weight rows before arithmetic: even enormous excluded
        # observations must not overflow scaling or change the training domain.
        with np.errstate(over="ignore", invalid="ignore"):
            mean = np.sum(features * weight[:, None], axis=0)
            centered = features - mean
            variance = np.sum(centered**2 * weight[:, None], axis=0)
            scale = np.sqrt(variance)
            scale = np.where(scale > 1e-12, scale, 1.)
            standardized = centered / scale
            intercept = np.sum(target * weight[:, None], axis=0)
        if not all(np.isfinite(value).all() for value in (mean, variance, scale, standardized, intercept)):
            raise ValueError("Training standardization produced nonfinite values")
        root_weight = np.sqrt(weight)[:, None]
        matrix = np.vstack((standardized * root_weight,
                            np.sqrt(penalty) * np.eye(features.shape[1])))
        rhs = np.vstack(((target - intercept) * root_weight,
                         np.zeros((features.shape[1], 2))))
        if not np.isfinite(matrix).all() or not np.isfinite(rhs).all():
            raise ValueError("Training regression produced nonfinite values")
        coefficients = np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        residual = standardized @ coefficients + intercept - target
        diagnostics = {"training_samples": training_count,
                       "positive_weight_samples": len(features),
                       "weighted_rmse_px_per_s": float(np.sqrt(np.sum(weight[:, None] * residual**2) / 2))}
        return cls(mean, scale, coefficients, intercept,
                   features.min(axis=0), features.max(axis=0), penalty, diagnostics)

    def _input(self, X):
        features = _array(X, "X")
        if features.ndim not in (1, 2) or features.shape[-1] != self.feature_count:
            raise ValueError("Prediction requires [D] or [N,D] matching the learned feature count")
        return features

    def predict(self, X):
        features = self._input(X)
        with np.errstate(over="ignore", invalid="ignore"):
            prediction = ((features - self.mean) / self.scale) @ self.coefficients + self.intercept
        if not np.isfinite(prediction).all():
            raise ValueError("Readout prediction produced nonfinite velocity")
        return prediction

    def feature_domain(self, X):
        """Describe extrapolation, without modifying features or predictions."""
        features = self._input(X)
        if features.ndim != 1:
            raise ValueError("feature_domain requires a single [D] feature vector")
        outside = (features < self.training_min) | (features > self.training_max)
        z = (features - self.mean) / self.scale
        if not np.isfinite(z).all():
            raise ValueError("Feature domain check produced nonfinite values")
        return {"feature_count": self.feature_count,
                "outside_training_range_count": int(outside.sum()),
                "max_abs_training_zscore": float(np.max(np.abs(z)))}

    def to_dict(self):
        return {"schema_version": 1, "feature_count": self.feature_count,
                **{name: getattr(self, name).tolist() for name in (
                    "mean", "scale", "coefficients", "intercept", "training_min", "training_max")},
                "ridge": self.ridge, "diagnostics": self.diagnostics.copy()}

    @classmethod
    def from_dict(cls, data):
        expected = {"schema_version", "feature_count", "mean", "scale", "coefficients", "intercept",
                    "training_min", "training_max", "ridge", "diagnostics"}
        if not isinstance(data, dict) or set(data) != expected:
            raise ValueError("Unexpected serialized readout fields")
        if type(data["schema_version"]) is not int or data["schema_version"] != 1:
            raise ValueError("Unsupported readout schema_version")
        if type(data["feature_count"]) is not int or data["feature_count"] < 1:
            raise ValueError("feature_count must be a positive integer")
        model = cls(**{key: data[key] for key in expected - {"schema_version", "feature_count"}})
        if model.feature_count != data["feature_count"]:
            raise ValueError("Serialized feature_count does not match readout")
        return model


class LearnedFlowTracker:
    """Fixed-size target box advected only by learned local neural flow.

    Missing support, out-of-frame candidates, or speed over the predeclared
    maximum cause permanent loss. A lost box retains its last accepted position;
    loss is never hidden by clipping motion or reinitializing from another model.
    """

    def __init__(self, centers_rc, initial_box, readout, mode="median", min_support=9, max_speed=120):
        centers, box = _geometry(centers_rc, initial_box)
        self.centers_rc, self.box = centers.copy(), box.copy()
        self.mode = _mode(mode)
        self.min_support = _minimum(min_support)
        self.max_speed = _scalar(max_speed, "max_speed", positive=True)
        if not isinstance(readout, RidgeReadout) or readout.feature_count != FEATURE_COUNTS[self.mode]:
            raise ValueError("readout must match the chosen feature mode")
        self.readout = RidgeReadout.from_dict(readout.to_dict())
        if not self._inside(self.box):
            raise ValueError("initial_box must be inside the 391-square image")
        self.status = "tracking"

    @staticmethod
    def _inside(box):
        return bool(np.all(box[:2] >= 0) and np.all(box[2:] <= IMAGE_SIDE))

    def _result(self, indices=None, features=None, pooled=None, velocity=None, domain=None):
        return {"box_xyxy": self.box.tolist(), "status": self.status,
                "selected_indices": [] if indices is None else indices.tolist(),
                "pooled_flow": None if pooled is None else pooled.tolist(),
                "velocity_xy": None if velocity is None else velocity.tolist(),
                "features": None if features is None else features.tolist(),
                "feature_mode": self.mode, "feature_count": self.readout.feature_count,
                "feature_domain": domain}

    def step(self, field, dt):
        flow = _field(field, len(self.centers_rc))
        elapsed = _scalar(dt, "dt", positive=True)
        if self.status != "tracking":
            return self._result()
        indices = supported_indices(self.centers_rc, self.box, min_support=self.min_support)
        if len(indices) < self.min_support:
            self.status = "insufficient_support"
            return self._result()
        features = _features(flow, self.centers_rc, self.box, indices, self.mode)
        pooled = np.median(flow[:, indices], axis=1)
        velocity = self.readout.predict(features)
        domain = self.readout.feature_domain(features)
        if np.linalg.norm(velocity) > self.max_speed:
            self.status = "uncertain_speed"
            return self._result(indices, features, pooled, velocity, domain)
        with np.errstate(over="ignore", invalid="ignore"):
            candidate = self.box + np.tile(velocity * elapsed, 2)
        if not np.isfinite(candidate).all():
            raise ValueError("Flow integration produced nonfinite coordinates")
        if not self._inside(candidate):
            self.status = "outside_frame"
        else:
            self.box = candidate
        return self._result(indices, features, pooled, velocity, domain)
