"""Stateful Flyvis visual population readout for an encoded YOLO target mask.

This is a simulation research interface, not a biological motor circuit. YOLO
supplies salience pixels; the spatial moments below come only from the actual
L2 neural activity, never from a box, image moments, or training labels.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np

from experiments.runtime import VideoFlyvisAdapter

IMAGE_SIDE = 391
DT = .02
CELL_TYPE = "L2"
FEATURE_NAMES = ("mean_x", "mean_y", "std_x", "std_y", "cov_xy",
                 "log1p_mass", "mean_evidence", "peak_evidence")
MIN_PEAK_EVIDENCE = .05
MIN_TOTAL_EVIDENCE = .05


def validate_mask(mask, hold_s=.1):
    """Validate before advancing recurrence; one image is held for N steps."""
    if (not isinstance(mask, np.ndarray) or mask.dtype != np.float32
            or mask.shape != (IMAGE_SIDE, IMAGE_SIDE)
            or not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 1)):
        raise ValueError("mask must be finite float32 [391,391] in [0,1]")
    if (isinstance(hold_s, (bool, np.bool_)) or not isinstance(hold_s, (int, float))
            or not math.isfinite(hold_s) or not DT <= hold_s <= .2):
        raise ValueError("hold_s must be between .02 and .2 seconds")
    steps = round(hold_s / DT)
    if not math.isclose(steps * DT, hold_s, rel_tol=0, abs_tol=1e-10):
        raise ValueError("hold_s must be an exact multiple of .02 seconds")
    return steps


def population_features(activity, baseline_activity, cell_indices, centers_rc):
    """Return eight moments of absolute baseline-relative L2 activity.

    Receptor centres with incomplete 13x13 kernels are excluded. Coordinates
    are (pixel-195)/195: x increases right, y increases down. The standard
    deviations use these same normalized units. Evidence is in model activity
    units, not calibrated confidence. A blank/zero-evidence population returns
    zero features and valid=False; centering is never invented from pixels.
    """
    activity = np.asarray(activity)
    baseline = np.asarray(baseline_activity)
    indices = np.asarray(cell_indices)
    centers = np.asarray(centers_rc)
    if (activity.shape != (45669,) or baseline.shape != activity.shape
            or activity.dtype.kind not in "fiu" or baseline.dtype.kind not in "fiu"
            or not np.isfinite(activity).all() or not np.isfinite(baseline).all()):
        raise ValueError("activity and baseline must contain 45669 finite values")
    if (indices.shape != (721,) or indices.dtype.kind not in "iu"
            or len(np.unique(indices)) != 721 or np.any(indices < 0)
            or np.any(indices >= len(activity))):
        raise ValueError("cell_indices must select 721 distinct neural sites")
    if (centers.shape != (721, 2) or centers.dtype.kind not in "fiu"
            or not np.isfinite(centers).all()):
        raise ValueError("centers_rc must contain 721 finite row/column sites")
    support = np.all((centers >= 6) & (centers <= IMAGE_SIDE - 7), axis=1)
    evidence = np.abs(activity[indices].astype(np.float64)
                      - baseline[indices].astype(np.float64))
    evidence[~support] = 0
    total = float(evidence.sum())
    peak = float(evidence.max())
    valid = bool(total >= MIN_TOTAL_EVIDENCE and peak >= MIN_PEAK_EVIDENCE)
    diagnostics = {"valid": valid, "total_evidence": total,
                   "peak_evidence": peak, "supported_sites": int(support.sum())}
    if not valid:
        return np.zeros(len(FEATURE_NAMES), np.float64), diagnostics
    xy = (centers[:, ::-1].astype(np.float64) - 195.) / 195.
    mean = np.sum(evidence[:, None] * xy, axis=0) / total
    delta = xy - mean
    variance = np.sum(evidence[:, None] * delta ** 2, axis=0) / total
    covariance = float(np.sum(evidence * delta[:, 0] * delta[:, 1]) / total)
    features = np.array([*mean, *np.sqrt(np.maximum(variance, 0)), covariance,
                         np.log1p(total), total / max(int(support.sum()), 1), peak],
                        dtype=np.float64)
    return features, diagnostics


class HybridFlyvis:
    """Frozen actual visual network with recurrent state retained across masks.

    Only reset() starts a new episode. step() holds the current salience frame
    at .02-second integration steps and returns its last neural response. This
    observation is available at response_time_s, not at stimulus_time_s.
    No physical flight interface or hidden detector coordinates are accepted.
    """

    def __init__(self, manifest_path: str | Path):
        self.adapter = VideoFlyvisAdapter(manifest_path)
        self.manifest_path = Path(manifest_path)
        nodes = self.adapter.network.connectome.nodes
        types = nodes.type[:].astype(str)
        u, v = np.asarray(nodes.u[:]), np.asarray(nodes.v[:])
        lattice = np.array([(a, b) for a in range(-15, 16)
                            for b in range(max(-15, -15-a), min(15, 15-a)+1)])
        cells = np.flatnonzero(types == CELL_TYPE)
        table = {(int(u[i]), int(v[i])): int(i) for i in cells}
        if len(table) != 721 or len(cells) != 721:
            raise ValueError("Expected exactly 721 L2 sites")
        self.cell_indices = np.array([table[tuple(coord)] for coord in lattice], np.int64)
        self.centers_rc = self.adapter.eye.receptor_centers.cpu().numpy() + [195, 195]
        expected = np.array([[int(13*(a+b/2))+195, 13*b+195] for a, b in lattice])
        if not np.array_equal(self.centers_rc, expected):
            raise ValueError("Retinal lattice does not match the L2 connectome ordering")
        self.feature_names = FEATURE_NAMES
        self.reset()

    def reset(self):
        started = time.perf_counter()
        adapter = self.adapter
        with adapter.torch.inference_mode():
            blank = adapter.torch.full((1, 1, IMAGE_SIDE, IMAGE_SIDE), .5,
                                       dtype=adapter.torch.float32, device=adapter.device)
            retina = adapter.eye(blank)
            self._state = adapter.network.fade_in_state(1., DT, retina[:, 0])
            self.baseline_activity = self._activity(self._state)
        self._time_s = 0.
        self._step_count = 0
        self.reset_elapsed_wall_s = time.perf_counter() - started
        return {"elapsed_wall_s": self.reset_elapsed_wall_s, "fade_in_s": 1.,
                "blank_luminance": .5, "state_reset": "episode"}

    @staticmethod
    def _activity(state):
        activity = state.nodes.activity.detach().cpu().numpy()[0].copy()
        if activity.shape != (45669,) or not np.isfinite(activity).all():
            raise ValueError("Flyvis returned invalid neural activity")
        return activity.astype(np.float32, copy=False)

    def step(self, mask, hold_s=.1):
        steps = validate_mask(mask, hold_s)
        started = time.perf_counter()
        adapter = self.adapter
        with adapter.torch.inference_mode():
            tensor = adapter.torch.as_tensor(mask[None, None].copy(),
                                            device=adapter.device)
            retina = adapter.eye(tensor)
            if tuple(retina.shape) != (1, 1, 1, 721):
                raise ValueError("BoxEye returned an invalid retinal shape")
            states = adapter.network.simulate(retina.repeat(1, steps, 1, 1), DT,
                                               initial_state=self._state, as_states=True)
            next_state = states[-1]
            activity = self._activity(next_state)
            retinal_values = retina.detach().cpu().numpy()[0, 0, 0].copy()
        features, diagnostic = population_features(activity, self.baseline_activity,
                                                   self.cell_indices, self.centers_rc)
        stimulus_time = self._time_s
        self._state = next_state
        self._step_count += steps
        self._time_s = self._step_count * DT
        return {"activity": activity, "retina": retinal_values,
                "features": features, **diagnostic,
                "feature_names": list(FEATURE_NAMES), "cell_type": CELL_TYPE,
                "stimulus_time_s": stimulus_time, "response_time_s": self._time_s,
                "neural_steps": steps, "elapsed_wall_s": time.perf_counter() - started}
