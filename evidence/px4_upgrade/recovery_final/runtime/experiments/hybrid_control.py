"""Engineered visual-servo readout around frozen Flyvis, for simulation only.

There are no simulator target coordinates, YOLO box coordinates, metric depth,
motor interfaces or world routes in this controller. Apparent target size is
an image-space setpoint, not a metric distance measurement.
"""
from __future__ import annotations

import numpy as np


class NeuralReadout:
    """Training-only standardized ridge: eight neural features -> x/y/height."""

    def __init__(self, mean, scale, coefficients, intercept):
        self.mean, self.scale, self.coefficients, self.intercept = [
            np.array(v, dtype=float, copy=True) for v in (mean, scale, coefficients, intercept)]
        if (self.mean.shape != (8,) or self.scale.shape != (8,)
                or self.coefficients.shape != (8, 3) or self.intercept.shape != (3,)
                or any(not np.isfinite(v).all() for v in
                       (self.mean, self.scale, self.coefficients, self.intercept))
                or np.any(self.scale <= 0)):
            raise ValueError("invalid neural readout")
        for a in (self.mean, self.scale, self.coefficients, self.intercept):
            a.setflags(write=False)

    @classmethod
    def fit(cls, features, labels, ridge=1e-4):
        x, y = np.asarray(features, float), np.asarray(labels, float)
        if (x.ndim != 2 or x.shape[1] != 8 or len(x) < 9 or y.shape != (len(x), 3)
                or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not np.isfinite(ridge) or ridge <= 0):
            raise ValueError("finite training features[N,8] and labels[N,3] required")
        mean, scale = x.mean(0), x.std(0)
        scale = np.where(scale > 1e-12, scale, 1.)
        z, intercept = (x - mean) / scale, y.mean(0)
        coefficients = np.linalg.solve(z.T @ z / len(x) + ridge*np.eye(8),
                                       z.T @ (y-intercept) / len(x))
        return cls(mean, scale, coefficients, intercept)

    def predict(self, features):
        x = np.asarray(features, float)
        if x.shape[-1:] != (8,) or not np.isfinite(x).all():
            raise ValueError("finite eight-element neural features required")
        return ((x-self.mean)/self.scale) @ self.coefficients + self.intercept

    def to_dict(self):
        return {k: getattr(self, k).tolist() for k in ("mean", "scale", "coefficients", "intercept")}


CONTROL_CONFIG = dict(desired_height_fraction=.38, lateral_gain=2.4,
                      size_gain=2., forward_limit=.8, vertical_limit=.6,
                      speed_limit=1.2, command_lifetime_s=.15,
                      max_response_age_s=.2)


class HybridController:
    def __init__(self, readout):
        self.readout = readout

    def command(self, features, *, target_valid, neural_valid, capture_time_s, response_time_s):
        """Return a command available at response time; loss requests braking."""
        def hold(reason):
            return dict(velocity=[0., 0., 0.], mode="brake", reason=reason,
                        decoded=None, issued_at_s=float(response_time_s),
                        valid_until_s=float(response_time_s)+CONTROL_CONFIG["command_lifetime_s"])
        if (not np.isfinite(capture_time_s) or not np.isfinite(response_time_s)
                or capture_time_s < 0 or response_time_s < capture_time_s
                or response_time_s-capture_time_s > CONTROL_CONFIG["max_response_age_s"]+1e-9):
            return hold("invalid_response_timing")
        if not target_valid:
            return hold("target_lost")
        f = np.asarray(features, float)
        if (not neural_valid or f.shape != (8,) or not np.isfinite(f).all()
                or f[7] < .05 or f[5] < np.log1p(.05)):
            return hold("neural_evidence_unavailable")
        decoded = self.readout.predict(f)
        x, y, height = decoded
        if (not np.isfinite(decoded).all() or abs(x) > .85 or abs(y) > .85
                or not .08 <= height <= .8):
            return hold("neural_readout_out_of_range")
        forward = CONTROL_CONFIG["size_gain"]*(CONTROL_CONFIG["desired_height_fraction"]/height-1)
        if max(abs(x), abs(y)) > .35:
            forward = 0.  # Centre the image target before advancing.
        cfg = CONTROL_CONFIG
        velocity = np.array([np.clip(forward, -cfg['forward_limit'], cfg['forward_limit']),
                             cfg['lateral_gain']*x,
                             np.clip(-cfg['lateral_gain']*y, -cfg['vertical_limit'], cfg['vertical_limit'])])
        velocity *= min(1., cfg['speed_limit'] / max(float(np.linalg.norm(velocity)), 1e-12))
        return dict(velocity=velocity.tolist(), mode="visual_servo", reason="neural_target_readout",
                    decoded=decoded.tolist(), issued_at_s=float(response_time_s),
                    valid_until_s=float(response_time_s)+CONTROL_CONFIG["command_lifetime_s"])


def active_velocity(command, now_s):
    """Commands cannot act before their neural response or after expiry."""
    if (command is None or not np.isfinite(now_s)
            or now_s < command["issued_at_s"]-1e-9
            or now_s >= command["valid_until_s"]-1e-9):
        return (0., 0., 0.)
    return tuple(command["velocity"])
