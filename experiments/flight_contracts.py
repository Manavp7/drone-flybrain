"""Simulation-only contracts: world/body X forward, Y left, Z up.

Camera optical axes are X right, Y down, Z forward. No transport or aircraft
interface is provided. State estimates below are ideal simulator sensors.
"""
from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R_BODY_CAMERA = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
PHYSICS_DT = .005
ALTITUDE_M = 1.1
MAX_SPEED_M_S = .45
MAX_OBSERVATION_AGE_S = .65
COMMAND_MAX_CAPTURE_AGE_S = .90
DEPTH_PERIOD_S = .05
DEPTH_MAX_AGE_S = .10
VEHICLE_RADIUS_M = .32
SAFETY_MARGIN_M = .20
BRAKING_ACCEL_M_S2 = .35
BRAKING_TRANSIENT_MARGIN_M = .15


def load_mujoco():
    """Use the pinned project-local optional runtime without changing .venv."""
    runtime = ROOT / '.cache/mujoco_runtime'
    if runtime.is_dir() and str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    module = importlib.import_module('mujoco')
    if module.__version__ != '3.2.7':
        raise RuntimeError('This experiment requires MuJoCo3.2.7')
    return module


def wrap_angle(value):
    return (float(value) + np.pi) % (2*np.pi) - np.pi


def rotation_from_euler(roll=0., pitch=0., yaw=0.):
    sr, cr, sp, cp, sy, cy = np.sin(roll), np.cos(roll), np.sin(pitch), np.cos(pitch), np.sin(yaw), np.cos(yaw)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


@dataclass(frozen=True)
class FlightState:
    time_s: float
    position: np.ndarray
    velocity: np.ndarray
    rotation: np.ndarray
    angular_velocity: np.ndarray
    motor_forces: np.ndarray

    @property
    def yaw(self):
        return float(np.arctan2(self.rotation[1, 0], self.rotation[0, 0]))


@dataclass(frozen=True)
class CameraFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: tuple
    rotation_world_camera: np.ndarray
    position_world_camera: np.ndarray
    capture_time_s: float
    registration_verified: bool = True

    def validate(self):
        if (self.rgb.dtype != np.uint8 or self.rgb.ndim != 3 or self.rgb.shape[2] != 3
                or self.depth_m.shape != self.rgb.shape[:2] or self.depth_m.dtype.kind != 'f'):
            raise ValueError('Camera requires aligned RGB uint8 and floating optical-Z depth')
        fx, fy, cx, cy = self.intrinsics
        h, w = self.depth_m.shape
        if (not np.isfinite([fx, fy, cx, cy, self.capture_time_s]).all() or min(fx,fy) <= 0
                or not 0 <= cx < w or not 0 <= cy < h or self.capture_time_s < 0):
            raise ValueError('Invalid camera calibration or timestamp')
        r = np.asarray(self.rotation_world_camera)
        if (r.shape != (3,3) or not np.isfinite(r).all()
                or not np.allclose(r.T@r,np.eye(3),atol=1e-6)
                or abs(np.linalg.det(r)-1) > 1e-6
                or np.asarray(self.position_world_camera).shape != (3,)
                or not np.isfinite(self.position_world_camera).all()
                or type(self.registration_verified) is not bool):
            raise ValueError('Invalid optical-to-world camera pose')
        return self
