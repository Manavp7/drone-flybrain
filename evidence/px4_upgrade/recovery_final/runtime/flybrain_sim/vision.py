"""Deterministic pinhole pixels for research-model shadow experiments.

NumPy is optional for the project and required only when importing this module.
Only ``CameraFrame.luminance`` is a neural input. Depth and object IDs are
renderer truth for diagnostics; they are not estimated vision or SLAM outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np

from .contracts import Box, Vec3


@dataclass(frozen=True)
class CameraConfig:
    width: int = 96
    height: int = 72
    horizontal_fov_deg: float = 90.0
    near_m: float = 0.03
    far_m: float = 80.0

    def __post_init__(self) -> None:
        if (type(self.width) is not int or type(self.height) is not int
                or not 1 <= self.width <= 4096 or not 1 <= self.height <= 4096):
            raise ValueError("camera dimensions must be integers in [1, 4096]")
        if not math.isfinite(self.horizontal_fov_deg) or not 1.0 < self.horizontal_fov_deg < 179.0:
            raise ValueError("horizontal FOV must be finite and between 1 and 179 degrees")
        if not all(math.isfinite(x) for x in (self.near_m, self.far_m)) or not 0 < self.near_m < self.far_m:
            raise ValueError("require 0 < near_m < far_m, both finite")


@dataclass(frozen=True)
class CameraPose:
    position: Vec3
    yaw_rad: float = 0.0
    pitch_rad: float = 0.0

    def __post_init__(self) -> None:
        if len(self.position) != 3 or not all(math.isfinite(x) for x in self.position):
            raise ValueError("camera position must contain three finite coordinates")
        if not all(math.isfinite(x) for x in (self.yaw_rad, self.pitch_rad)):
            raise ValueError("camera angles must be finite")
        if abs(self.pitch_rad) > math.pi / 2:
            raise ValueError("pitch must be within [-pi/2, pi/2]")

    @classmethod
    def from_velocity(cls, position: Vec3, velocity: Vec3,
                      previous_yaw_rad: float = 0.0) -> "CameraPose":
        """Level camera heading follows horizontal velocity; retains heading at rest.

        This is a declared rendering convention, not an attitude estimator or a
        claim that the simulated vehicle has a physical yaw controller.
        """
        if len(velocity) != 3 or not all(math.isfinite(v) for v in velocity):
            raise ValueError("velocity must contain three finite coordinates")
        yaw = math.atan2(velocity[1], velocity[0]) if math.hypot(*velocity[:2]) > 1e-6 else previous_yaw_rad
        return cls(position, yaw)


@dataclass(frozen=True)
class CameraFrame:
    luminance: np.ndarray                 # float32 [H,W], range [0,1]
    depth_m: np.ndarray                   # float32 ray range, +inf for no hit
    depth_valid_mask: np.ndarray          # bool [H,W], truth surface in clip range
    image_valid_mask: np.ndarray          # bool [H,W], includes valid sky pixels
    surface_ids: np.ndarray               # int32: -1 sky, 0 ground, 1.. sorted boxes
    capture_time: float
    receive_time: float
    camera: CameraConfig
    pose: CameraPose
    valid: bool = True


def camera_rays(camera: CameraConfig, pose: CameraPose) -> np.ndarray:
    """Unit world rays [H,W,3], sampled at pixel centers with square pixels.

    World z is up. Yaw zero looks along +x; positive yaw turns toward +y.
    Screen right at yaw zero is world -y; row zero is the top. Pitch positive
    looks upward. Roll is fixed at zero.
    """
    cy, sy = math.cos(pose.yaw_rad), math.sin(pose.yaw_rad)
    cp, sp = math.cos(pose.pitch_rad), math.sin(pose.pitch_rad)
    forward = np.array((cp * cy, cp * sy, sp), dtype=np.float64)
    right = np.array((sy, -cy, 0.0), dtype=np.float64)
    up = np.array((-sp * cy, -sp * sy, cp), dtype=np.float64)
    focal = camera.width / (2.0 * math.tan(math.radians(camera.horizontal_fov_deg) / 2.0))
    columns = (np.arange(camera.width, dtype=np.float64) + 0.5 - camera.width / 2.0) / focal
    rows = (np.arange(camera.height, dtype=np.float64) + 0.5 - camera.height / 2.0) / focal
    rays = forward + columns[None, :, None] * right - rows[:, None, None] * up
    return rays / np.linalg.norm(rays, axis=-1, keepdims=True)


def _box_intersection(origin: np.ndarray, rays: np.ndarray, box: Box,
                      near_m: float) -> tuple[np.ndarray, np.ndarray]:
    """First positive box-surface distance and outward normal for each ray."""
    low, high = np.asarray(box.low, dtype=np.float64), np.asarray(box.high, dtype=np.float64)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError(f"invalid box geometry: {box.id}")
    parallel = np.abs(rays) < 1e-12
    outside_parallel = np.any(parallel & ((origin < low) | (origin > high)), axis=-1)
    a = np.divide(low - origin, rays, out=np.zeros_like(rays), where=~parallel)
    b = np.divide(high - origin, rays, out=np.zeros_like(rays), where=~parallel)
    lower = np.where(parallel, -np.inf, np.minimum(a, b))
    upper = np.where(parallel, np.inf, np.maximum(a, b))
    enter, leave = np.max(lower, axis=-1), np.min(upper, axis=-1)
    entry_surface = enter >= near_m
    distance = np.where(entry_surface, enter, leave)
    hit = ~outside_parallel & (leave >= enter) & (distance >= near_m)
    distance = np.where(hit, distance, np.inf)
    axis = np.where(entry_surface, np.argmax(lower, axis=-1), np.argmin(upper, axis=-1))
    selected_ray = np.take_along_axis(rays, axis[..., None], axis=-1)[..., 0]
    direction = np.where(entry_surface, -np.sign(selected_ray), np.sign(selected_ray))
    normals = np.zeros_like(rays)
    np.put_along_axis(normals, axis[..., None], direction[..., None], axis=-1)
    return distance, normals


def render_frame(obstacles: Iterable[Box], pose: CameraPose, capture_time: float,
                 camera: CameraConfig | None = None,
                 receive_time: float | None = None) -> CameraFrame:
    """Ray-cast opaque boxes and z=0 ground with deterministic world textures.

    Supply obstacle positions at capture_time (for a moving scene, use
    geometry.obstacles_at). No hidden prediction or time interpolation occurs.
    """
    camera = camera or CameraConfig()
    receive_time = capture_time if receive_time is None else receive_time
    if not all(math.isfinite(t) for t in (capture_time, receive_time)) or capture_time < 0 or receive_time < capture_time:
        raise ValueError("require finite 0 <= capture_time <= receive_time")
    origin = np.asarray(pose.position, dtype=np.float64)
    rays = camera_rays(camera, pose)
    shape = (camera.height, camera.width)
    depth = np.full(shape, np.inf, dtype=np.float64)
    ids = np.full(shape, -1, dtype=np.int32)
    # Analytic sky: bright near horizon, darker overhead. Sky has no depth.
    luminance = 0.67 - 0.17 * np.clip(rays[..., 2], 0.0, 1.0)
    ground_distance = np.divide(-origin[2], rays[..., 2], out=np.full(shape, np.inf), where=rays[..., 2] < -1e-12)
    ground_hit = (ground_distance >= camera.near_m) & (ground_distance <= camera.far_m)
    if np.any(ground_hit):
        points = origin + rays[ground_hit] * ground_distance[ground_hit, None]
        checker = (np.floor(points[:, 0] / 0.75) + np.floor(points[:, 1] / 0.75)) % 2.0
        luminance[ground_hit] = 0.24 + 0.13 * checker + 0.025 * np.sin(points[:, 0] * 3.1 + points[:, 1] * 1.7)
        depth[ground_hit] = ground_distance[ground_hit]
        ids[ground_hit] = 0
    light = np.asarray((-0.4, -0.3, 0.866025403784), dtype=np.float64)
    boxes = sorted(obstacles, key=lambda box: box.id)
    if len({b.id for b in boxes}) != len(boxes):
        raise ValueError("box IDs must be unique for deterministic surface IDs")
    for index, box in enumerate(boxes, start=1):
        distances, normals = _box_intersection(origin, rays, box, camera.near_m)
        hit = (distances < depth) & (distances <= camera.far_m)
        if not np.any(hit):
            continue
        points = origin + rays[hit] * distances[hit, None]
        # Texture moves with the object by using local coordinates.
        local = np.maximum(points - np.asarray(box.low), 0.0)
        digest = hashlib.sha256(box.id.encode("utf-8")).digest()
        base = 0.36 + digest[0] / 255.0 * 0.24
        phase = digest[1] / 255.0 * math.tau
        # Tiny tie tolerance keeps exact surface/grid boundaries invariant to
        # floating-point cancellation after a joint camera/object translation.
        checker = np.sum(np.floor(local / 0.5 + 1e-9), axis=1) % 2.0
        texture = 0.86 + 0.20 * checker + 0.10 * np.sin(local @ np.array((4.7, 3.3, 5.9)) + phase)
        lighting = 0.67 + 0.33 * np.clip(normals[hit] @ light, 0.0, 1.0)
        luminance[hit] = base * texture * lighting
        depth[hit], ids[hit] = distances[hit], index
    return CameraFrame(
        luminance=np.clip(luminance, 0.0, 1.0).astype(np.float32),
        depth_m=depth.astype(np.float32), depth_valid_mask=np.isfinite(depth),
        image_valid_mask=np.ones(shape, dtype=np.bool_), surface_ids=ids,
        capture_time=float(capture_time), receive_time=float(receive_time),
        camera=camera, pose=pose,
    )


class FrameGate:
    """Fail-closed image/timestamp gate before a recurrent neural input stream.

    Invalid pixels are rejected, not silently filled. A failed frame does not
    advance state. A long gap requires explicitly creating a new gate and
    resetting the recurrent model; it cannot silently resume the old stream.
    Depth/segmentation are never returned to the neural consumer.
    """
    def __init__(self, camera: CameraConfig, max_age_s: float = 0.25,
                 max_gap_s: float = 0.25) -> None:
        if not all(math.isfinite(t) and t > 0 for t in (max_age_s, max_gap_s)):
            raise ValueError("time limits must be finite and positive")
        self.camera, self.max_age_s, self.max_gap_s = camera, max_age_s, max_gap_s
        self.last_capture_time: float | None = None
        self.last_now: float | None = None

    def pixels(self, frame: CameraFrame, now: float) -> np.ndarray:
        shape = (self.camera.height, self.camera.width)
        if not frame.valid or frame.camera != self.camera:
            raise ValueError("invalid frame or camera changed")
        if not isinstance(frame.luminance, np.ndarray) or frame.luminance.shape != shape or frame.luminance.dtype != np.float32:
            raise ValueError("luminance must be float32 [H,W] for the configured camera")
        if not np.isfinite(frame.luminance).all() or np.any(frame.luminance < 0) or np.any(frame.luminance > 1):
            raise ValueError("luminance must be finite and within [0,1]")
        mask = frame.image_valid_mask
        if not isinstance(mask, np.ndarray) or mask.shape != shape or mask.dtype != np.bool_ or not np.all(mask):
            raise ValueError("all neural input pixels must be valid")
        capture, receive = frame.capture_time, frame.receive_time
        if not all(math.isfinite(t) for t in (capture, receive, now)) or not 0 <= capture <= receive <= now + 1e-9:
            raise ValueError("invalid image timestamps")
        if now - capture > self.max_age_s + 1e-9:
            raise ValueError("stale camera frame")
        if self.last_now is not None and now < self.last_now - 1e-9:
            raise ValueError("clock moved backward; reset the recurrent stream")
        if self.last_capture_time is not None:
            delta = capture - self.last_capture_time
            if delta <= 0:
                raise ValueError("duplicate or out-of-order camera frame")
            if delta > self.max_gap_s + 1e-9:
                raise ValueError("camera gap too long; reset the recurrent stream")
        self.last_capture_time, self.last_now = capture, now
        return frame.luminance.copy()
