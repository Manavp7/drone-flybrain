"""Bounded ROS 2 RGB/depth capture; no flight commands or autopilot transport.

Pure decoding and synchronisation need NumPy only. ROS dependencies are imported
inside capture(). Output is an atomic latest.npz on the same host as perception.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np

PX4_TAG = "v1.16.2"
PX4_COMMIT = "54f0455ffcd755534539a7cf33a09a20bf71d29d"
MAX_PIXELS = 8_500_000
MAX_IMAGE_BYTES = MAX_PIXELS * 8


class CameraError(ValueError):
    pass


def stamp_seconds(stamp) -> float:
    sec, nsec = int(stamp.sec), int(stamp.nanosec)
    if sec < 0 or not 0 <= nsec < 1_000_000_000:
        raise CameraError("Invalid nonnegative ROS source timestamp")
    return sec + nsec / 1_000_000_000


def _dimensions(message) -> tuple[int, int]:
    width, height = int(message.width), int(message.height)
    if not 0 < width <= 4096 or not 0 < height <= 2160 or width * height > MAX_PIXELS:
        raise CameraError("Image dimensions exceed the supported bound")
    return width, height


def decode_image(message) -> np.ndarray:
    """sensor_msgs/Image to owned RGB uint8 or optical-Z float32 metres.

    ROS REP 118: 32FC1 uses metres; 16UC1 uses millimetres. Unknown/zero,
    nonfinite and negative depths become NaN; they are never free space.
    Row padding and byte order are honoured without interpreting padding pixels.
    """
    width, height = _dimensions(message)
    encoding = str(message.encoding)
    if encoding not in ("rgb8", "bgr8", "32FC1", "16UC1"):
        raise CameraError(f"Unsupported image encoding: {encoding}")
    big = int(message.is_bigendian)
    if big not in (0, 1):
        raise CameraError("is_bigendian must be 0 or 1")
    channels = 3 if encoding in ("rgb8", "bgr8") else 1
    dtype = np.dtype("u1" if channels == 3 else ((">" if big else "<") + ("f4" if encoding == "32FC1" else "u2")))
    step = int(message.step)
    minimum = width * channels * dtype.itemsize
    if step < minimum or step * height > MAX_IMAGE_BYTES:
        raise CameraError("Invalid or excessive image row stride")
    try:
        data = memoryview(message.data)
    except TypeError as exc:
        raise CameraError("Image data must provide a contiguous byte buffer") from exc
    if not data.contiguous or data.nbytes != step * height:
        raise CameraError("Image data length does not match step times height")
    if channels == 3:
        result = np.ndarray((height, width, 3), dtype=dtype, buffer=data,
                            strides=(step, 3, 1))
        return result[:, :, ::-1].copy() if encoding == "bgr8" else result.copy()
    result = np.ndarray((height, width), dtype=dtype, buffer=data,
                        strides=(step, dtype.itemsize)).astype(np.float32)
    if encoding == "16UC1":
        result *= np.float32(0.001)
    result[~np.isfinite(result) | (result <= 0)] = np.nan
    return result


@dataclass(frozen=True)
class Calibration:
    width: int
    height: int
    frame_id: str
    stamp: float
    intrinsics: tuple[float, float, float, float]


def camera_intrinsics(message) -> Calibration:
    """Accept the simple undistorted, unbinned Gazebo pinhole camera only.

    Real camera rectification/resampling must happen upstream. Equal image shape
    is not proof of depth registration; registration remains an explicit opt-in.
    """
    width, height = _dimensions(message)
    matrix = np.asarray(message.k, dtype=float)
    if matrix.shape != (9,) or not np.all(np.isfinite(matrix)):
        raise CameraError("CameraInfo K must have nine finite entries")
    fx, fy, cx, cy = matrix[[0, 4, 2, 5]]
    if fx <= 0 or fy <= 0 or not 0 <= cx < width or not 0 <= cy < height:
        raise CameraError("Uncalibrated or invalid pinhole intrinsics")
    if not np.allclose(matrix[[1, 3, 6, 7, 8]], [0, 0, 0, 0, 1], atol=1e-9, rtol=0):
        raise CameraError("Skewed or noncanonical camera matrix is unsupported")
    distortion = np.asarray(message.d, dtype=float)
    if not np.all(np.isfinite(distortion)) or np.any(np.abs(distortion) > 1e-9):
        raise CameraError("Distorted images require upstream rectification")
    if int(getattr(message, "binning_x", 0)) not in (0, 1) or int(getattr(message, "binning_y", 0)) not in (0, 1):
        raise CameraError("Binned images require adjusted upstream calibration")
    roi = getattr(message, "roi", None)
    if roi and (roi.x_offset or roi.y_offset or roi.width not in (0, width) or roi.height not in (0, height)):
        raise CameraError("Cropped images require adjusted upstream calibration")
    frame_id = str(message.header.frame_id)
    if not frame_id or len(frame_id) > 256:
        raise CameraError("CameraInfo optical frame_id is missing or excessive")
    return Calibration(width, height, frame_id, stamp_seconds(message.header.stamp),
                       tuple(float(v) for v in (fx, fy, cx, cy)))


@dataclass(frozen=True)
class ReceivedImage:
    image: np.ndarray
    stamp: float
    received_monotonic: float
    frame_id: str
    age_at_receive: float


class LatestFrameSynchronizer:
    """Single pending RGB, single latest depth, single calibration. No backlog.

    A matching pair can publish immediately; otherwise RGB publishes after a
    bounded wall-time wait. A new RGB replaces an unconsumed older RGB. Only a
    verified /clock rewind resets the stream; old individual frames are dropped.
    Caller uses one thread (the ROS executor below is single-threaded).
    """
    def __init__(self, *, registration_verified=False, pair_tolerance_s=0.03,
                 max_pair_wait_s=0.04, max_frame_age_s=0.3,
                 max_clock_silence_s=1.0, max_calibration_age_s=2.0):
        for name, value, upper in (("pair_tolerance_s", pair_tolerance_s, 0.2),
                                   ("max_pair_wait_s", max_pair_wait_s, 0.2),
                                   ("max_frame_age_s", max_frame_age_s, 2.0),
                                   ("max_clock_silence_s", max_clock_silence_s, 5.0),
                                   ("max_calibration_age_s", max_calibration_age_s, 10.0)):
            if not math.isfinite(value) or not 0 < value <= upper:
                raise CameraError(f"{name} must be positive and at most {upper}")
        self.registration_opt_in = bool(registration_verified)
        self.tolerance = pair_tolerance_s
        self.wait = max_pair_wait_s
        self.max_age = max_frame_age_s
        self.max_clock_silence = max_clock_silence_s
        self.max_calibration_age = max_calibration_age_s
        self.clock = None
        self.clock_received = None
        self.counters = {k: 0 for k in ("rgb", "depth", "published", "paired", "replaced_rgb", "rejected", "clock_resets")}
        self.reset()

    def reset(self):
        self.stream_id = uuid.uuid4().hex
        self.sequence = 0
        self.pending_rgb = None
        self.pending_deadline = None
        self.depth = None
        self.calibration = None
        self.last_stamps = {"rgb": -math.inf, "depth": -math.inf}

    def ingest_clock(self, source_time: float, received_monotonic: float):
        if not math.isfinite(source_time) or source_time < 0 or not math.isfinite(received_monotonic) or received_monotonic < 0:
            raise CameraError("Invalid clock sample")
        if self.clock_received is not None and received_monotonic < self.clock_received:
            raise CameraError("Local monotonic clock moved backwards")
        if self.clock is not None and source_time < self.clock - 1e-9:
            self.reset()
            self.counters["clock_resets"] += 1
        self.clock, self.clock_received = source_time, received_monotonic

    def _freshness(self, stamp, received_monotonic):
        if self.clock is None or self.clock_received is None:
            raise CameraError("Simulation clock has not been observed")
        if not math.isfinite(received_monotonic) or not 0 <= received_monotonic - self.clock_received <= self.max_clock_silence:
            raise CameraError("Simulation clock is stale or local clock is invalid")
        source_age = self.clock - stamp
        # The latest /clock sample also ages on this host. A delayed or paused
        # clock publisher must not make a buffered camera frame look current.
        # At the documented real-time factor 1, adding clock observation age is
        # conservative; pausing simulation may reject otherwise usable frames.
        clock_age = received_monotonic - self.clock_received
        age_upper_bound = max(0.0, source_age) + clock_age
        # A tiny source skew is tolerated because separate topics can arrive in
        # either order. Do not let that tolerance subtract elapsed wall time.
        if source_age < -0.03 or age_upper_bound > self.max_age:
            raise CameraError("Image source timestamp is stale or in the future")
        return age_upper_bound

    def ingest_info(self, message):
        try:
            candidate = camera_intrinsics(message)
        except CameraError:
            self.calibration = None
            self.counters["rejected"] += 1
            raise
        self.calibration = candidate

    def ingest(self, kind: str, message, received_monotonic: float):
        if kind not in ("rgb", "depth"):
            raise CameraError("Expected rgb or depth channel")
        try:
            stamp = stamp_seconds(message.header.stamp)
            age = self._freshness(stamp, received_monotonic)
            if stamp <= self.last_stamps[kind]:
                raise CameraError("Duplicate or out-of-order image")
            frame_id = str(message.header.frame_id)
            if not frame_id or len(frame_id) > 256:
                raise CameraError("Image optical frame_id is missing or excessive")
            expected = ("rgb8", "bgr8") if kind == "rgb" else ("32FC1", "16UC1")
            if message.encoding not in expected:
                raise CameraError("Image encoding does not match the channel")
            decoded = decode_image(message)
            if kind == "rgb" and min(decoded.shape[:2]) < 2:
                raise CameraError("Perception requires RGB dimensions of at least 2 by 2")
            sample = ReceivedImage(decoded, stamp, received_monotonic, frame_id, age)
        except CameraError:
            self.counters["rejected"] += 1
            raise
        self.last_stamps[kind] = stamp
        self.counters[kind] += 1
        if kind == "rgb":
            if self.pending_rgb is not None:
                self.counters["replaced_rgb"] += 1
            else:
                self.pending_deadline = received_monotonic + self.wait
            self.pending_rgb = sample
        else:
            self.depth = sample
        return self.flush(received_monotonic)

    def flush(self, now_monotonic: float):
        rgb = self.pending_rgb
        if rgb is None:
            return None
        waited = now_monotonic - rgb.received_monotonic
        if not math.isfinite(now_monotonic) or waited < 0:
            raise CameraError("Local monotonic time is invalid")
        if rgb.age_at_receive + waited > self.max_age:
            self.pending_rgb = None
            self.counters["rejected"] += 1
            return None
        calibration = self.calibration
        calibration_valid = bool(calibration and
            calibration.frame_id == rgb.frame_id and
            (calibration.height, calibration.width) == rgb.image.shape[:2] and
            -self.tolerance <= rgb.stamp - calibration.stamp <= self.max_calibration_age)
        depth = self.depth
        paired = bool(self.registration_opt_in and calibration_valid and depth and
            depth.frame_id == rgb.frame_id and depth.image.shape == rgb.image.shape[:2] and
            abs(depth.stamp - rgb.stamp) <= self.tolerance and
            0 <= now_monotonic - depth.received_monotonic <= self.max_age and
            depth.age_at_receive + now_monotonic - depth.received_monotonic <= self.max_age)
        if not paired and self.registration_opt_in and now_monotonic < self.pending_deadline:
            return None
        self.pending_rgb = None
        self.pending_deadline = None
        self.sequence += 1
        packet = {"image_rgb": rgb.image,
                  "capture_time_s": np.asarray(rgb.stamp),
                  "received_monotonic_s": np.asarray(rgb.received_monotonic),
                  "capture_age_at_receive_s": np.asarray(rgb.age_at_receive),
                  "sequence": np.asarray(self.sequence, dtype=np.int64),
                  "frame_id": np.asarray(rgb.frame_id),
                  "clock_domain": np.asarray("ros"),
                  "stream_id": np.asarray(self.stream_id),
                  "registration_verified": np.asarray(paired)}
        if calibration_valid:
            packet["camera_intrinsics"] = np.asarray(calibration.intrinsics, dtype=np.float64)
        if paired:
            packet["depth_m"] = depth.image
            packet["depth_time_s"] = np.asarray(depth.stamp)
            self.counters["paired"] += 1
        self.counters["published"] += 1
        return packet


class AtomicFrameWriter:
    """A single durable latest.npz; replacement is atomic on one filesystem."""
    def __init__(self, spool: Path):
        self.spool = Path(spool)
        self.spool.mkdir(parents=True, exist_ok=True)
        self.path = self.spool / "latest.npz"

    def write(self, packet: dict) -> Path:
        # No object arrays / pickle content are admitted into the spool.
        if not packet or any(np.asarray(v).dtype.hasobject for v in packet.values()):
            raise CameraError("Packet cannot contain object arrays")
        fd, temporary = tempfile.mkstemp(prefix=".frame-", suffix=".npz", dir=self.spool)
        try:
            with os.fdopen(fd, "wb") as handle:
                np.savez(handle, **packet)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            directory_fd = os.open(self.spool, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.path


def environment_status(px4_source: Path | None = None) -> dict:
    programs = {name: shutil.which(name) for name in ("gz", "ros2")}
    modules = {name: importlib.util.find_spec(name) is not None for name in ("rclpy", "sensor_msgs", "rosgraph_msgs")}
    missing = [name for name, value in {**programs, **modules}.items() if not value]
    checkout = None
    if px4_source:
        try:
            actual = subprocess.check_output(["git", "-C", str(px4_source), "rev-parse", "HEAD"], text=True, timeout=5).strip()
            checkout = {"actual_commit": actual, "matches_pin": actual == PX4_COMMIT}
            if actual != PX4_COMMIT:
                missing.append("pinned PX4 checkout")
        except (OSError, subprocess.SubprocessError):
            checkout = {"matches_pin": False}
            missing.append("readable PX4 checkout")
    return {"camera_dependencies_available": not missing, "missing": missing,
            "programs": programs, "python_modules": modules, "px4_checkout": checkout,
            "target_px4_tag": PX4_TAG, "target_px4_commit": PX4_COMMIT,
            "target_model": "gz_x500_depth", "autostart": 4002,
            "actual_sitl_run": False, "actual_ros_camera_run": False,
            "observation_only": True,
            "unverified": ["ROS/Gazebo ABI compatibility", "camera topics and registration", "rendering backend", "runtime timing"]}


def bridge_configuration(rgb_topic, depth_topic, info_topic, clock_topic="/clock") -> str:
    """Generate YAML for discovered Gazebo topics, strictly GZ_TO_ROS."""
    mappings = [(rgb_topic, "/inspection/rgb", "sensor_msgs/msg/Image", "gz.msgs.Image"),
                (info_topic, "/inspection/camera_info", "sensor_msgs/msg/CameraInfo", "gz.msgs.CameraInfo"),
                (clock_topic, "/clock", "rosgraph_msgs/msg/Clock", "gz.msgs.Clock")]
    if depth_topic:
        mappings.insert(1, (depth_topic, "/inspection/depth", "sensor_msgs/msg/Image", "gz.msgs.Image"))
    if len({entry[0] for entry in mappings}) != len(mappings):
        raise CameraError("Each Gazebo stream must use a distinct topic")
    lines = ["# Gazebo to ROS camera observations only; no command bridge."]
    for gz_topic, ros_topic, ros_type, gz_type in mappings:
        if not isinstance(gz_topic, str) or not re.fullmatch(r"/[A-Za-z0-9_/]+", gz_topic) or len(gz_topic) > 1024:
            raise CameraError("Expected an absolute discovered Gazebo topic")
        values = {"ros_topic_name": ros_topic, "gz_topic_name": gz_topic,
                  "ros_type_name": ros_type, "gz_type_name": gz_type,
                  "direction": "GZ_TO_ROS", "subscriber_queue": 1, "publisher_queue": 1,
                  "lazy": False}
        for index, (name, value) in enumerate(values.items()):
            lines.append(("- " if index == 0 else "  ") + name + ": " + json.dumps(value))
    return "\n".join(lines) + "\n"


def capture(args, ros_args=None) -> int:
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import Image, CameraInfo
        from rosgraph_msgs.msg import Clock
    except ImportError as exc:
        raise CameraError("ROS 2 camera dependencies unavailable; run doctor and read docs/PX4_CAMERA.md") from exc
    sync = LatestFrameSynchronizer(registration_verified=args.registered_depth,
                                 pair_tolerance_s=args.pair_tolerance,
                                 max_pair_wait_s=args.pair_wait,
                                 max_frame_age_s=args.max_frame_age)
    writer = AtomicFrameWriter(args.spool)
    rclpy.init(args=ros_args)
    node = Node("inspection_camera_capture", automatically_declare_parameters_from_overrides=True)
    try:
        if not node.has_parameter("use_sim_time") or not node.get_parameter("use_sim_time").value:
            raise CameraError("Capture requires --ros-args -p use_sim_time:=true and a bridged /clock")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        last_warning = [-math.inf]
        def guarded(fn):
            def callback(message=None):
                try:
                    packet = fn(message)
                    if packet is not None:
                        writer.write(packet)
                except (CameraError, OSError) as exc:
                    now = time.monotonic()
                    if now - last_warning[0] > 2:
                        node.get_logger().warning(str(exc))
                        last_warning[0] = now
            return callback
        def clock_callback(message):
            sync.ingest_clock(stamp_seconds(message.clock), time.monotonic())
        node.create_subscription(Clock, "/clock", guarded(clock_callback), qos)
        node.create_subscription(CameraInfo, args.info_topic, guarded(sync.ingest_info), qos)
        node.create_subscription(Image, args.rgb_topic,
                                 guarded(lambda msg: sync.ingest("rgb", msg, time.monotonic())), qos)
        if args.depth_topic:
            node.create_subscription(Image, args.depth_topic,
                                     guarded(lambda msg: sync.ingest("depth", msg, time.monotonic())), qos)
        # Wall timer, so a missing/paused simulation clock cannot strand pending RGB.
        from rclpy.clock import Clock as RclClock, ClockType
        node.create_timer(0.01, guarded(lambda _: sync.flush(time.monotonic())),
                          clock=RclClock(clock_type=ClockType.STEADY_TIME))
        node.get_logger().info(f"Observation-only capture writing {writer.path}; no flight commands")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print(json.dumps({"counters": sync.counters, "spool": str(writer.path),
                          "actual_ros_camera_run": sync.counters["published"] > 0,
                          "flight_commands_sent": 0}, indent=2))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--px4-source", type=Path)
    config = commands.add_parser("bridge-config")
    config.add_argument("--rgb-topic", required=True)
    config.add_argument("--depth-topic")
    config.add_argument("--info-topic", required=True)
    config.add_argument("--clock-topic", default="/clock")
    config.add_argument("--output", type=Path, required=True)
    live = commands.add_parser("capture")
    live.add_argument("--rgb-topic", default="/inspection/rgb")
    live.add_argument("--depth-topic")
    live.add_argument("--info-topic", default="/inspection/camera_info")
    live.add_argument("--spool", type=Path, required=True)
    live.add_argument("--registered-depth", action="store_true",
                      help="Explicitly assert verified depth registration to RGB optical pixels")
    live.add_argument("--pair-tolerance", type=float, default=0.03)
    live.add_argument("--pair-wait", type=float, default=0.04)
    live.add_argument("--max-frame-age", type=float, default=0.3)
    args, extra = parser.parse_known_args(argv)
    try:
        if args.command == "doctor":
            if extra:
                parser.error("Unexpected arguments")
            status = environment_status(args.px4_source)
            print(json.dumps(status, indent=2))
            return 0 if status["camera_dependencies_available"] else 2
        if args.command == "bridge-config":
            if extra:
                parser.error("Unexpected arguments")
            result = bridge_configuration(args.rgb_topic, args.depth_topic, args.info_topic, args.clock_topic)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(result)
            print(args.output)
            return 0
        if args.registered_depth and not args.depth_topic:
            raise CameraError("--registered-depth requires --depth-topic")
        return capture(args, extra)
    except (CameraError, OSError) as exc:
        print(f"Camera integration unavailable: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
