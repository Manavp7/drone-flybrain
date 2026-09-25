"""Local PX4/Gazebo SITL only. No serial, remote target, or hardware adapter.

MAVLink transport is lazy-loaded so exports/tests need only the standard library.
Protocol unit tests do not constitute a PX4 flight test.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time
from typing import Callable
import xml.etree.ElementTree as ET

from .autonomy import Autonomy
from .contracts import BrainOutput, Guidance, Observation, Scenario, Vec3
from .geometry import distance, norm, swept_collision, within_bounds
from .scenarios import generate_scenario

PX4_COMMIT = "54f0455ffcd755534539a7cf33a09a20bf71d29d"
PX4_TAG = "v1.16.2"
SITL_ADDRESS = "udpin:127.0.0.1:14540"
POSITION_MASK = 8 | 16 | 32 | 64 | 128 | 256 | 1024 | 2048
VELOCITY_MASK = 1 | 2 | 4 | 64 | 128 | 256 | 1024 | 2048


class SitlError(RuntimeError):
    pass


def enu_to_ned(vector: Vec3) -> Vec3:
    """Gazebo/project: x East, y North, z Up; PX4: North, East, Down."""
    return vector[1], vector[0], -vector[2]


def ned_to_enu(vector: Vec3) -> Vec3:
    return vector[1], vector[0], -vector[2]


@dataclass(frozen=True)
class LocalFrame:
    """Align local PX4 estimator origin at ground before arming.

    This is an assumed alignment to the declared Gazebo spawn, not localization
    against an image or an independent measurement of simulator ground truth.
    """
    spawn_enu: Vec3
    initial_ned: Vec3

    def position_enu(self, position_ned: Vec3) -> Vec3:
        delta = ned_to_enu(tuple(a-b for a, b in zip(position_ned, self.initial_ned)))
        return tuple(a+b for a, b in zip(self.spawn_enu, delta))

    def position_ned(self, position_enu: Vec3) -> Vec3:
        delta = enu_to_ned(tuple(a-b for a, b in zip(position_enu, self.spawn_enu)))
        return tuple(a+b for a, b in zip(self.initial_ned, delta))


def scenario_hash(scenario: Scenario) -> str:
    return hashlib.sha256(json.dumps(asdict(scenario), sort_keys=True).encode()).hexdigest()


def static_sitl_scenario(seed: int) -> Scenario:
    # Fault regimes from the reduced model must never silently be called Gazebo
    # wind, battery, latency, or moving-object experiments.
    return generate_scenario(seed, "nominal")


def export_world(scenario: Scenario, output: Path) -> dict:
    if scenario.category != "nominal" or any(norm(b.velocity) for b in scenario.obstacles):
        raise SitlError("SITL export currently supports nominal static geometry only")
    output.mkdir(parents=True, exist_ok=True)
    root = ET.Element("sdf", version="1.9")
    world = ET.SubElement(root, "world", name="flybrain_inspection")
    physics = ET.SubElement(world, "physics", name="1ms", type="ignored")
    ET.SubElement(physics, "max_step_size").text = "0.001"
    ET.SubElement(physics, "real_time_factor").text = "1.0"
    for filename, name in (
        ("gz-sim-physics-system", "gz::sim::systems::Physics"),
        ("gz-sim-user-commands-system", "gz::sim::systems::UserCommands"),
        ("gz-sim-scene-broadcaster-system", "gz::sim::systems::SceneBroadcaster"),
        ("gz-sim-imu-system", "gz::sim::systems::Imu"),
        ("gz-sim-air-pressure-system", "gz::sim::systems::AirPressure"),
        ("gz-sim-navsat-system", "gz::sim::systems::NavSat"),
        ("gz-sim-magnetometer-system", "gz::sim::systems::Magnetometer"),
    ):
        ET.SubElement(world, "plugin", filename=filename, name=name)
    ET.SubElement(world, "gravity").text = "0 0 -9.80665"
    ET.SubElement(world, "magnetic_field").text = "2.15e-5 0 4.27e-5"
    spherical = ET.SubElement(world, "spherical_coordinates")
    for tag, value in (("surface_model", "EARTH_WGS84"), ("world_frame_orientation", "ENU"),
                       ("latitude_deg", "47.397742"), ("longitude_deg", "8.545594"),
                       ("elevation", "488.0"), ("heading_deg", "0")):
        ET.SubElement(spherical, tag).text = value
    light = ET.SubElement(world, "light", name="sun", type="directional")
    ET.SubElement(light, "pose").text = "0 0 100 0 0 0"
    ET.SubElement(light, "diffuse").text = "0.8 0.8 0.8 1"
    ET.SubElement(light, "direction").text = "-0.5 0.1 -0.9"

    def box(name, center, size, color):
        model = ET.SubElement(world, "model", name=name)
        ET.SubElement(model, "static").text = "true"
        ET.SubElement(model, "pose").text = " ".join(map(str, center)) + " 0 0 0"
        link = ET.SubElement(model, "link", name="body")
        for tag in ("collision", "visual"):
            item = ET.SubElement(link, tag, name=tag)
            geom = ET.SubElement(item, "geometry")
            ET.SubElement(ET.SubElement(geom, "box"), "size").text = " ".join(map(str, size))
            if tag == "visual":
                material = ET.SubElement(item, "material")
                ET.SubElement(material, "ambient").text = color
                ET.SubElement(material, "diffuse").text = color
    box("ground", (24, 18, -0.1), (100, 100, 0.2), "0.3 0.35 0.4 1")
    for obstacle in scenario.obstacles:
        box(obstacle.id, tuple((a+b)/2 for a, b in zip(obstacle.low, obstacle.high)),
            tuple(b-a for a, b in zip(obstacle.low, obstacle.high)), "0.6 0.65 0.7 1")
    ET.indent(root)
    path = output / "flybrain_inspection.sdf"
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    manifest = {"scenario": asdict(scenario), "scenario_sha256": scenario_hash(scenario),
                "world": str(path.resolve()), "world_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "frame": "Gazebo/project ENU; PX4 local NED aligned at ground spawn",
                "spawn_enu": [scenario.home[0], scenario.home[1], 0.0],
                "implemented": "static collision/visual boxes, ground, physical x500 vehicle from PX4",
                "not_implemented": ["dynamic fixtures", "fault injection", "camera inputs", "contact truth scoring"]}
    (output / "world_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest


def environment_status(px4_source: Path | None = None) -> dict:
    binaries = {name: shutil.which(name) for name in ("git", "make", "cmake", "ninja", "gz")}
    binary = (px4_source / "build/px4_sitl_default/bin/px4") if px4_source else None
    missing = [name for name, value in binaries.items() if not value]
    if not importlib.util.find_spec("pymavlink"):
        missing.append("pymavlink==2.4.49")
    if not binary or not binary.is_file():
        missing.append("built PX4 SITL executable")
    if sys.platform != "linux":
        missing.append("Linux host with /proc process identity")
    return {"ready": not missing, "missing": missing, "tools": binaries,
            "px4_binary": str(binary) if binary else None,
            "actual_sitl_run": False, "target_px4_tag": PX4_TAG, "target_px4_commit": PX4_COMMIT}


def process_identity(pid: int) -> dict:
    directory = Path("/proc") / str(pid)
    try:
        stat = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": stat[19], "exe": str((directory / "exe").resolve(strict=True)),
                "environment": dict(item.split(b"=", 1) for item in (directory / "environ").read_bytes().split(b"\0") if b"=" in item)}
    except (OSError, ValueError, IndexError) as exc:
        raise SitlError("The locally launched PX4 process is unavailable") from exc


@dataclass
class ProcessLease:
    pid: int
    start_ticks: str
    executable: str
    token: str

    def verify(self) -> None:
        identity = process_identity(self.pid)
        expected = {b"FLYBRAIN_SITL_TOKEN": self.token.encode(), b"PX4_SIM_MODEL": b"gz_x500",
                    b"PX4_SYS_AUTOSTART": b"4001", b"PX4_GZ_WORLD": b"flybrain_inspection"}
        if identity["start_ticks"] != self.start_ticks or identity["exe"] != self.executable:
            raise SitlError("SITL process identity changed")
        if any(identity["environment"].get(k) != v for k, v in expected.items()):
            raise SitlError("SITL process launch identity does not match")


@dataclass
class Telemetry:
    position_ned: Vec3 = (math.nan,)*3
    velocity_ned: Vec3 = (math.nan,)*3
    position_at: float = -math.inf
    heartbeat_at: float = -math.inf
    estimator_at: float = -math.inf
    battery_at: float = -math.inf
    estimator_flags: int = 0
    battery_fraction: float = math.nan
    armed: bool = False
    offboard: bool = False
    landed: bool = False
    landed_at: float = -math.inf
    boot_ms: int | None = None
    clock_reset: bool = False

    def health_error(self, now: float) -> str:
        if self.clock_reset:
            return "autopilot_clock_reset"
        if now-self.heartbeat_at > 1.5:
            return "heartbeat_lost"
        if now-self.position_at > 0.4:
            return "position_stale"
        if not all(math.isfinite(v) for v in self.position_ned+self.velocity_ned):
            return "nonfinite_telemetry"
        # Attitude, horizontal/vertical velocity, relative horizontal and absolute
        # vertical position flags from MAVLink ESTIMATOR_STATUS_FLAGS.
        required = 1 | 2 | 4 | 8 | 32
        if now-self.estimator_at > 1.5 or self.estimator_flags & required != required:
            return "estimator_unhealthy"
        if now-self.battery_at > 3 or not 0 <= self.battery_fraction <= 1:
            return "battery_telemetry_unavailable"
        if self.battery_fraction < 0.2:
            return "battery_reserve"
        return ""

    def ingest_position(self, message, now: float) -> None:
        boot = int(message.time_boot_ms)
        if self.boot_ms is not None and boot < self.boot_ms:
            self.clock_reset = True
        if self.boot_ms is None or boot > self.boot_ms:
            self.position_ned = (message.x, message.y, message.z)
            self.velocity_ned = (message.vx, message.vy, message.vz)
            self.position_at, self.boot_ms = now, boot


def expired_velocity(guidance: Guidance, now: float, elapsed: float) -> Vec3:
    if (not math.isfinite(guidance.valid_until) or elapsed >= guidance.valid_until
            or not all(math.isfinite(x) for x in guidance.velocity)):
        raise SitlError("Controller command expired or became nonfinite")
    if norm(guidance.velocity) > 2.50001:
        raise SitlError("Controller exceeded SITL command speed limit")
    return enu_to_ned(guidance.velocity)


def decode_parameter(value: float, param_type: int) -> float:
    # PX4 standard parameter protocol uses bytewise integer encoding.
    if param_type == 6:  # MAV_PARAM_TYPE_INT32
        return float(struct.unpack("<i", struct.pack("<f", value))[0])
    if param_type == 9:  # MAV_PARAM_TYPE_REAL32
        return float(value)
    raise SitlError(f"Unsupported parameter type {param_type}")


class MavlinkLink:
    """Single-thread bounded polling. Stalls stop the actual setpoint stream."""
    def __init__(self, lease: ProcessLease):
        try:
            from pymavlink import mavutil
        except ImportError as exc:
            raise SitlError("Install integrations/px4/requirements.txt to run real SITL") from exc
        self.lease = lease
        lease.verify()
        self.connection = mavutil.mavlink_connection(SITL_ADDRESS, source_system=245, source_component=191)
        self.telemetry = Telemetry()
        self.params: dict[str, tuple[float, float]] = {}
        self.acks: dict[int, tuple[int, float]] = {}
        self.last_tx_heartbeat = -math.inf
        self.identified = False
        self.arm_authorized = False

    def close(self):
        self.connection.close()

    def poll(self) -> None:
        self.lease.verify()
        now = time.monotonic()
        if now-self.last_tx_heartbeat >= 0.5:
            self.connection.mav.heartbeat_send(6, 8, 0, 0, 4)  # GCS, invalid autopilot, active
            self.last_tx_heartbeat = now
        # Do not permit an unbounded receive burst to starve command deadlines.
        for _ in range(200):
            msg = self.connection.recv_match(blocking=False)
            if msg is None:
                break
            if msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                continue
            kind = msg.get_type()
            if kind == "HEARTBEAT":
                if msg.autopilot != 12 or msg.type != 2:
                    raise SitlError("Expected PX4 quadrotor heartbeat")
                self.identified = True
                self.telemetry.heartbeat_at = now
                self.telemetry.armed = bool(msg.base_mode & 128)
                self.telemetry.offboard = ((msg.custom_mode >> 16) & 255) == 6
            elif kind == "LOCAL_POSITION_NED":
                self.telemetry.ingest_position(msg, now)
            elif kind == "ESTIMATOR_STATUS":
                self.telemetry.estimator_flags, self.telemetry.estimator_at = int(msg.flags), now
            elif kind == "SYS_STATUS":
                self.telemetry.battery_fraction = msg.battery_remaining / 100 if msg.battery_remaining >= 0 else math.nan
                self.telemetry.battery_at = now
            elif kind == "EXTENDED_SYS_STATE":
                self.telemetry.landed, self.telemetry.landed_at = msg.landed_state == 1, now
            elif kind == "PARAM_VALUE":
                key = msg.param_id.decode() if isinstance(msg.param_id, bytes) else msg.param_id
                self.params[key.rstrip("\0")] = (decode_parameter(msg.param_value, msg.param_type), now)
            elif kind == "COMMAND_ACK":
                self.acks[int(msg.command)] = (int(msg.result), now)

    def wait_for(self, predicate: Callable[[], bool], timeout: float, pump=None) -> None:
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            self.poll()
            if pump:
                pump()
            if predicate():
                return
            time.sleep(0.05)
        raise SitlError("Timed out waiting for PX4 telemetry or command acknowledgement")

    def command(self, command: int, values=(), pump=None) -> None:
        self.lease.verify()
        if command == 400 and values and values[0] == 1 and not self.arm_authorized:
            raise SitlError("Arming requires verified local simulation identity")
        started = time.monotonic()
        args = tuple(values)+(0.0,)*(7-len(values))
        self.connection.mav.command_long_send(1, 1, command, 0, *args)
        self.wait_for(lambda: command in self.acks and self.acks[command][1] >= started and self.acks[command][0] != 5,
                      3.0, pump)
        if self.acks[command][0] != 0:
            raise SitlError(f"PX4 rejected command {command}, result {self.acks[command][0]}")

    def get_parameter(self, name: str) -> float:
        started = time.monotonic()
        self.connection.mav.param_request_read_send(1, 1, name.encode(), -1)
        self.wait_for(lambda: name in self.params and self.params[name][1] >= started, 4)
        return self.params[name][0]

    def set_parameter(self, name: str, value: float, integer=False) -> None:
        self.lease.verify()
        if not self.arm_authorized:
            raise SitlError("Parameter writes require verified local simulation identity")
        encoded = struct.unpack("<f", struct.pack("<i", int(value)))[0] if integer else value
        started = time.monotonic()
        self.connection.mav.param_set_send(1, 1, name.encode(), encoded, 6 if integer else 9)
        self.wait_for(lambda: name in self.params and self.params[name][1] >= started, 4)
        if not math.isclose(self.params[name][0], value, abs_tol=0.001):
            raise SitlError(f"PX4 parameter {name} did not retain requested value")

    def verify_simulator(self) -> None:
        self.wait_for(lambda: self.identified, 30)
        if self.telemetry.armed:
            raise SitlError("SITL must start disarmed")
        if self.get_parameter("SIM_GZ_EN") != 1 or self.get_parameter("SYS_AUTOSTART") != 4001:
            raise SitlError("PX4 is not the expected Gazebo X500 simulator")
        self.lease.verify()
        self.arm_authorized = True

    def setpoint(self, vector_ned: Vec3, position=False) -> None:
        self.lease.verify()
        if not all(math.isfinite(v) for v in vector_ned):
            raise SitlError("Nonfinite NED setpoint")
        p, v = (vector_ned, (0,0,0)) if position else ((0,0,0), vector_ned)
        self.connection.mav.set_position_target_local_ned_send(
            int(time.monotonic()*1000) & 0xFFFFFFFF, 1, 1, 1,
            POSITION_MASK if position else VELOCITY_MASK, *p, *v, 0, 0, 0, 0, 0)


def run_mission(lease: ProcessLease, scenario: Scenario, output: Path) -> dict:
    """Run actual PX4 transport; no mock path exists in this entry point."""
    link = MavlinkLink(lease)
    log = {"backend": "PX4 SITL + Gazebo", "px4_commit": PX4_COMMIT,
           "scenario_sha256": scenario_hash(scenario), "actual_sitl_run": False,
           "perception": "PX4 local estimator + declared static world map",
           "research_model_in_control": False, "events": [], "trajectory": [],
           "mission_complete": False, "landed": False, "collision_truth_available": False}
    armed = False
    arming_requested = False
    mission_epoch = None
    previous_position = None
    terminal = "setup_failed"
    try:
        link.verify_simulator()
        for message_id, hz in ((32, 20), (230, 5), (1, 2), (245, 2)):
            link.command(511, (message_id, 1_000_000/hz))
        link.set_parameter("COM_OBL_RC_ACT", 4, integer=True)
        link.set_parameter("COM_OF_LOSS_T", 0.5)
        link.set_parameter("COM_RC_IN_MODE", 4, integer=True)
        link.wait_for(lambda: not link.telemetry.health_error(time.monotonic()), 60)
        frame = LocalFrame((scenario.home[0], scenario.home[1], 0.0), link.telemetry.position_ned)
        home_ned = frame.position_ned(scenario.home)
        def prime():
            error = link.telemetry.health_error(time.monotonic())
            if error:
                raise SitlError(error)
            link.setpoint(home_ned, position=True)
        prime_started = time.monotonic()
        link.wait_for(lambda: time.monotonic()-prime_started >= 1.2, 3, prime)
        link.command(176, (1, 6, 0), prime)  # MAV_CMD_DO_SET_MODE; PX4 main OFFBOARD
        arming_requested = True
        link.command(400, (1,), prime)
        armed = True
        log["actual_sitl_run"] = True
        link.wait_for(lambda: link.telemetry.armed and link.telemetry.offboard, 3, prime)
        link.wait_for(lambda: distance(frame.position_enu(link.telemetry.position_ned), scenario.home) < 0.35
                      and norm(link.telemetry.velocity_ned) < 0.4, 30, prime)
        controller = Autonomy(scenario)
        mission_epoch = time.monotonic()
        command = None
        next_control = mission_epoch
        start = mission_epoch
        while time.monotonic()-start < scenario.max_time:
            loop_started = time.monotonic()
            link.poll()
            elapsed = loop_started-mission_epoch
            error = link.telemetry.health_error(loop_started)
            if error:
                raise SitlError(error)
            if not link.telemetry.armed or not link.telemetry.offboard:
                raise SitlError("PX4 left armed offboard mission state")
            position = frame.position_enu(link.telemetry.position_ned)
            velocity = ned_to_enu(link.telemetry.velocity_ned)
            if not within_bounds(position, scenario.bounds, radius=0.45):
                raise SitlError("estimated_geofence_violation")
            if previous_position and swept_collision(previous_position, position, scenario.obstacles, radius=0.45):
                raise SitlError("estimated_static_obstacle_contact")
            previous_position = position
            if loop_started >= next_control:
                observation = Observation(elapsed-(loop_started-link.telemetry.position_at), elapsed,
                    position, velocity, scenario.obstacles,
                    scenario.initial_battery_wh*link.telemetry.battery_fraction)
                command = controller.update(observation, elapsed, BrainOutput())
                next_control = loop_started+0.2
                log["trajectory"].append({"t": elapsed, "position_enu": position,
                    "velocity_enu": velocity, "mode": command.mode,
                    "battery_fraction": link.telemetry.battery_fraction})
                if controller.returned_home:
                    log["mission_complete"] = bool(controller.mission_complete and distance(position, scenario.home) <= 1
                                                   and norm(velocity) <= 0.6)
                    terminal = "inspection_returned" if log["mission_complete"] else "aborted_returned"
                    break
            # Compute elapsed again after control work; an overrun cannot renew its own stale command.
            after_control = time.monotonic()
            velocity_ned = expired_velocity(command, after_control, after_control-mission_epoch)
            link.setpoint(velocity_ned)
            time.sleep(max(0.0, 0.05-(time.monotonic()-loop_started)))
        else:
            terminal = "mission_timeout"
    except (SitlError, OSError, KeyboardInterrupt) as exc:
        terminal = str(exc) or "interrupted"
        log["events"].append({"type": "abort", "reason": terminal})
    finally:
        if armed or arming_requested:
            # End streaming and request PX4's LAND. Never airborne-disarm. If
            # telemetry/ACK is lost, PX4's configured offboard timeout remains.
            try:
                link.command(21)
                link.wait_for(lambda: link.telemetry.landed and time.monotonic()-link.telemetry.landed_at < 2
                              and not link.telemetry.armed, 45)
                log["landed"] = True
            except (SitlError, OSError) as exc:
                log["events"].append({"type": "land_unconfirmed", "reason": str(exc)})
        log["outcome"] = terminal
        log["mission_complete"] = bool(log["mission_complete"] and log["landed"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(log, indent=2, allow_nan=False)+"\n")
        link.close()
    return log


def run_local(px4_source: Path, output: Path, seed: int) -> dict:
    status = environment_status(px4_source)
    if not status["ready"]:
        raise SitlError("Missing: "+", ".join(status["missing"]))
    px4_source = px4_source.resolve()
    actual_commit = subprocess.check_output(["git", "-C", str(px4_source), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != PX4_COMMIT:
        raise SitlError(f"PX4 source must be pinned to {PX4_TAG}: {PX4_COMMIT}")
    executable = px4_source / "build/px4_sitl_default/bin/px4"
    provenance_path = px4_source / "build/px4_sitl_default/flybrain_build.json"
    if not provenance_path.is_file():
        raise SitlError("Build provenance is missing; use integrations/px4/bootstrap_ubuntu.sh")
    provenance = json.loads(provenance_path.read_text())
    if (provenance.get("px4_commit") != PX4_COMMIT or provenance.get("binary_sha256") !=
            hashlib.sha256(executable.read_bytes()).hexdigest()):
        raise SitlError("SITL executable differs from its pinned build provenance")
    if subprocess.check_output(["git", "-C", str(px4_source), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        raise SitlError("Use an unmodified pinned PX4 source checkout")
    # Reject a competing local instance instead of connecting ambiguously.
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if (entry/"exe").resolve(strict=True).name == "px4":
                    raise SitlError("Another PX4 process is running; this runner owns its simulation")
            except OSError:
                pass
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 14540))
    output.mkdir(parents=True, exist_ok=True)
    scenario = static_sitl_scenario(seed)
    export_world(scenario, output / "world")
    token = os.urandom(24).hex()
    env = os.environ.copy()
    env.update({"HEADLESS": "1", "PX4_SYS_AUTOSTART": "4001", "PX4_SIM_MODEL": "gz_x500",
                "PX4_GZ_WORLD": "flybrain_inspection", "PX4_GZ_MODEL_POSE": "3,3,0,0,0,0",
                "PX4_SIM_SPEED_FACTOR": "1", "FLYBRAIN_SITL_TOKEN": token,
                "GZ_SIM_RESOURCE_PATH": str((output/"world").resolve())+os.pathsep+env.get("GZ_SIM_RESOURCE_PATH", "")})
    for key in ("PX4_GZ_STANDALONE", "PX4_GZ_MODEL_NAME"):
        env.pop(key, None)
    with (output/"px4.log").open("w") as stdout:
        process = subprocess.Popen([str(executable)], cwd=px4_source, env=env,
                                   stdout=stdout, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic()+5
            while True:
                identity = process_identity(process.pid)
                if identity["exe"] == str(executable.resolve()):
                    break
                if time.monotonic() >= deadline:
                    raise SitlError("PX4 process failed to launch")
                time.sleep(0.05)
            lease = ProcessLease(process.pid, identity["start_ticks"], identity["exe"], token)
            lease.verify()
            return run_mission(lease, scenario, output/"sitl_result.json")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--px4-source", type=Path)
    export = commands.add_parser("export-world")
    export.add_argument("--seed", type=int, default=500000)
    export.add_argument("--output", type=Path, default=Path("results/px4_world"))
    run = commands.add_parser("run")
    run.add_argument("--px4-source", type=Path, required=True)
    run.add_argument("--seed", type=int, default=500000)
    run.add_argument("--output", type=Path, default=Path("results/px4_sitl"))
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            status = environment_status(args.px4_source)
            print(json.dumps(status, indent=2))
            return 0 if status["ready"] else 2
        if args.command == "export-world":
            print(json.dumps(export_world(static_sitl_scenario(args.seed), args.output), indent=2))
            return 0
        result = run_local(args.px4_source, args.output, args.seed)
        print(json.dumps({k:v for k,v in result.items() if k not in ("trajectory",)}, indent=2))
        return 0 if result["mission_complete"] else 1
    except (SitlError, OSError, subprocess.SubprocessError) as exc:
        print(f"PX4 SITL unavailable/aborted: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
