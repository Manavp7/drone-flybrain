"""Owned, loopback-only PX4 SIH transport. No attach, serial or remote mode.

The existing Gazebo runner and its Linux identity checks are unchanged. Only
its bounded parameter/command protocol helpers are inherited here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .px4_sitl import MavlinkLink, PX4_COMMIT, SitlError, Telemetry, decode_parameter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / '.cache/PX4-Autopilot-v1.16.2'
BUILD = SOURCE / 'build/px4_sitl_mantis'
ALTITUDE = 1.1
# XY velocity, Z position, yaw angle. Ignore XY position, Z velocity,
# accelerations and yaw rate; MAV_FRAME_LOCAL_NED, never BODY_NED.
FOLLOW_MASK = 1 | 2 | 32 | 64 | 128 | 256 | 2048
POSITION_YAW_MASK = 8 | 16 | 32 | 64 | 128 | 256 | 2048


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha(path):
    digest = hashlib.sha256()
    for file in sorted(path.rglob('*')):
        if file.is_file():
            digest.update(file.relative_to(path).as_posix().encode()+b'\0')
            digest.update(file.read_bytes())
            digest.update(b'\0')
    return digest.hexdigest()


class SourceClock:
    """Conservative acquisition lower bounds, including socket queue delays.

An echoed request proves remote timestamp R was sampled after host send S.
For a later source sample T>R, S is a lower bound on acquisition. No 1:1
clock-rate assumption, extrapolation, or arrival-time freshness reset.
"""
    def __init__(self):
        self.pending = {}
        self.receipts = deque(maxlen=100)
        self.last_remote = 0
        self.last_send = -math.inf
        self.reset = False

    def request(self, now):
        nonce = int(now * 1e9)
        self.pending = {n: t for n, t in self.pending.items() if now-t < .2}
        self.pending[nonce] = now
        self.last_send = now
        return nonce

    def receive(self, nonce, remote_ns, now):
        sent = self.pending.pop(nonce, None)
        if sent is None or not 0 <= now-sent <= .05 or remote_ns <= 0:
            return False
        remote = remote_ns / 1e9
        if remote < self.last_remote:
            self.reset = True
            return False
        if remote == self.last_remote:
            return False
        self.last_remote = remote
        self.receipts.append((remote, sent))
        return True

    def capture_bound(self, source_s, now, max_age=.4):
        if self.reset or not math.isfinite(source_s) or source_s <= 0:
            raise SitlError('source_clock_unavailable_or_reset')
        for remote, sent in reversed(self.receipts):
            if remote < source_s:
                if not 0 <= now-sent <= max_age:
                    break
                return sent
        raise SitlError('no_fresh_preceding_source_clock_receipt')


ESTIMATOR_PROFILES = {
    'stock': {},
    # SIH generates ~1Pa pressure noise (~.087m at its start altitude), without
    # rotor ground effect. GPS altitude is substantially noisier (.5m RMS).
    'baro': {'EKF2_HGT_REF': 0, 'EKF2_GPS_CTRL': 5,
             'EKF2_BARO_NOISE': .10, 'EKF2_GND_EFF_DZ': 0},
    'baro_settled': {'EKF2_HGT_REF': 0, 'EKF2_GPS_CTRL': 5,
             'EKF2_BARO_NOISE': .10, 'EKF2_GND_EFF_DZ': 0,
             'EKF2_REQ_GPS_H': 10},
    'sih_covariance': {'EKF2_HGT_REF': 0, 'EKF2_GPS_CTRL': 5,
             'EKF2_BARO_NOISE': .10, 'EKF2_GND_EFF_DZ': 0,
             'EKF2_ACC_NOISE': 1.0, 'EKF2_GYR_NOISE': .10},
    'sih_calibrated': {'EKF2_HGT_REF': 0, 'EKF2_GPS_CTRL': 5,
             'EKF2_BARO_NOISE': .10, 'EKF2_GND_EFF_DZ': 0,
             'EKF2_ACC_NOISE': 1.0, 'EKF2_GYR_NOISE': .10,
             'EKF2_GPS_P_NOISE': .2, 'EKF2_GPS_V_NOISE': .158},
}
ESTIMATOR_PROFILES['sih_calibrated_settled'] = dict(ESTIMATOR_PROFILES['sih_calibrated'],
    EKF2_REQ_GPS_H=10)
ESTIMATOR_PROFILES['sih_velocity_settled'] = dict(ESTIMATOR_PROFILES['sih_calibrated_settled'],
    EKF2_GPS_V_NOISE=.11)


def launch_environment(inherited, low_noise_sensors=False, estimator_profile='stock'):
    if estimator_profile not in ESTIMATOR_PROFILES:
        raise ValueError('Unknown estimator profile')
    env = {k: v for k, v in inherited.items()
           if not k.startswith(('PX4_', 'MANTIS_SIH_'))}
    env.update(PX4_SYS_AUTOSTART='10040', PX4_SIM_MODEL='sihsim_quadx',
               PX4_SIMULATOR='sihsim', PX4_SIM_SPEED_FACTOR='1', PX4_PARAM_SDLOG_MODE='-1')
    if low_noise_sensors:
        env['MANTIS_SIH_LOW_NOISE_SENSORS'] = '1'
    if estimator_profile in ('sih_calibrated', 'sih_calibrated_settled'):
        env['MANTIS_SIH_CALIBRATED_GPS'] = '1'
    if estimator_profile == 'sih_velocity_settled':
        env['MANTIS_SIH_CALIBRATED_GPS'] = '2'
    for name, value in ESTIMATOR_PROFILES[estimator_profile].items():
        env['PX4_PARAM_'+name] = str(value)
    return env


def validate_build_receipt(receipt):
    if (receipt['px4_commit'] != PX4_COMMIT
            or receipt.get('build_recipe_sha256') != sha(ROOT / 'scripts/build_px4_sih.py')
            or receipt['binary_sha256'] != sha(BUILD / 'bin/px4')
            or receipt['overlay_sha256'] != sha(ROOT / 'integrations/px4/sih.px4board')
            or receipt['startup_tree_sha256'] != tree_sha(BUILD / 'etc')):
        raise SitlError('PX4 build receipt does not match the pinned executable/profile/recipe')


class OwnedSih:
    """Lifetime OS lock, private working directory and retained child handle."""
    def __init__(self, folder, low_noise_sensors=False, estimator_profile='stock'):
        self.process = self.log = self.lock = None
        self.connections = []
        self.receipt = json.loads((BUILD / 'mantis-build.json').read_text())
        validate_build_receipt(self.receipt)
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.lock = (ROOT / '.cache/mantis-px4-sih.lock').open('a')
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # Existing services cause refusal; never kill or attach to them.
            for port in (14540, 14580, 19410, 19450, 18570, 14280, 13030):
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
                    probe.bind(('127.0.0.1', port))
            for local, remote in ((14540, 14580), (19410, 19450)):
                self.connections.append(loopback_connection(local, remote))
            self.log = (self.folder / 'px4.log').open('xb')
            env = launch_environment(os.environ, low_noise_sensors, estimator_profile)
            self.process = subprocess.Popen([str(BUILD / 'bin/px4'), '-d',
                str(BUILD / 'etc')], cwd=self.folder, env=env, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        except BaseException:
            self.close()
            raise

    def verify(self):
        if self.process is None or self.process.poll() is not None:
            raise SitlError('owned_PX4_child_exited')

    def verify_ports(self):
        self.verify()
        listing = subprocess.check_output(['/usr/sbin/lsof', '-nP', '-a', '-p',
            str(self.process.pid), '-iUDP', '-Fn'], text=True)
        names = listing.splitlines()
        for port in (14580, 19450):
            if not any(line.startswith('n') and line.rsplit(':', 1)[-1] == str(port)
                       for line in names):
                raise SitlError('PX4 child does not own the expected UDP ports')

    def close(self):
        if self.process is not None and self.process.poll() is None:
            # Stops our simulation process only. Never an airborne disarm.
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        if self.log is not None:
            self.log.close()
        for connection in self.connections:
            connection.close()
        if self.lock is not None:
            self.lock.close()


def loopback_connection(local_port, remote_port):
    from pymavlink import mavutil
    connection = mavutil.mavlink_connection(f'udpin:127.0.0.1:{local_port}',
        source_system=245, source_component=191, dialect='common')
    connection.port.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(('127.0.0.1', local_port))
        except OSError:
            pass
        else:
            connection.close()
            raise SitlError('Receiving UDP port is not exclusive')
    # Connected UDP filters incoming peers in the kernel. pymavlink's udpin
    # client discovery must never add another destination, even on loopback.
    connection.port.connect(('127.0.0.1', remote_port))
    connection.write = lambda data: connection.port.send(data)
    return connection


class SihLink(MavlinkLink):
    """Separate SIH initialization, identity, polling and setpoint semantics."""
    def __init__(self, lease):
        lease.verify()
        self.lease = lease
        self.connection, self.truth_connection = lease.connections
        deadline = time.monotonic()+30
        while True:
            try:
                lease.verify_ports()
                break
            except (subprocess.CalledProcessError, SitlError):
                lease.verify()
                if time.monotonic() >= deadline:
                    raise SitlError('Owned PX4 UDP startup timed out')
                time.sleep(.1)
        original_write = self.connection.write
        def guarded_write(data):
            lease.verify()
            return original_write(data)
        self.connection.write = guarded_write
        self.telemetry = Telemetry()
        self.params, self.acks = {}, {}
        self.last_tx_heartbeat = -math.inf
        self.identified = self.arm_authorized = False
        self.clock = SourceClock()
        self.attitude = self.truth = None
        self.attitude_source = self.truth_source = 0.
        self.last_status = deque(maxlen=30)

    def close(self):
        self.connection.close()
        self.truth_connection.close()

    def poll(self):
        self.lease.verify()
        now = time.monotonic()
        if now-self.last_tx_heartbeat >= .5:
            self.connection.mav.heartbeat_send(6, 8, 0, 0, 4)
            self.last_tx_heartbeat = now
        if now-self.clock.last_send >= .02:
            self.connection.mav.timesync_send(0, self.clock.request(now))
        for connection in (self.connection, self.truth_connection):
            for _ in range(200):
                msg = connection.recv_match(blocking=False)
                if msg is None:
                    break
                if msg.get_srcSystem() != 1 or msg.get_srcComponent() != 1:
                    continue
                kind = msg.get_type()
                if connection is self.truth_connection:
                    if kind == 'HIL_STATE_QUATERNION':
                        source = msg.time_usec / 1e6
                        if source < self.truth_source:
                            self.clock.reset = True
                        elif source > self.truth_source:
                            self.truth, self.truth_source = msg, source
                    continue
                if kind == 'HEARTBEAT':
                    if msg.autopilot != 12 or msg.type != 2:
                        raise SitlError('Expected PX4 quadrotor heartbeat')
                    self.identified = True
                    self.telemetry.heartbeat_at = now
                    self.telemetry.armed = bool(msg.base_mode & 128)
                    self.telemetry.offboard = ((msg.custom_mode >> 16) & 255) == 6
                elif kind == 'TIMESYNC' and msg.tc1 > 0:
                    self.clock.receive(msg.ts1, msg.tc1, time.monotonic())
                elif kind == 'LOCAL_POSITION_NED':
                    self.telemetry.ingest_position(msg, now)
                elif kind == 'ATTITUDE_QUATERNION':
                    source = msg.time_boot_ms / 1000
                    if source < self.attitude_source:
                        self.clock.reset = True
                    elif source > self.attitude_source:
                        self.attitude, self.attitude_source = msg, source
                elif kind == 'ESTIMATOR_STATUS':
                    self.telemetry.estimator_flags, self.telemetry.estimator_at = int(msg.flags), now
                elif kind == 'SYS_STATUS':
                    self.telemetry.battery_fraction = (msg.battery_remaining / 100
                        if msg.battery_remaining >= 0 else math.nan)
                    self.telemetry.battery_at = now
                elif kind == 'EXTENDED_SYS_STATE':
                    self.telemetry.landed = msg.landed_state == 1
                    self.telemetry.landed_at = now
                elif kind == 'PARAM_VALUE':
                    key = msg.param_id.decode() if isinstance(msg.param_id, bytes) else msg.param_id
                    self.params[key.rstrip('\0')] = (decode_parameter(msg.param_value, msg.param_type), now)
                elif kind == 'COMMAND_ACK':
                    self.acks[int(msg.command)] = (int(msg.result), now)
                elif kind == 'STATUSTEXT':
                    self.last_status.append(str(msg.text))

    def verify_simulator(self):
        self.wait_for(lambda: self.identified, 30)
        self.lease.verify_ports()
        if self.telemetry.armed:
            raise SitlError('SIH must start disarmed')
        for name, expected in [('SYS_AUTOSTART', 10040), ('SIH_VEHICLE_TYPE', 0),
                               ('SENS_EN_GPSSIM', 1), ('SENS_EN_BAROSIM', 1),
                               ('SENS_EN_MAGSIM', 1)]:
            if self.get_parameter(name) != expected:
                raise SitlError(f'Unexpected SIH parameter {name}')
        self.arm_authorized = True

    def health_error(self, now):
        error = self.telemetry.health_error(now)
        if error:
            return error
        if self.attitude is None or self.telemetry.boot_ms is None:
            return 'attitude_unavailable'
        try:
            self.clock.capture_bound(self.attitude_source, now, .2)
            self.clock.capture_bound(self.telemetry.boot_ms/1000, now, .2)
        except SitlError as exc:
            return str(exc)
        return ''

    def setpoint(self, vector_ned, position=False, yaw_ned=0., altitude_ned=None):
        self.lease.verify()
        if not self.arm_authorized:
            raise SitlError('Setpoints require verified owned simulation')
        values = tuple(vector_ned) + (yaw_ned,)
        if len(values) != 4 or not all(math.isfinite(v) for v in values):
            raise SitlError('Nonfinite NED/yaw setpoint')
        if position:
            p, v, mask = vector_ned, (0, 0, 0), POSITION_YAW_MASK
        else:
            if (altitude_ned is None or not math.isfinite(altitude_ned)
                    or math.hypot(*vector_ned[:2]) > .45001 or vector_ned[2] != 0):
                raise SitlError('Unsupported SIH follow velocity/altitude')
            p, v, mask = (0, 0, altitude_ned), vector_ned, FOLLOW_MASK
        self.connection.mav.set_position_target_local_ned_send(
            int(time.monotonic()*1000) & 0xFFFFFFFF, 1, 1, 1,
            mask, *p, *v, 0, 0, 0, yaw_ned, 0)
