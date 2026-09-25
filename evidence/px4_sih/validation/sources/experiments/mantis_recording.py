"""Bounded, causal recording for interactive Mantis simulation sessions.

Compact recordings retain H.264 video, JSONL telemetry and provenance. Research
arrays are an explicit opt-in. This module never removes another run. The byte
limit covers the logical sizes of every file it creates, including finalization;
filesystem allocation overhead is outside that limit. An exhausted recording
retains only complete MP4 fragments and marks the saved evidence as partial.
"""
from __future__ import annotations

from io import BytesIO
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess
import threading

import numpy as np

from experiments.mantis_report import ACTOR_CREDIT

_FINAL_RESERVE = 32 * 1024
_MAX_JSON = 24 * 1024
_MAX_CHUNK = 16 * 1024 * 1024


def _number(value, name, minimum, maximum):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not minimum <= value <= maximum):
        raise ValueError(f'{name} must be finite and between {minimum} and {maximum}')
    return value


def _json(value, limit=_MAX_JSON):
    if not isinstance(value, dict):
        raise ValueError('Metadata must be a JSON object')
    try:
        data = (json.dumps(value, allow_nan=False, sort_keys=True, separators=(',', ':'))+'\n').encode()
    except (TypeError, ValueError) as error:
        raise ValueError('Metadata must contain finite JSON values') from error
    if len(data) > limit:
        raise ValueError(f'Metadata exceeds {limit} bytes')
    return data


class _Budget:
    def __init__(self, limit):
        self.limit = limit
        self.used = 0
        self.exhausted = False
        self.lock = threading.Lock()

    def remaining(self):
        with self.lock:
            return max(0, self.limit-_FINAL_RESERVE-self.used)

    def write(self, stream, data, final=False):
        with self.lock:
            ceiling = self.limit if final else self.limit-_FINAL_RESERVE
            if (self.exhausted and not final) or self.used+len(data) > ceiling:
                self.exhausted = True
                return False
            written = stream.write(data)
            if written != len(data):
                raise OSError('Short recording write')
            self.used += written
            return True

    def discard(self, path):
        # Only used for this recorder's own header-only failed video.
        with self.lock:
            size = path.stat().st_size
            path.unlink()
            self.used -= size


class _CappedBuffer(BytesIO):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def write(self, data):
        if self.tell()+len(data) > self.limit:
            raise BufferError('Research frame exceeds remaining recording budget')
        return super().write(data)


class _Encoder:
    """Drain ffmpeg continuously, admitting complete H.264 MP4 fragments only."""
    def __init__(self, path, shape, fps, budget, executable):
        self.path, self.shape, self.budget = path, shape, budget
        self.frames = 0
        self.error = None
        self.closed = False
        height, width, _ = shape
        self.output = path.open('xb', buffering=0)
        command = [executable, '-nostdin', '-hide_banner', '-loglevel', 'error', '-xerror',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}',
                   '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-preset', 'ultrafast', '-tune', 'zerolatency', '-crf', '25',
                   '-bf', '0', '-g', str(max(1, round(fps))), '-threads', '1',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-pix_fmt', 'yuv420p',
                   '-movflags', '+empty_moov+default_base_moof+frag_every_frame',
                   '-flush_packets', '1', '-f', 'mp4', 'pipe:1']
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)
        except BaseException:
            self.output.close()
            self.budget.discard(path)
            raise
        self.thread = threading.Thread(target=self._drain, name=f'mantis-{path.stem}', daemon=True)
        self.thread.start()

    def _exact(self, size):
        result = bytearray()
        while len(result) < size:
            chunk = self.process.stdout.read(size-len(result))
            if not chunk:
                break
            result.extend(chunk)
        return bytes(result)

    def _drain(self):
        pending = b''
        try:
            while True:
                header = self._exact(8)
                if not header:
                    break
                if len(header) != 8:
                    raise RuntimeError('Incomplete MP4 box header')
                size, kind = struct.unpack('>I4s', header)
                if size == 1:
                    extension = self._exact(8)
                    if len(extension) != 8:
                        raise RuntimeError('Incomplete extended MP4 header')
                    size = struct.unpack('>Q', extension)[0]
                    header += extension
                if size < len(header) or size > _MAX_CHUNK:
                    raise RuntimeError('Invalid or oversized MP4 fragment')
                payload = self._exact(size-len(header))
                if len(payload) != size-len(header):
                    raise RuntimeError('Incomplete MP4 fragment')
                box = header+payload
                if kind in (b'ftyp', b'moov'):
                    self.budget.write(self.output, box)
                elif kind == b'moof':
                    if pending:
                        raise RuntimeError('MP4 fragment has no media payload')
                    pending = box
                elif kind == b'mdat':
                    if not pending:
                        raise RuntimeError('MP4 media has no fragment header')
                    if self.budget.write(self.output, pending+box):
                        self.frames += 1
                    pending = b''
                # mfra is optional seek metadata; omitting it preserves a playable
                # prefix and means finalization needs no unbounded trailer budget.
            if pending:
                raise RuntimeError('Incomplete final MP4 fragment')
        except BaseException as error:
            self.error = str(error)
            # Unblock a writer if parsing or filesystem output has failed.
            self.process.kill()
        finally:
            self.process.stdout.close()
            self.output.close()

    def append(self, frame):
        if self.error:
            raise RuntimeError(self.error)
        try:
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            raise RuntimeError('Video encoder stopped unexpectedly') from error

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.process.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        try:
            code = self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
            code = -1
            self.error = self.error or 'Video encoder finalization timed out'
        self.thread.join(timeout=5)
        stderr = self.process.stderr.read(4096).decode(errors='replace')
        self.process.stderr.close()
        if self.thread.is_alive():
            self.error = self.error or 'Video encoder output did not close'
        if code:
            self.error = self.error or stderr or f'Video encoder exited {code}'
        if self.frames == 0 and self.path.exists() and not self.thread.is_alive():
            self.budget.discard(self.path)


class SessionRecorder:
    """Create a new bounded session; existing nonempty folders are never changed.

    ``append`` takes the time at which its images/telemetry become available, not
    a future capture time. It returns False when recording stops for its budget.
    Images are held causally on the fixed FPS clock, with black before the first
    available image. ``finish({'duration_s': end_time, ...})`` can hold the final
    image up to a known simulation end time. A frame at exactly that end is not
    extrapolated into an extra video interval. No whole run is buffered in RAM.
    """
    def __init__(self, folder: Path, profile='compact', max_bytes=64*1024**2, fps=10,
                 provenance: dict | None = None):
        if profile not in ('compact', 'full-research'):
            raise ValueError('profile must be compact or full-research')
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 128*1024:
            raise ValueError('max_bytes must be an integer of at least 131072')
        _number(fps, 'fps', 1, 60)
        executable = shutil.which('ffmpeg')
        if not executable:
            raise RuntimeError('ffmpeg with libx264 is required for browser-playable recording')
        if provenance is not None and not isinstance(provenance, dict):
            raise ValueError('provenance must be a JSON object')
        provenance_data = _json(dict(schema='mantis-session-provenance-v1', profile=profile,
            fps=fps, max_bytes=max_bytes, video_clock='simulation time; hold latest available image',
            raw_arrays_retained=profile == 'full-research', source=provenance or {}))
        requested = Path(folder).expanduser().absolute()
        if '..' in requested.parts:
            raise ValueError('Recording folder cannot contain parent traversal')
        if any(part.is_symlink() for part in (requested, *requested.parents)):
            raise ValueError('Recording folder cannot contain symlinks')
        if requested.exists() and (not requested.is_dir() or any(requested.iterdir())):
            raise FileExistsError('Recording requires a new or empty folder; no files will be overwritten')
        requested.mkdir(parents=True, exist_ok=True)
        self.folder, self.profile, self.fps = requested, profile, fps
        self.budget = _Budget(max_bytes)
        self.executable = executable
        self.encoders = {}
        self.previous = None
        self.last_time = None
        self.next_frame = 0
        self.observations = 0
        self.raw_frames = 0
        self.finished = False
        self.receipt = None
        self.error = None
        self.telemetry = None
        try:
            self._file('provenance.json', provenance_data)
            self._file('CREDITS.txt', ACTOR_CREDIT.encode())
            self.telemetry = (requested/'telemetry.jsonl').open('xb', buffering=0)
            if profile == 'full-research':
                (requested/'raw').mkdir()
        except BaseException:
            if self.telemetry is not None:
                self.telemetry.close()
            raise

    def _file(self, name, data, final=False):
        path = self.folder/name
        with path.open('xb', buffering=0) as stream:
            written = self.budget.write(stream, data, final=final)
        if not written:
            self.budget.discard(path)
        return written

    @property
    def budget_exhausted(self):
        return self.budget.exhausted

    @property
    def bytes_written(self):
        return self.budget.used

    def _validate_image(self, value):
        if (not isinstance(value, np.ndarray) or value.dtype != np.uint8 or value.ndim != 3
                or value.shape[2] != 3 or min(value.shape[:2]) < 2 or value.nbytes > _MAX_CHUNK):
            raise ValueError('Images must be uint8 HWC RGB arrays, at least 2x2 and at most 16 MiB')
        return np.ascontiguousarray(value)

    def _emit(self, images):
        if self.budget.exhausted:
            return False
        for name, frame in images.items():
            self.encoders[name].append(frame)
        self.next_frame += 1
        return not self.budget.exhausted

    def append(self, sim_time_s, rgb, overview, telemetry, raw=None):
        if self.finished:
            raise RuntimeError('Recording is already finalized')
        if self.budget.exhausted:
            return False
        try:
            _number(sim_time_s, 'sim_time_s', 0, 3600)
            if self.last_time is not None and sim_time_s <= self.last_time:
                raise ValueError('Simulation times must strictly increase')
            images = dict(camera=self._validate_image(rgb), overview=self._validate_image(overview))
            row = _json(dict(sim_time_s=sim_time_s, sequence=self.observations, telemetry=telemetry))
            if not isinstance(telemetry, dict):
                raise ValueError('telemetry must be a JSON object')
            if self.previous is not None and any(images[key].shape != self.previous[key].shape for key in images):
                raise ValueError('Video image dimensions cannot change during a session')
            raw_bytes = None
            if self.profile == 'full-research':
                if raw is not None and not isinstance(raw, dict):
                    raise ValueError('raw must be a mapping of named numeric arrays')
                arrays = dict(rgb=images['camera'], overview=images['overview'])
                for key, value in (raw or {}).items():
                    if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', key) or key in arrays:
                        raise ValueError('Research array names must be safe and unique')
                    value = np.asarray(value)
                    if value.nbytes > _MAX_CHUNK:
                        raise ValueError('One research array may contain at most 16 MiB')
                    if value.dtype.kind not in 'biufc' or not np.isfinite(value).all():
                        raise ValueError('Research arrays must be finite numeric arrays')
                    arrays[key] = value
                if sum(value.nbytes for value in arrays.values()) > _MAX_CHUNK:
                    raise ValueError('One research capture may contain at most 16 MiB of arrays')
                raw_buffer = _CappedBuffer(min(_MAX_CHUNK, self.budget.remaining()))
                try:
                    np.savez_compressed(raw_buffer, **arrays)
                    raw_bytes = raw_buffer.getvalue()
                except BufferError:
                    self.budget.exhausted = True
                    return False
                finally:
                    raw_buffer.close()
            if not self.encoders:
                for name, frame in images.items():
                    self.encoders[name] = _Encoder(self.folder/f'{name}.mp4', frame.shape,
                                                   self.fps, self.budget, self.executable)
            # A new image at .25 must not appear in the .1 or .2 video samples.
            held = self.previous or {name: np.zeros_like(frame) for name, frame in images.items()}
            while self.next_frame/self.fps < sim_time_s-1e-9:
                if not self._emit(held):
                    return False
            if not self.budget.write(self.telemetry, row):
                return False
            if raw_bytes is not None:
                if not self._file(f'raw/{self.observations:06d}.npz', raw_bytes):
                    return False
                self.raw_frames += 1
            self.observations += 1
            self.last_time = float(sim_time_s)
            self.previous = {name: frame.copy() for name, frame in images.items()}
            return not self.budget.exhausted
        except BaseException as error:
            self.error = str(error)
            self.finish(dict(status='error', error=str(error)[:1000]))
            raise

    def finish(self, summary: dict | None = None):
        if self.finished:
            return self.receipt
        summary = {} if summary is None else summary
        validation_error = None
        end_time = self.last_time+1/self.fps if self.last_time is not None else 0.
        try:
            _json(summary)
            if 'duration_s' in summary:
                end_time = _number(summary['duration_s'], 'duration_s', 0, 3600)
                if self.last_time is not None and end_time < self.last_time-1e-9:
                    raise ValueError('duration_s cannot precede the last observation')
            if self.previous is not None:
                while self.next_frame/self.fps < end_time-1e-9:
                    if not self._emit(self.previous):
                        break
        except BaseException as error:
            validation_error = error
            self.error = str(error)
            summary = dict(status='error', error=str(error)[:1000])
        finally:
            for encoder in self.encoders.values():
                encoder.close()
                self.error = self.error or encoder.error
            if self.telemetry is not None:
                self.telemetry.close()
        self.error = str(self.error)[:4096] if self.error is not None else None
        status = ('error' if self.error or summary.get('status') == 'error' else 'budget-exhausted' if self.budget.exhausted
                  else 'interrupted' if summary.get('status') == 'interrupted' else 'completed')
        videos = {name: dict(path=f'{name}.mp4', frames=encoder.frames,
                            duration_s=encoder.frames/self.fps)
                  for name, encoder in self.encoders.items() if encoder.frames}
        receipt = dict(schema='mantis-session-v1', status=status, profile=self.profile,
                       max_bytes=self.budget.limit, bytes=0, fps=self.fps,
                       observations=self.observations, raw_frames=self.raw_frames,
                       last_observation_time_s=self.last_time, simulation_end_s=end_time,
                       videos=videos, paths=dict(telemetry='telemetry.jsonl', provenance='provenance.json',
                                                credits='CREDITS.txt', summary='summary.json'),
                       error=self.error, summary=summary,
                       complete=status == 'completed',
                       raw_capture_complete=self.profile == 'full-research' and status == 'completed')
        if self.profile == 'full-research':
            receipt['paths']['raw'] = 'raw'
        for _ in range(8):
            encoded = _json(receipt, _FINAL_RESERVE)
            total = self.budget.used+len(encoded)
            if total == receipt['bytes']:
                break
            receipt['bytes'] = total
        if not self._file('summary.json', encoded, final=True):
            raise RuntimeError('Reserved recording finalization budget was insufficient')
        self.receipt = receipt
        self.finished = True
        if validation_error is not None:
            raise validation_error
        return receipt

    def close(self):
        return self.finish(dict(status='interrupted'))

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            self.error = str(error)
        self.close()
        return False
