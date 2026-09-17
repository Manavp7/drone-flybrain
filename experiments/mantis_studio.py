"""Loopback-only Mantis Studio: one bounded simulation child at a time."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from urllib.parse import urlsplit, parse_qs, unquote

from experiments.flight_contracts import ROOT
from experiments.mantis_studio_config import DEFAULTS, LIVE_KEYS, validate_config

RUN_ID = re.compile(r'^run-[0-9]{8}T[0-9]{6}-[a-f0-9]{8}$')
TERMINAL = {'completed', 'interrupted', 'error', 'budget-exhausted'}
OVERHEAD_BYTES = 4*1024**2


def write_json(path, payload):
    data = json.dumps(payload, allow_nan=False, separators=(',', ':')).encode()
    if len(data) > 128*1024:
        raise ValueError('Metadata too large')
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_bytes(data)
    temporary.replace(path)


def directory_bytes(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


class StudioManager:
    def __init__(self, root, max_total_mb=256):
        root = Path(root).absolute()
        if any(p.is_symlink() for p in [root, *root.parents]):
            raise ValueError('Runs folder may not traverse symlinks')
        root.mkdir(parents=True, exist_ok=True)
        self.root = root.resolve()
        if isinstance(max_total_mb, bool) or not isinstance(max_total_mb, int) or not 32 <= max_total_mb <= 2048:
            raise ValueError('Total budget must be between 32 and 2048 MiB')
        self.max_bytes = max_total_mb*1024**2
        self.lock = threading.RLock()
        self.process = None
        self.active = None
        self.control = None
        self.token = secrets.token_urlsafe(32)

    def run_folder(self, identity):
        if not isinstance(identity, str) or not RUN_ID.fullmatch(identity):
            raise ValueError('Invalid run identifier')
        folder = self.root/identity
        if folder.is_symlink() or folder.resolve().parent != self.root:
            raise ValueError('Invalid run path')
        return folder

    def running(self):
        return self.process is not None and self.process.poll() is None

    def _drain_log(self, process, folder):
        recent = b''
        while True:
            chunk = process.stdout.read(1024)
            if not chunk:
                break
            recent = (recent+chunk)[-32768:]
            (folder/'runtime.log').write_bytes(recent)

    def start(self, value):
        config = validate_config(value)
        with self.lock:
            if self.running():
                raise RuntimeError('A simulation is already active')
            used = directory_bytes(self.root)
            if used+config['max_recording_mb']*1024**2+OVERHEAD_BYTES > self.max_bytes:
                raise RuntimeError('Storage budget cannot reserve this run. Choose a smaller run budget or move a saved run; no runs are deleted automatically.')
            identity = datetime.now(timezone.utc).strftime('run-%Y%m%dT%H%M%S-')+secrets.token_hex(4)
            folder = self.run_folder(identity)
            folder.mkdir()
            write_json(folder/'config.json', config)
            self.control = dict(revision=0, operation='run', settings={}, selection=None)
            write_json(folder/'control.json', self.control)
            write_json(folder/'state.json', dict(phase='loading', config=config, sim_s=0., message='Starting simulation'))
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
            self.process = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'experiments.mantis_session',
                '--folder', str(folder)], cwd=ROOT, env=environment,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            self.active = identity
            threading.Thread(target=self._drain_log, args=(self.process, folder), daemon=True).start()
            return self.snapshot()

    def command(self, value):
        if not isinstance(value, dict) or set(value)-{'run_id', 'operation', 'settings', 'selection'}:
            raise ValueError('Invalid control fields')
        with self.lock:
            if not self.running():
                raise RuntimeError('No active simulation')
            if value.get('run_id') != self.active:
                raise ValueError('Run changed; refresh before sending controls')
            if self.control.get('operation') == 'stop':
                if set(value) <= {'run_id', 'operation'} and value.get('operation') == 'stop':
                    return dict(accepted=True, revision=self.control['revision'])
                raise RuntimeError('Simulation is stopping; launch a new run after it finishes')
            new = json.loads(json.dumps(self.control))
            if 'operation' in value:
                if value['operation'] not in ('run', 'pause', 'stop'):
                    raise ValueError('Invalid operation')
                new['operation'] = value['operation']
            if 'settings' in value:
                current = validate_config(json.loads((self.run_folder(self.active)/'config.json').read_text()))
                current.update(new['settings'])
                updated = validate_config(value['settings'], base=current, live=True)
                new['settings'] = {k: updated[k] for k in LIVE_KEYS}
            if 'selection' in value:
                selection = value['selection']
                if not isinstance(selection, dict) or set(selection)-{'track_id', 'sequence', 'clear'}:
                    raise ValueError('Invalid selection')
                if selection.get('clear') is True:
                    selection = dict(clear=True)
                else:
                    for key, low in [('track_id', 1), ('sequence', 0)]:
                        if type(selection.get(key)) is not int or not low <= selection[key] <= 2**31-1:
                            raise ValueError('Selection needs an integer track ID and frame sequence')
                    # The child checks this again against its current camera.
                    state = self._state(self.active)
                    if selection['sequence'] != state.get('selection', {}).get('sequence'):
                        raise ValueError('Camera frame changed; select again from the latest image')
                    selection = {k: selection[k] for k in ('track_id', 'sequence')}
                new['selection'] = dict(selection, revision=new['revision']+1)
            new['revision'] += 1
            write_json(self.run_folder(self.active)/'control.json', new)
            self.control = new
            return dict(accepted=True, revision=new['revision'])

    def _state(self, identity):
        folder = self.run_folder(identity)
        path = folder/'state.json'
        if path.is_symlink():
            raise ValueError('Invalid state path')
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            return dict(phase='error', message='Run status unavailable')
        if identity == self.active and self.process is not None and not self.running() and state.get('phase') not in TERMINAL:
            state.update(phase='error', message='Simulation exited before finalizing', exit_code=self.process.returncode)
        return state

    def snapshot(self):
        with self.lock:
            runs = []
            for folder in sorted(self.root.iterdir(), reverse=True):
                if RUN_ID.fullmatch(folder.name) and folder.is_dir() and not folder.is_symlink():
                    state = self._state(folder.name)
                    runs.append(dict(id=folder.name, phase=state.get('phase'), sim_s=state.get('sim_s'),
                                     config=state.get('config'), receipt=state.get('receipt')))
                if len(runs) >= 12:
                    break
            return dict(active_id=self.active, running=self.running(),
                state=self._state(self.active) if self.active else None, runs=runs,
                defaults=DEFAULTS, storage=dict(bytes=directory_bytes(self.root), max_bytes=self.max_bytes))

    def close(self):
        with self.lock:
            if self.running():
                self.command(dict(run_id=self.active, operation='stop'))
                try:
                    self.process.wait(timeout=12)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=8)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=4)


def make_handler(manager):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'MantisStudio/3'

        def log_message(self, *args):
            pass

        def _valid_host(self):
            return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}'}

        def _headers(self, status, kind, length):
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(length))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'")

        def _json(self, status, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self._headers(status, 'application/json', len(body))
            self.end_headers()
            if self.command != 'HEAD':
                self.wfile.write(body)

        def do_HEAD(self):
            self.do_GET()

        def _range_error(self, size):
            self._headers(416, 'application/octet-stream', 0)
            self.send_header('Content-Range', f'bytes */{size}')
            self.end_headers()

        def do_POST(self):
            try:
                if not self._valid_host():
                    return self._json(403, dict(error='Loopback Host required'))
                origin = self.headers.get('Origin')
                allowed = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
                if origin is not None and origin not in allowed:
                    return self._json(403, dict(error='Same-origin requests required'))
                if not hmac.compare_digest(self.headers.get('X-Mantis-Token', ''), manager.token):
                    return self._json(403, dict(error='Studio token required'))
                if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                    return self._json(415, dict(error='JSON required'))
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 8192:
                    return self._json(413, dict(error='Request body too large or empty'))
                value = json.loads(self.rfile.read(length))
                if self.path == '/api/start':
                    result = manager.start(value)
                elif self.path == '/api/control':
                    result = manager.command(value)
                else:
                    return self._json(404, dict(error='Unknown endpoint'))
                self._json(200, result)
            except (ValueError, TypeError, KeyError) as exc:
                self._json(400, dict(error=str(exc)))
            except RuntimeError as exc:
                self._json(409, dict(error=str(exc)))

        def do_GET(self):
            try:
                if not self._valid_host():
                    return self._json(403, dict(error='Loopback Host required'))
                parsed = urlsplit(self.path)
                if parsed.path == '/':
                    body = (ROOT/'assets/mantis_studio.html').read_text().replace('__MANTIS_TOKEN__', manager.token).encode()
                    self._headers(200, 'text/html; charset=utf-8', len(body))
                    self.end_headers()
                    if self.command != 'HEAD':
                        self.wfile.write(body)
                    return
                if parsed.path == '/api/state':
                    return self._json(200, manager.snapshot())
                if parsed.path == '/api/frame':
                    query = parse_qs(parsed.query)
                    identity = query.get('id', [manager.active])[0]
                    view = query.get('view', ['camera'])[0]
                    if view not in ('camera', 'overview'):
                        raise ValueError('Unknown camera')
                    sequence = query.get('sequence', [str((manager._state(identity).get('frame') or {}).get('sequence', ''))])[0]
                    if not sequence.isdigit() or len(sequence) > 10:
                        raise ValueError('A valid frame sequence is required')
                    path = manager.run_folder(identity)/(view+'-'+sequence+'.jpg')
                elif parsed.path.startswith('/media/'):
                    parts = unquote(parsed.path).split('/')
                    if len(parts) != 4 or parts[3] not in {'camera.mp4', 'overview.mp4', 'telemetry.jsonl', 'summary.json', 'provenance.json', 'CREDITS.txt'}:
                        raise ValueError('Unknown recording artifact')
                    folder = manager.run_folder(parts[2])
                    path = folder/'capture'/parts[3]
                    if path.parent.is_symlink():
                        raise ValueError('Invalid recording path')
                else:
                    return self._json(404, dict(error='Not found'))
                if path.is_symlink() or not path.is_file():
                    return self._json(404, dict(error='Artifact not ready'))
                size = path.stat().st_size
                first, last, status = 0, size-1, 200
                requested = self.headers.get('Range')
                if requested:
                    match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
                    if not match or not any(match.groups()) or not size:
                        return self._range_error(size)
                    if match[1]:
                        first = int(match[1])
                        last = min(size-1, int(match[2]) if match[2] else size-1)
                    else:
                        suffix = int(match[2])
                        if suffix == 0:
                            return self._range_error(size)
                        first = max(0, size-suffix)
                    if first >= size or first > last:
                        return self._range_error(size)
                    status = 206
                kind = mimetypes.guess_type(str(path))[0] or 'application/octet-stream'
                self._headers(status, kind, max(0, last-first+1))
                self.send_header('Accept-Ranges', 'bytes')
                if status == 206:
                    self.send_header('Content-Range', f'bytes {first}-{last}/{size}')
                self.end_headers()
                if self.command == 'HEAD':
                    return
                with path.open('rb') as file:
                    file.seek(first)
                    remaining = last-first+1
                    while remaining > 0:
                        chunk = file.read(min(65536, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (ValueError, TypeError) as exc:
                self._json(400, dict(error=str(exc)))
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--port', type=int, default=8875)
    parser.add_argument('--runs', type=Path, default=ROOT/'results/studio')
    parser.add_argument('--max-total-mb', type=int, default=256)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    manager = StudioManager(args.runs.resolve(), args.max_total_mb)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(manager))
    print(f'Mantis Studio: http://127.0.0.1:{args.port} — simulation only', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        manager.close()


if __name__ == '__main__':
    main()
