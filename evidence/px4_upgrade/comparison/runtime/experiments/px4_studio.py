"""File bridge between loopback Studio and its owned, simulation-only PX4 runner.

The runner owns flight authority and LAND. This bridge only carries observed
selection requests, bounded preview images, status and a cooperative stop flag.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import queue
import signal
import threading
import time

from experiments.mantis_studio import write_json
from experiments.mantis_studio_config import validate_config


def _warm_preview_encoder():
    # Import and initialize native codecs before the flight runner can start.
    # A first-use import/codec stall on the preview thread can hold the GIL and
    # delay the independently freshness-checked control loop.
    import cv2
    import numpy as np
    sample = np.zeros((8, 8, 3), dtype=np.uint8)
    okay, jpeg = cv2.imencode('.jpg', cv2.cvtColor(sample, cv2.COLOR_RGB2BGR),
                             [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not okay or jpeg.nbytes == 0:
        raise RuntimeError('Could not initialize Studio preview encoder')
    return cv2


class StudioBridge:
    def __init__(self, folder, config):
        self.folder = Path(folder).absolute()
        if any(p.is_symlink() for p in (self.folder, *self.folder.parents)):
            raise ValueError('Studio run path may not traverse symlinks')
        self.config = validate_config(config)
        if self.config['backend'] != 'px4_sih':
            raise ValueError('PX4 bridge requires the PX4 backend')
        self.recording_folder = self.folder/'capture'
        self.stop = threading.Event()
        self.queue = queue.Queue(maxsize=1)
        self.closed = False
        self.error = None
        self.dropped = 0
        self.revision = -1
        self.last_frame = None
        self.frame_sequences = []
        self.last_selection = dict(people=[], track_id=None, reason='selection_required')
        self.selection_history = []
        self.latest = dict(phase='loading', message='Starting owned PX4/SIH simulation',
                          sim_s=0., config=self.config, backend='px4_sih', simulation_only=True)
        self._cv2 = _warm_preview_encoder()
        self.thread = threading.Thread(target=self._drain, name='px4-studio-preview', daemon=True)
        self.thread.start()

    def request_stop(self, *_):
        self.stop.set()

    def control(self):
        if self.stop.is_set():
            return dict(revision=max(0, self.revision+1), operation='stop', selection=None)
        path = self.folder/'control.json'
        if path.is_symlink() or path.stat().st_size > 8192:
            raise ValueError('Invalid Studio control file')
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or set(value)-{'revision', 'operation', 'settings', 'selection'}:
            raise ValueError('Invalid Studio control fields')
        revision = value.get('revision')
        if type(revision) is not int or revision < max(0, self.revision):
            raise ValueError('Studio control revision regressed')
        if value.get('operation') not in ('run', 'stop'):
            raise ValueError('PX4 dynamics cannot pause; request LAND instead')
        validate_config(value.get('settings', {}), base=self.config, live=True)
        selection = value.get('selection')
        if selection is not None:
            if not isinstance(selection, dict) or set(selection)-{'revision', 'clear', 'track_id', 'sequence'}:
                raise ValueError('Invalid observed-person selection')
            if type(selection.get('revision')) is not int or not 0 <= selection['revision'] <= revision:
                raise ValueError('Invalid selection revision')
            if selection.get('clear') is not True:
                for key, minimum in [('track_id', 1), ('sequence', 0)]:
                    if type(selection.get(key)) is not int or not minimum <= selection[key] <= 2**31-1:
                        raise ValueError('Selection requires an observed track and frame sequence')
        self.revision = revision
        if value['operation'] == 'stop':
            self.stop.set()
        return dict(revision=revision, operation=value['operation'], selection=selection)

    def publish(self, snapshot, rgb=None, overview=None):
        if self.error is not None:
            raise RuntimeError('PX4 Studio preview failed') from self.error
        if self.closed:
            raise RuntimeError('PX4 Studio preview is closed')
        if not isinstance(snapshot, dict) or (rgb is None) != (overview is None):
            raise ValueError('Studio preview requires paired camera/overview images')
        state = deepcopy(snapshot)
        if rgb is not None:
            frame = state.get('frame', {})
            selection = state.get('frame_selection', state.get('selection', {}))
            sequence = frame.get('sequence')
            if type(sequence) is not int or not 0 <= sequence <= 2**31-1 or selection.get('sequence') != sequence:
                raise ValueError('Preview images and observed selection must share their frame sequence')
            for image in (rgb, overview):
                if image.ndim != 3 or image.shape[2] != 3 or str(image.dtype) != 'uint8' or image.nbytes > 4*1024**2:
                    raise ValueError('Preview image must be bounded RGB uint8')
            rgb, overview = rgb.copy(), overview.copy()
        try:
            self.queue.put_nowait((state, rgb, overview))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def _write_images(self, state, rgb, overview):
        cv2 = self._cv2
        sequence = state['frame']['sequence']
        capture = state['frame'].get('capture_time_s')
        if (not isinstance(capture, (int, float)) or isinstance(capture, bool)
                or not math.isfinite(capture)):
            raise ValueError('Published preview needs its original finite capture time')
        previous = next((item for item in self.selection_history if item['sequence'] == sequence), None)
        if previous is not None and previous['capture_time_s'] != capture:
            raise ValueError('Published preview timestamp changed')
        for name, image in (('camera', rgb), ('overview', overview)):
            okay, jpeg = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                     [cv2.IMWRITE_JPEG_QUALITY, 82])
            if not okay or jpeg.nbytes > 1024**2:
                raise RuntimeError('Could not encode bounded Studio preview')
            path = self.folder/f'{name}-{sequence}.jpg'
            temporary = path.with_suffix('.jpg.tmp')
            temporary.write_bytes(jpeg.tobytes())
            temporary.replace(path)
        self.last_frame = state['frame']
        self.last_selection = state.get('frame_selection', state.get('selection', {}))
        if previous is None and 0 <= time.monotonic()-capture <= 3.:
            self.selection_history.append(dict(sequence=sequence, capture_time_s=capture,
                selectable_tracks=[p['track_id'] for p in self.last_selection.get('people', [])
                                   if p.get('selectable') is True]))
        self.selection_history = self.selection_history[-8:]
        if sequence not in self.frame_sequences:
            self.frame_sequences = (self.frame_sequences+[sequence])[-2:]

    def _drain(self):
        try:
            while True:
                try:
                    state, rgb, overview = self.queue.get(timeout=.1)
                except queue.Empty:
                    if self.closed:
                        return
                    continue
                if rgb is not None:
                    self._write_images(state, rgb, overview)
                self.selection_history = [item for item in self.selection_history
                    if 0 <= time.monotonic()-item['capture_time_s'] <= 3.]
                # Current hold/loss/clear status is independent of the latest
                # admitted image. Cached boxes never overwrite live authority.
                state.update(config=self.config, backend='px4_sih', simulation_only=True,
                             frame=self.last_frame, frame_selection=self.last_selection,
                             selection_history=list(self.selection_history),
                             preview_dropped_samples=self.dropped)
                write_json(self.folder/'state.json', state)
                self.latest = state
                if self.last_frame:
                    for name in ('camera', 'overview'):
                        for old in self.folder.glob(name+'-*.jpg'):
                            index = old.stem.rsplit('-', 1)[-1]
                            if index.isdigit() and int(index) not in self.frame_sequences:
                                old.unlink()
        except BaseException as exc:
            self.error = exc
            self.stop.set()

    def finish(self, result):
        self.closed = True
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Studio preview worker did not finish')
        receipt = None
        path = self.recording_folder/'summary.json'
        if path.is_file() and not path.is_symlink():
            receipt = json.loads(path.read_text())
        error = result.get('error') or (str(self.error) if self.error is not None else None)
        if receipt and receipt.get('error'):
            error = error or receipt['error']
        phase = ('error' if error else 'budget-exhausted' if receipt and receipt.get('status') == 'budget-exhausted'
                 else 'interrupted' if self.stop.is_set() else 'completed')
        state = dict(self.latest, phase=phase, config=self.config, backend='px4_sih', simulation_only=True,
                     failure=dict(reason=error) if error else None, receipt=receipt, result=result,
                     message=error or ('PX4 simulation stopped' if self.stop.is_set()
                                       else 'PX4 simulation finished'))
        write_json(self.folder/'state.json', state)
        if error:
            raise RuntimeError(error)
        return state


def run_session(folder, runner=None):
    folder = Path(folder).absolute()
    if any(p.is_symlink() for p in (folder, *folder.parents)) or (folder/'config.json').is_symlink():
        raise ValueError('Invalid Studio run path')
    config = validate_config(json.loads((folder/'config.json').read_text()))
    bridge = StudioBridge(folder, config)
    handlers = {sig: signal.signal(sig, bridge.request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    result = dict(error='PX4 runner did not complete')
    try:
        if runner is None:
            from experiments.px4_follow import run
            runner = run
        result = runner(folder/'flight', selected_track=None,
                        low_noise_sensors=config['low_noise_sensors'],
                        estimator_profile='stock' if config['low_noise_sensors'] else 'sih_velocity_settled',
                        studio=bridge)
        return result
    except BaseException as exc:
        result = dict(error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            bridge.finish(result)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, required=True)
    args = parser.parse_args()
    run_session(args.folder)


if __name__ == '__main__':
    main()
