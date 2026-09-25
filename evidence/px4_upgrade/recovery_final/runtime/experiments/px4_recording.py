"""Bounded video backpressure isolation for the live PX4 control loop."""
from __future__ import annotations

import queue
import threading
from copy import deepcopy


class AsyncRecorder:
    """Drop video samples when encoding falls behind; never delay flight polling.

    Only this thread calls the underlying recorder. Finish runs after landing
    and simulator shutdown. Captured frames are copied at admission so later
    renderer reuse cannot change their recorded pixels.
    """
    def __init__(self, recorder):
        self.recorder = recorder
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.dropped = 0
        self.closed = False
        self.final_result = None
        self.thread = threading.Thread(target=self._run, name='px4-recording', daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                try:
                    kind, payload = self.queue.get(timeout=.1)
                except queue.Empty:
                    if self.closed:
                        receipt = self.recorder.finish(dict(self.final_result,
                                                            recording_dropped_samples=self.dropped))
                        if not isinstance(receipt, dict) or receipt.get('error') or receipt.get('status') == 'error':
                            raise RuntimeError(f'Encoder finalization failed: {receipt}')
                        return
                    continue
                self.recorder.append(*payload)
        except BaseException as exc:
            self.error = exc
            try:
                self.recorder.finish(dict(passed=False, error=f'Recording failed: {exc}',
                                          recording_dropped_samples=self.dropped))
            except BaseException:
                pass

    def append(self, available_s, rgb, overview, telemetry):
        if self.error is not None:
            raise RuntimeError('Recording worker failed') from self.error
        if self.closed:
            raise RuntimeError('Recording already closed')
        try:
            self.queue.put_nowait(('frame', (available_s, rgb.copy(), overview.copy(), deepcopy(telemetry))))
            return True
        except queue.Full:
            self.dropped += 1
            return False

    def finish(self, result):
        if self.closed:
            return
        # Completion uses a separate flag, so a full frame queue cannot prevent
        # admission of shutdown. The worker drains already accepted samples.
        self.final_result = dict(result)
        self.closed = True
        self.thread.join(timeout=60)
        if self.thread.is_alive():
            # Kill only this recorder's encoders to unblock raw-pipe writes.
            # PX4 has already been stopped before recording finalization.
            for encoder in self.recorder.encoders.copy().values():
                if encoder.process.poll() is None:
                    encoder.process.kill()
            self.thread.join(timeout=5)
            raise RuntimeError('Recording worker did not finish')
        if self.error is not None:
            raise RuntimeError('Recording worker failed') from self.error
