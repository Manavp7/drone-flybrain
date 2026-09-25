"""Fault-injected ownership checks; no native autopilot or model is launched."""
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments import px4_follow


class CleanupTests(unittest.TestCase):
    def provenance_root(self, directory):
        root = Path(directory).resolve()
        for name in ('scripts/build_px4_sih.py', 'integrations/px4/sih.px4board',
                     'models/flyvis_0000_000.manifest.json'):
            file = root/name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('test provenance fixture\n')
        return root

    def test_scoring_failure_still_closes_the_owned_simulator(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.provenance_root(directory)
            owned = Mock()
            with patch.object(px4_follow, 'ROOT', root), \
                    patch.object(px4_follow, 'OwnedSih', return_value=owned), \
                    patch.object(px4_follow, 'SihLink', side_effect=RuntimeError('transport failed')), \
                    patch.object(px4_follow, 'score', side_effect=RuntimeError('scorer failed')), \
                    patch('builtins.print'):
                try:
                    px4_follow.run(root/'run', smoke=True)
                except RuntimeError:
                    # The run may propagate the report failure or save a failed
                    # result, but ownership cleanup is mandatory in both cases.
                    pass
            owned.close.assert_called_once_with()

    def test_broken_status_output_does_not_skip_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.provenance_root(directory)
            owned = Mock()
            with patch.object(px4_follow, 'ROOT', root), \
                    patch.object(px4_follow, 'OwnedSih', return_value=owned), \
                    patch.object(px4_follow, 'SihLink', side_effect=RuntimeError('transport failed')), \
                    patch('builtins.print', side_effect=BrokenPipeError('status reader closed')):
                result = px4_follow.run(root/'run', smoke=True)
            owned.close.assert_called_once_with()
            self.assertFalse(result['passed'])
            self.assertIn('transport failed', result['error'])

    def interrupted_landing(self, root, stage, interruption, *, cleanup_error=False,
                            reporting_interrupt=False, earlier_failure=False):
        owned, worker, camera, recorder = (Mock() for _ in range(4))
        owned.receipt, owned.process.pid = {}, 123
        worker.receive.return_value = dict(kind='ready', model=None)
        link = Mock()
        link.health_error.return_value = ''
        link.truth = SimpleNamespace(lat=470000000, lon=80000000, alt=500000)
        link.telemetry = SimpleNamespace(position_ned=[0., 0., 0.], boot_ms=1000)
        link.attitude_source = link.truth_source = 1.
        link.clock.receipts, link.clock.pending, link.clock.reset = [], {}, False
        link.last_status = []
        def command(number, *args):
            if number == 400:
                if earlier_failure:
                    raise RuntimeError('original arm acknowledgement failure')
                raise px4_follow.StudioStop('stop while awaiting arm acknowledgement')
            if number == 21 and stage == 'command':
                raise interruption
        def wait_for(predicate, timeout, *args):
            if timeout == 30 and stage == 'confirmation':
                raise interruption
        link.command.side_effect, link.wait_for.side_effect = command, wait_for
        if cleanup_error:
            link.close.side_effect = RuntimeError('link cleanup failed independently')
            owned.close.side_effect = SystemExit('owned simulator cleanup interrupted')
        def status_output(value, **kwargs):
            if reporting_interrupt and json.loads(value)['event'] in ('landing_unconfirmed', 'cleanup_failed'):
                raise KeyboardInterrupt('reporting interrupted')
        with patch.object(px4_follow, 'ROOT', root), \
                patch.object(px4_follow, 'OwnedSih', return_value=owned), \
                patch.object(px4_follow, 'SihLink', return_value=link), \
                patch.object(px4_follow, 'VisionWorker', return_value=worker), \
                patch.object(px4_follow, 'SihCamera', return_value=camera), \
                patch.object(px4_follow, 'AsyncRecorder', return_value=recorder), \
                patch('experiments.mantis_recording.SessionRecorder'), \
                patch('builtins.print', side_effect=status_output):
            result = px4_follow.run(root/'run')
        for resource in (link, owned, worker, camera):
            resource.close.assert_called_once_with()
        recorder.finish.assert_called_once()
        self.assertIn(21, [call.args[0] for call in link.command.call_args_list])
        events = json.loads((root/'run/events.json').read_text())
        self.assertFalse(result['passed'])
        self.assertFalse(any(event['event'] == 'landed_disarmed' for event in events))
        self.assertEqual(json.loads((root/'run/summary.json').read_text()), result)
        landing = next(event for event in events if event['event'] == 'landing_unconfirmed')
        self.assertIn(type(interruption).__name__, landing['error'])
        return result, events

    def test_interrupt_during_land_command_or_confirmation_closes_every_resource(self):
        for stage in ('command', 'confirmation'):
            for interruption in (KeyboardInterrupt(), SystemExit('termination during landing')):
                with self.subTest(stage=stage, interruption=type(interruption).__name__), \
                        tempfile.TemporaryDirectory() as directory:
                    root = self.provenance_root(directory)
                    result, _ = self.interrupted_landing(root, stage, interruption)
                    self.assertEqual(result['error'], 'Landing confirmation unavailable')

    def test_landing_interrupt_does_not_mask_original_or_independent_cleanup_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.provenance_root(directory)
            result, events = self.interrupted_landing(root, 'confirmation', KeyboardInterrupt(),
                cleanup_error=True, reporting_interrupt=True, earlier_failure=True)
            self.assertEqual(result['error'], 'RuntimeError: original arm acknowledgement failure')
            failures = [event for event in events if event['event'] == 'cleanup_failed']
            self.assertEqual([event['resource'] for event in failures], ['link', 'px4'])
            self.assertIn('link cleanup failed independently', failures[0]['error'])
            self.assertIn('owned simulator cleanup interrupted', failures[1]['error'])
            reporting = [event for event in events if event['event'] == 'teardown_reporting_failed']
            self.assertEqual(len(reporting), 3)


if __name__ == '__main__':
    unittest.main()
