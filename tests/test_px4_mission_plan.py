"""Frozen mission scheduling and failure admission; fake runner only."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.run_px4_missions import (METHODS, SCENES, STORAGE_BUDGET_BYTES,
    RUN_ADMISSION_BYTES, METADATA_RESERVE_BYTES, SOURCE_CLOCK_ERROR, directory_bytes, make_plan, run_batch)


HASHES = {'experiments/fake_runtime.py': 'a'*64}


class FakeRunner:
    """Writes explicitly synthetic native receipts into temporary test folders."""
    def __init__(self, outcomes=None):
        self.calls = []
        self.outcomes = outcomes or {}

    def __call__(self, folder, **kwargs):
        index = len(self.calls)
        self.calls.append((folder, deepcopy(kwargs)))
        plan = json.loads((folder.parent/'plan.json').read_text())
        progress = json.loads((folder.parent/'progresssummary.json').read_text())
        assert progress['cases'][index]['status'] == 'running'
        assert all(item['status'] != 'running' for item in progress['cases'][:index])
        outcome = self.outcomes.get(index, {})
        if isinstance(outcome, BaseException):
            raise outcome
        folder.mkdir()
        case = plan['cases'][index]
        config = case['config']
        summary = dict(passed=True, actual_px4=True, session='synthetic-test-session', error=None,
            checks=dict(mission_window_complete=True, no_runtime_failure=True),
            metrics=dict(intended_actor='fixture-person', identity_unknown_count=2, tracking_loss_duration_s=1.,
                tracking_available_fraction=.9, correct_identity_duration_fraction=.8,
                command_valid_fraction=.7, wrong_person_frames=0))
        summary.update(outcome.get('summary', {}))
        events = [dict(event=name) for name in ('owned_px4_verified', 'armed_offboard',
                                              'stable_takeoff', 'landed_disarmed')]
        events.append(dict(event='mission_delay_injected', delay_s=1.))
        events = [event for event in events if event['event'] not in outcome.get('omit_events', [])]
        events.extend(outcome.get('events', []))
        mission = dict(method=kwargs['method'], **{key: config[key] for key in
            ('trajectory', 'target_speed', 'duration_s', 'faults', 'recovery', 'scenario_phase', 'noise_seed')})
        provenance = dict(session=summary['session'], mission_config=mission,
            source_sha256=plan['source_sha256'], estimator_profile=config['estimator_profile'],
            estimator_parameters=config['estimator_parameters'], px4_sensor_profile=config['sensor_profile'],
            selection=dict(track_id=config['selected_track']))
        provenance.update(outcome.get('provenance', {}))
        for name, value in [('summary.json', summary), ('events.json', events),
                            ('provenance.json', provenance)]:
            (folder/name).write_text(json.dumps(value))
        return summary


class MissionPlanTests(unittest.TestCase):
    def test_default_declares_every_pair_and_reverses_second_repeat(self):
        plan = make_plan(source_hashes=HASHES)
        self.assertEqual(len(plan['cases']), 18)
        self.assertEqual(len({case['case_id'] for case in plan['cases']}), 18)
        self.assertEqual(plan['stopping_policy']['consecutive_source_clock_failures'], 2)
        self.assertEqual(plan['stopping_policy']['source_clock_error'], SOURCE_CLOCK_ERROR)
        for repeat in (0, 1):
            expected = list(METHODS if repeat == 0 else reversed(METHODS))
            for scene in SCENES:
                cases = [case for case in plan['cases']
                         if case['repeat'] == repeat and case['config']['trajectory'] == scene]
                self.assertEqual([case['method'] for case in cases], expected)
                self.assertTrue(all(case['config'] == cases[0]['config'] for case in cases))
                self.assertEqual(cases[0]['config']['duration_s'], 60.)
                self.assertEqual(cases[0]['config']['target_speed'], .08)
                self.assertTrue(cases[0]['config']['faults'])
                self.assertFalse(cases[0]['config']['low_noise_sensors'])

    def test_configuration_bounds_and_explicit_diagnostic(self):
        for config in ({'repeats': 0}, {'repeats': 4}, {'repeats': True}, {'duration_s': 29},
                       {'duration_s': 121}, {'duration_s': float('nan')}, {'duration_s': True},
                       {'estimator_profile': 'invented'}, {'low_noise_sensors': 1}):
            with self.assertRaises(ValueError):
                make_plan(source_hashes=HASHES, **config)
        plan = make_plan(repeats=3, duration_s=120, low_noise_sensors=True,
                         estimator_profile='baro', source_hashes=HASHES)
        self.assertEqual(len(plan['cases']), 27)
        self.assertTrue(all(case['config']['low_noise_sensors'] for case in plan['cases']))
        self.assertTrue(all(case['config']['estimator_parameters'] for case in plan['cases']))

    def run_fake(self, root, fake, **kwargs):
        return run_batch(root/'batch', repeats=1, runner=fake, source_reader=lambda: HASHES, **kwargs)

    def test_serialized_declared_calls_retained_and_compared(self):
        with tempfile.TemporaryDirectory() as directory:
            root, fake = Path(directory), FakeRunner()
            result = self.run_fake(root, fake)
            self.assertEqual(len(fake.calls), 9)
            self.assertEqual(result['actual_px4_cases'], 9)  # Synthetic receipt contract only.
            self.assertEqual(result['passed_cases'], 9)
            self.assertEqual(len(result['comparison']['pairs']), 3)
            self.assertTrue(all(pair['comparable'] for pair in result['comparison']['pairs']))
            self.assertFalse(result['comparison']['superiority_established'])
            self.assertTrue(result['completed_all_declared_cases'])
            for folder, arguments in fake.calls:
                self.assertTrue(arguments['mission'])
                self.assertTrue(arguments['faults'])
                self.assertEqual(arguments['target_speed'], .08)
                self.assertTrue((folder/'batch_outcome.json').is_file())
            self.assertEqual(json.loads((root/'batch/progresssummary.json').read_text()), result)

    def test_tracking_failure_is_retained_without_retry_or_false_identity_credit(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeRunner({1: dict(summary=dict(passed=False, metrics={}))})
            result = self.run_fake(Path(directory), fake)
            self.assertEqual(len(fake.calls), 9)
            self.assertEqual(result['failed_cases'], 1)
            self.assertEqual(result['skipped_cases'], 0)
            self.assertFalse(result['passed'])
            self.assertEqual(result['unknown_identity']['cases_reporting_count'], 8)
            self.assertEqual(result['unknown_identity']['attempted_cases_without_count'], 1)
            self.assertFalse(result['unknown_identity']['missing_or_unknown_is_correct'])
            self.assertEqual(result['comparison']['methods']['direct']['failed_or_unqualified'], 1)

    def test_startup_exceptions_stop_after_two_and_do_not_claim_native_flight(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = FakeRunner({0: RuntimeError('model startup failed'), 1: RuntimeError('still unavailable')})
            result = self.run_fake(root, fake)
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(result['failed_cases'], 2)
            self.assertEqual(result['skipped_cases'], 7)
            self.assertEqual(result['actual_px4_cases'], 0)
            self.assertEqual(result['stopped_reason'], 'two_consecutive_failed_takeoffs')
            self.assertIsNone(result['unknown_identity']['count'])
            self.assertTrue(all(not pair['comparable'] for pair in result['comparison']['pairs']))
            for folder, _ in fake.calls:
                failure = json.loads((folder/'batch_failure.json').read_text())
                self.assertFalse(failure['actual_px4'])
            self.assertTrue(all(case.get('reason') == result['stopped_reason']
                                for case in result['cases'][2:]))

    def test_landing_or_resource_cleanup_failure_stops_immediately(self):
        for outcome, reason in [
                (dict(omit_events=['landed_disarmed']), 'landing_unconfirmed'),
                (dict(events=[dict(event='landing_unconfirmed')]), 'landing_unconfirmed'),
                (dict(events=[dict(event='cleanup_failed', resource='vision')]), 'resource_cleanup_error')]:
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                fake = FakeRunner({0: outcome})
                result = self.run_fake(Path(directory), fake)
                self.assertEqual(len(fake.calls), 1)
                self.assertEqual(result['stopped_reason'], reason)
                self.assertEqual(result['skipped_cases'], 8)
                self.assertEqual(result['failed_cases'], 1)

    def test_returned_native_flag_alone_is_not_px4_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeRunner({0: dict(omit_events=['owned_px4_verified'])})
            result = self.run_fake(Path(directory), fake)
            self.assertFalse(result['cases'][0]['actual_px4'])
            self.assertFalse(result['cases'][0]['summary']['passed'])
            self.assertFalse(result['comparison']['pairs'][0]['comparable'])

    def test_failed_takeoff_counter_resets_only_after_observed_stable_takeoff(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = dict(summary=dict(passed=False),
                          omit_events=['armed_offboard', 'stable_takeoff', 'landed_disarmed'])
            fake = FakeRunner({0: failed, 2: failed, 3: failed})
            result = self.run_fake(Path(directory), fake)
            self.assertEqual(len(fake.calls), 4)
            self.assertEqual(result['stopped_reason'], 'two_consecutive_failed_takeoffs')
            self.assertEqual(result['failed_cases'], 3)
            self.assertEqual(result['passed_cases'], 1)
            self.assertEqual(result['skipped_cases'], 5)

    def test_two_native_source_clock_failures_stop_and_preserve_attempts(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = dict(summary=dict(passed=False, error=SOURCE_CLOCK_ERROR))
            fake = FakeRunner({0: failure, 1: failure})
            result = self.run_fake(Path(directory), fake)
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(result['stopped_reason'], 'two_consecutive_source_clock_failures')
            self.assertEqual(result['failed_cases'], 2)
            self.assertEqual(result['actual_px4_cases'], 2)
            self.assertTrue(all(c['source_clock_failure'] for c in result['cases'][:2]))
            self.assertTrue(all(c['reason'] == result['stopped_reason'] for c in result['cases'][2:]))
            self.assertTrue(all((folder/'summary.json').is_file() for folder, _ in fake.calls))

    def test_clock_counter_resets_on_nonqualifying_and_non_native_outcomes(self):
        failure = dict(summary=dict(passed=False, error=SOURCE_CLOCK_ERROR))
        for middle in ({}, dict(summary=dict(passed=False, error='SitlError: another freshness problem')),
                       dict(summary=dict(passed=False, error=SOURCE_CLOCK_ERROR),
                            omit_events=['owned_px4_verified'])):
            with self.subTest(middle=middle), tempfile.TemporaryDirectory() as directory:
                fake = FakeRunner({0: failure, 1: middle, 2: failure, 3: failure})
                result = self.run_fake(Path(directory), fake)
                self.assertEqual(len(fake.calls), 4)
                self.assertFalse(result['cases'][1]['source_clock_failure'])
                self.assertEqual(result['stopped_reason'], 'two_consecutive_source_clock_failures')

    def test_landing_failure_takes_precedence_over_second_clock_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            failure = dict(summary=dict(passed=False, error=SOURCE_CLOCK_ERROR))
            fake = FakeRunner({0: failure, 1: dict(failure, omit_events=['landed_disarmed'])})
            result = self.run_fake(Path(directory), fake)
            self.assertEqual(len(fake.calls), 2)
            self.assertEqual(result['stopped_reason'], 'landing_unconfirmed')

    def test_declared_delay_must_actually_be_submitted(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeRunner({0: dict(omit_events=['mission_delay_injected'])})
            result = self.run_fake(Path(directory), fake)
            first = result['cases'][0]
            self.assertFalse(first['protocol_checks']['declared_delay_submitted'])
            self.assertFalse(first['summary']['passed'])
            self.assertTrue(first['runner_passed'])

    def test_mismatched_configuration_cannot_complete_a_comparable_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeRunner({2: dict(provenance=dict(estimator_profile='different'))})
            result = self.run_fake(Path(directory), fake)
            self.assertEqual(result['stopped_reason'], 'runner_configuration_or_source_mismatch')
            self.assertEqual(len(fake.calls), 3)
            self.assertTrue(all(not pair['comparable'] for pair in result['comparison']['pairs']))
            self.assertIn('unverified_configuration_case', result['cases'][2]['config'])

    def test_storage_stops_admission_and_preserves_existing_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root, fake = Path(directory), FakeRunner()
            def measured(folder):
                return (1024 if not fake.calls else
                        STORAGE_BUDGET_BYTES-RUN_ADMISSION_BYTES-METADATA_RESERVE_BYTES+1)
            result = self.run_fake(root, fake, byte_counter=measured)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(result['stopped_reason'], 'storage_admission_budget_exhausted')
            self.assertEqual(result['skipped_cases'], 8)
            self.assertTrue((fake.calls[0][0]/'summary.json').is_file())
            with self.assertRaises(FileExistsError):
                self.run_fake(root, fake)
            self.assertEqual(len(fake.calls), 1)

    def test_source_change_between_runs_stops_without_cherry_picking(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeRunner()
            def hashes():
                return HASHES if not fake.calls else {'experiments/fake_runtime.py': 'b'*64}
            result = run_batch(Path(directory)/'batch', repeats=1, runner=fake, source_reader=hashes)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(result['stopped_reason'], 'runtime_source_changed')
            self.assertEqual(result['skipped_cases'], 8)

    def test_keyboard_interrupt_is_preserved_and_stops_whole_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeRunner({0: KeyboardInterrupt()})
            result = self.run_fake(Path(directory), fake)
            self.assertEqual(len(fake.calls), 1)
            self.assertEqual(result['stopped_reason'], 'interrupted')
            self.assertEqual(result['failed_cases'], 1)
            self.assertEqual(result['skipped_cases'], 8)

    def test_directory_size_never_follows_external_links(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'outside').write_bytes(b'x'*1000)
            folder = root/'batch'
            folder.mkdir()
            (folder/'owned').write_bytes(b'a'*10)
            (folder/'link').symlink_to(root/'outside')
            self.assertEqual(directory_bytes(folder), 10+(folder/'link').lstat().st_size)


if __name__ == '__main__':
    unittest.main()
