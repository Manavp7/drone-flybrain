"""Bounded fake receipt files; never start PX4 or execute a saved snapshot."""
import copy
import ast
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from experiments.px4_evaluation import mission_score, paired_summary
from scripts.verify_px4_missions import (POLICY_SOURCES, REQUIRED_SOURCES, ROOT,
    PX4_COMMIT, VerificationError, verify_batch, verify_run)
from tests.test_px4_evaluation import observation, send


def write(folder, name, value):
    (folder/name).write_text(json.dumps(value, allow_nan=False))


def read(folder, name):
    return json.loads((folder/name).read_text())


def fixture(folder, *, method='neural', duration=1., faults=False):
    folder.mkdir(parents=True)
    hashes = {}
    for relative in REQUIRED_SOURCES:
        path = folder/'runtime'/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # A saved Python source that would fail if the verifier executed it.
        body = (ROOT/relative).read_bytes() if relative in (*POLICY_SOURCES, 'experiments/px4_follow.py') else b'raise RuntimeError("never execute saved code")\n'
        path.write_bytes(body)
        hashes[relative] = hashlib.sha256(body).hexdigest()
    config = dict(method=method, trajectory='walk', target_speed=.1, duration_s=duration,
                  faults=faults, recovery=False, scenario_phase='source clock', noise_seed='fixture')
    provenance = dict(session='test-session', source_sha256=hashes, px4_sensor_profile='stock sensor noise',
        estimator_profile='stock', estimator_parameters={}, mission_config=config, selection=dict(track_id=1))
    build = dict(px4_commit=PX4_COMMIT, profile='px4_sitl_mantis',
        build_recipe_sha256=hashes['scripts/build_px4_sih.py'],
        overlay_sha256=hashes['integrations/px4/sih.px4board'], binary_sha256='a'*64,
        startup_tree_sha256='b'*64, truth_header_sha256='c'*64,
        timestamp_patch='msg.time_usec = att.timestamp_sample;', submodules=[],
        sensor_noise_patches={f'src/sensor{i}.cpp': dict(original_sha256='d'*64, patched_sha256='e'*64)
                              for i in range(4)})
    events = [dict(event='owned_px4_verified', elapsed_s=0., pid=1, build=build,
                   px4_sensor_profile='stock sensor noise'),
              dict(event='stable_takeoff', elapsed_s=.1),
              dict(event='mission_window_complete', elapsed_s=duration+1),
              dict(event='landed_disarmed', elapsed_s=duration+2)]
    items = [observation(method=method), observation(1, 100.5, selected=False, method=method)]
    rows = [send(item) for item in items]
    summary = mission_score(rows, items, events, mission_start_s=100., mission_end_s=100.+duration,
                            min_tracking_fraction=.8)
    if faults:
        summary['checks'].update(actual_delay_injected=False, delayed_output_rejected=False)
        summary.update(passed=False, safety_passed=False, tracking_success=False)
    summary.update(mode=method+'_follow', session='test-session', actual_px4=True, error=None,
        px4_sensor_profile='stock sensor noise', estimator_profile='stock', mission_config=config,
        mission_start_s=100., mission_end_s=100.+duration)
    for name, value in [('provenance.json', provenance), ('summary.json', summary),
                        ('control.json', rows), ('observations.json', items), ('events.json', events)]:
        write(folder, name, value)
    return provenance, summary


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/'run'
        fixture(self.run)

    def test_recomputes_without_executing_runtime_python(self):
        result = verify_run(self.run)
        self.assertTrue(result['verified'])
        self.assertTrue(result['passed'])
        self.assertTrue(result['score_recomputed'])
        self.assertIn('not physics replay', result['scope'])

    def test_failed_trial_is_verified_as_failure_not_removed(self):
        shutil.rmtree(self.run)
        fixture(self.run, faults=True)
        result = verify_run(self.run)
        self.assertTrue(result['verified'])
        self.assertFalse(result['passed'])

    def test_rejects_changed_score_metrics_or_bool_schema(self):
        original = read(self.run, 'summary.json')
        for change in ('metric', 'check', 'bool'):
            summary = copy.deepcopy(original)
            if change == 'metric':
                summary['metrics']['wrong_person_frames'] = 99
            elif change == 'check':
                summary['checks']['landed_disarmed'] = False
            else:
                summary['safety_passed'] = 1
            write(self.run, 'summary.json', summary)
            with self.subTest(change=change), self.assertRaises(VerificationError):
                verify_run(self.run)

    def test_runtime_hash_escape_symlink_and_historical_policy_rejected(self):
        original = read(self.run, 'provenance.json')
        for path in ('../outside.py', '/tmp/outside.py'):
            provenance = copy.deepcopy(original)
            provenance['source_sha256'][path] = 'a'*64
            write(self.run, 'provenance.json', provenance)
            with self.subTest(path=path), self.assertRaises(VerificationError):
                verify_run(self.run)
        write(self.run, 'provenance.json', original)
        source = self.run/'runtime'/POLICY_SOURCES[0]
        source.write_text('# changed policy\n')
        with self.assertRaises(VerificationError):
            verify_run(self.run)
        original['source_sha256'][POLICY_SOURCES[0]] = hashlib.sha256(source.read_bytes()).hexdigest()
        write(self.run, 'provenance.json', original)
        with self.assertRaisesRegex(VerificationError, 'historical evaluator'):
            verify_run(self.run)
        source.unlink()
        source.symlink_to(ROOT/POLICY_SOURCES[0])
        with self.assertRaisesRegex(VerificationError, 'Symlink'):
            verify_run(self.run)

    def test_noise_session_and_build_receipt_must_agree(self):
        original = read(self.run, 'events.json')
        for field, value in [('pid', 0), ('px4_sensor_profile', '1% stock GPS/baro/mag/IMU noise'),
                             ('build', {})]:
            events = copy.deepcopy(original)
            events[0][field] = value
            write(self.run, 'events.json', events)
            with self.subTest(field=field), self.assertRaises(VerificationError):
                verify_run(self.run)

    def test_error_and_declared_duration_cannot_be_rewritten(self):
        summary = read(self.run, 'summary.json')
        summary['error'] = 'invented error'
        write(self.run, 'summary.json', summary)
        with self.assertRaisesRegex(VerificationError, 'Error is inconsistent'):
            verify_run(self.run)
        summary['error'] = None
        summary['mission_end_s'] = 100.8
        write(self.run, 'summary.json', summary)
        with self.assertRaisesRegex(VerificationError, 'shorter'):
            verify_run(self.run)

    def test_nonfinite_and_duplicate_json_keys_rejected(self):
        for text in ('{"passed":NaN}', '{"passed":true,"passed":false}'):
            (self.run/'summary.json').write_text(text)
            with self.subTest(text=text), self.assertRaises(VerificationError):
                verify_run(self.run)

    def test_gzipped_json_and_source_bytes_verified_exactly(self):
        for path in list(self.run.rglob('*')):
            if path.is_file():
                with gzip.open(str(path)+'.gz', 'wb') as stream:
                    stream.write(path.read_bytes())
                path.unlink()
        self.assertTrue(verify_run(self.run)['verified'])

    def test_shared_runtime_requires_matching_parent_batch_manifest(self):
        provenance = read(self.run, 'provenance.json')
        (self.run/'runtime').rename(self.root/'runtime')
        with self.assertRaises(VerificationError):
            verify_run(self.run)
        write(self.root, 'plan.json', dict(source_sha256=provenance['source_sha256']))
        self.assertTrue(verify_run(self.run)['verified'])
        write(self.root, 'plan.json', dict(source_sha256={}))
        with self.assertRaisesRegex(VerificationError, 'Shared runtime'):
            verify_run(self.run)

    def test_only_comparison_body_may_differ_without_changing_run_policy(self):
        source = self.run/'runtime/experiments/px4_evaluation.py'
        original = source.read_text()
        tree = ast.parse(original)
        comparison = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                          and node.name == 'paired_summary')
        comparison.body = [ast.Raise(exc=ast.Call(func=ast.Name(id='RuntimeError', ctx=ast.Load()),
            args=[ast.Constant('must never execute this saved function')], keywords=[]), cause=None)]
        source.write_text(ast.unparse(tree))
        provenance = read(self.run, 'provenance.json')
        provenance['source_sha256']['experiments/px4_evaluation.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
        write(self.run, 'provenance.json', provenance)
        self.assertTrue(verify_run(self.run)['comparison_only_source_change'])
        # Changing a default/decorator or any scoring statement is not exempt.
        comparison.args.defaults = [ast.Constant('changed default')]
        source.write_text(ast.unparse(tree))
        provenance['source_sha256']['experiments/px4_evaluation.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
        write(self.run, 'provenance.json', provenance)
        with self.assertRaisesRegex(VerificationError, 'historical evaluator'):
            verify_run(self.run)

    def test_studio_scope_is_descriptive_and_benchmark_claim_rejected(self):
        summary = read(self.run, 'summary.json')
        summary.update(evaluation_scope='operator_demonstration', tracking_success=None)
        write(self.run, 'summary.json', summary)
        result = verify_run(self.run)
        self.assertFalse(result['benchmark_verified'])
        self.assertFalse(result['score_recomputed'])
        self.assertFalse(result['mission_policy_compatibility_checked'])
        source = self.run/'runtime/experiments/px4_evaluation.py'
        source.write_text('raise RuntimeError("operator snapshot must never be executed")\n')
        provenance = read(self.run, 'provenance.json')
        provenance['source_sha256']['experiments/px4_evaluation.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
        write(self.run, 'provenance.json', provenance)
        self.assertTrue(verify_run(self.run)['verified'])
        summary['tracking_success'] = True
        write(self.run, 'summary.json', summary)
        with self.assertRaisesRegex(VerificationError, 'Studio'):
            verify_run(self.run)

    def test_smoke_recomputes_raw_truth_hover_metrics(self):
        from experiments.px4_follow import score
        events = read(self.run, 'events.json')
        row = dict(phase='smoke_hold', elapsed_s=3., stall_elapsed_s=0., sent_speed=0.,
            position=[0., 0., 1.1], velocity=[0., 0., 0.], truth_velocity_ned=[0., 0., 0.],
            truth_altitude_above_start_m=1.1)
        summary = read(self.run, 'summary.json')
        for key in ('metrics', 'tracking_success', 'safety_passed', 'tracking_policy', 'tracking_checks', 'recovery'):
            summary.pop(key)
        summary.update(score([row], [], events, smoke=True), mode='autopilot_smoke',
            hover_metrics=dict(duration_s=3., estimated_position_error_max_m=0.,
                estimated_speed_p95_m_s=0., truth_speed_p95_m_s=0., truth_altitude_range_m=0.))
        write(self.run, 'summary.json', summary)
        write(self.run, 'control.json', [row])
        write(self.run, 'observations.json', [])
        self.assertTrue(verify_run(self.run)['verified'])
        summary['hover_metrics']['truth_speed_p95_m_s'] = .2
        write(self.run, 'summary.json', summary)
        with self.assertRaisesRegex(VerificationError, 'hover_metrics'):
            verify_run(self.run)

    def test_prearm_failed_attempt_has_no_native_execution_claim(self):
        error = 'RuntimeError: model unavailable'
        events = [dict(event='failure', elapsed_s=.1, error=error)]
        summary = read(self.run, 'summary.json')
        summary.update(mission_score([], [], events, mission_start_s=100., mission_end_s=101.,
                                     min_tracking_fraction=.8), actual_px4=False, error=error)
        for name, value in [('events.json', events), ('control.json', []),
                            ('observations.json', []), ('summary.json', summary)]:
            write(self.run, name, value)
        result = verify_run(self.run)
        self.assertTrue(result['verified'])
        self.assertFalse(result['actual_px4_receipt_verified'])
        self.assertFalse(result['passed'])

    def test_cleanup_error_cannot_be_hidden_after_scoring(self):
        events = read(self.run, 'events.json')
        events.append(dict(event='cleanup_failed', elapsed_s=4., resource='recording', error='encoder failed'))
        summary = read(self.run, 'summary.json')
        summary.update(passed=False, error='recording cleanup failed: encoder failed')
        write(self.run, 'summary.json', summary)
        write(self.run, 'events.json', events)
        result = verify_run(self.run)
        self.assertTrue(result['verified'])
        self.assertFalse(result['passed'])


class BatchVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = dict(schema_version=1, methods=['neural', 'direct', 'filtered'],
            scenes=['walk', 'crossing', 'occlusion'], repeats=1, cases=[],
            stopping_policy=dict(consecutive_failed_takeoffs=2, consecutive_source_clock_failures=2,
                source_clock_error='SitlError: no_fresh_preceding_source_clock_receipt'))
        self.cases = []
        for scene in self.plan['scenes']:
            for method in self.plan['methods']:
                case_id = scene+'-'+method
                config = dict(trajectory=scene, target_speed=.1, duration_s=1., faults=False,
                    recovery=False, scenario_phase='source clock', noise_seed='fixture',
                    low_noise_sensors=False, sensor_profile='stock sensor noise',
                    estimator_profile='stock', estimator_parameters={}, selected_track=1)
                declaration = dict(case_id=case_id, method=method, repeat=0, config=config)
                self.plan['cases'].append(declaration)
                if len(self.cases):
                    self.cases.append(dict(declaration, status='skipped', actual_px4=False, reason='test stop'))
                    continue
                folder = self.root/case_id
                provenance, summary = fixture(folder, method=method)
                self.plan['source_sha256'] = provenance['source_sha256']
                checks = dict(declared_configuration_matches=True, frozen_sources_match=True,
                              actual_px4_verified=True, declared_delay_submitted=False)
                summary['passed'] = False  # batch's mandatory delay protocol was not met
                outcome = dict(declaration, declared_config=config, status='failed', actual_px4=True,
                    runner_passed=True, summary=summary, protocol_checks=checks, evidence_errors=[],
                    runner_exception=None, stop_reason=None, source_clock_failure=False,
                    stable_takeoff=True, landed_disarmed=True)
                self.cases.append(outcome)
                write(folder, 'batch_outcome.json', outcome)
        write(self.root, 'plan.json', self.plan)
        self.progress = dict(schema_version=1, declared_cases=9, attempted_cases=1,
            passed_cases=0, failed_cases=1, skipped_cases=8, actual_px4_cases=1,
            completed_all_declared_cases=False, finished=True, passed=False,
            unknown_identity=dict(count=0, cases_reporting_count=1, attempted_cases_without_count=0,
                                  missing_or_unknown_is_correct=False),
            cases=self.cases, comparison=paired_summary(self.cases[:1]))
        write(self.root, 'progresssummary.json', self.progress)

    def store_first_case(self, case):
        write(self.root/case['case_id'], 'batch_outcome.json', case)
        progress = copy.deepcopy(self.progress)
        progress['cases'][0] = case
        unknown = case['summary']['metrics'].get('identity_unknown_count')
        progress.update(actual_px4_cases=int(case['actual_px4']), comparison=paired_summary([case]),
            unknown_identity=dict(count=unknown, cases_reporting_count=int(unknown is not None),
                attempted_cases_without_count=int(unknown is None), missing_or_unknown_is_correct=False))
        progress['stopped_reason'] = case['stop_reason']
        if case['stop_reason'] is not None:
            for skipped in progress['cases'][1:]:
                skipped['reason'] = case['stop_reason']
        write(self.root, 'progresssummary.json', progress)

    def test_preserves_failed_and_skipped_cases_in_independent_aggregate(self):
        result = verify_batch(self.root)
        self.assertTrue(result['verified'])
        self.assertFalse(result['passed'])
        self.assertEqual(result['skipped_cases'], 8)
        self.assertEqual(result['failed_cases'], 1)
        self.assertFalse(result['completed_all_declared_cases'])

    def test_count_pair_and_noise_tampering_rejected(self):
        for change in ('count', 'comparison', 'noise'):
            progress = copy.deepcopy(self.progress)
            if change == 'count':
                progress['passed_cases'] = 1
            elif change == 'comparison':
                progress['comparison']['superiority_established'] = True
            else:
                progress['cases'][0]['config']['sensor_profile'] = '1% stock GPS/baro/mag/IMU noise'
            write(self.root, 'progresssummary.json', progress)
            with self.subTest(change=change), self.assertRaises(VerificationError):
                verify_batch(self.root)

    def test_missing_case_is_reported_and_must_not_count_as_completed(self):
        progress = copy.deepcopy(self.progress)
        progress['cases'][-1]['status'] = 'planned'
        progress.update(skipped_cases=7, finished=False)
        write(self.root, 'progresssummary.json', progress)
        result = verify_batch(self.root)
        self.assertEqual(result['missing_cases'], [self.cases[-1]['case_id']])

    def test_declared_matrix_cannot_drop_a_method(self):
        self.plan['cases'].pop()
        write(self.root, 'plan.json', self.plan)
        with self.assertRaises(VerificationError):
            verify_batch(self.root)

    def test_skipped_or_planned_status_cannot_hide_an_attempted_failure(self):
        for status in ('skipped', 'planned'):
            progress = copy.deepcopy(self.progress)
            progress['cases'][0].update(status=status, actual_px4=False)
            write(self.root, 'progresssummary.json', progress)
            with self.subTest(status=status), self.assertRaisesRegex(VerificationError, 'hides original'):
                verify_batch(self.root)

    def test_startup_exception_requires_exact_outcome_without_invented_metrics(self):
        case = copy.deepcopy(self.cases[0])
        folder = self.root/case['case_id']
        shutil.rmtree(folder)
        folder.mkdir()
        error = 'RuntimeError: startup unavailable'
        case.update(actual_px4=False, runner_passed=False, stable_takeoff=False, landed_disarmed=False,
            runner_exception=error, summary=dict(passed=False, actual_px4=False, error=error, metrics={}),
            protocol_checks=dict(declared_configuration_matches=False, frozen_sources_match=False,
                                 actual_px4_verified=False, declared_delay_submitted=False))
        case['config'] = dict(case['declared_config'], unverified_configuration_case=case['case_id'])
        write(folder, 'batch_failure.json', dict(error=error, actual_px4=False))
        write(folder, 'batch_outcome.json', case)
        progress = copy.deepcopy(self.progress)
        progress['cases'][0] = case
        progress.update(actual_px4_cases=0, comparison=paired_summary([case]),
            unknown_identity=dict(count=None, cases_reporting_count=0, attempted_cases_without_count=1,
                                  missing_or_unknown_is_correct=False))
        write(self.root, 'progresssummary.json', progress)
        self.assertTrue(verify_batch(self.root)['verified'])
        case['summary']['metrics'] = dict(command_valid_fraction=1.)
        progress['comparison'] = paired_summary([case])
        write(folder, 'batch_outcome.json', case)
        write(self.root, 'progresssummary.json', progress)
        with self.assertRaisesRegex(VerificationError, 'cannot invent'):
            verify_batch(self.root)

    def test_genuine_driver_stop_reasons_survive_independent_verification(self):
        from scripts.run_px4_missions import _receipt
        declaration = self.plan['cases'][0]
        folder = self.root/declaration['case_id']
        original_events = read(folder, 'events.json')
        original_summary = read(folder, 'summary.json')
        rows, items = read(folder, 'control.json'), read(folder, 'observations.json')
        for reason in ('resource_cleanup_error', 'landing_unconfirmed', 'interrupted'):
            events = copy.deepcopy(original_events)
            if reason == 'resource_cleanup_error':
                events.append(dict(event='cleanup_failed', elapsed_s=4., resource='recording', error='encoder failed'))
                error = 'recording cleanup failed: encoder failed'
            elif reason == 'landing_unconfirmed':
                events = [e for e in events if e['event'] != 'landed_disarmed']
                events.append(dict(event='landing_unconfirmed', elapsed_s=4., error='confirmation timeout'))
                error = 'Landing confirmation unavailable'
            else:
                events.insert(-1, dict(event='failure', elapsed_s=2.5, error='KeyboardInterrupt: stopped'))
                error = 'KeyboardInterrupt: stopped'
            saved = dict(original_summary)
            saved.update(mission_score(rows, items, events, mission_start_s=100., mission_end_s=101.,
                                       min_tracking_fraction=.8), passed=False, error=error)
            write(folder, 'events.json', events)
            write(folder, 'summary.json', saved)
            case = _receipt(declaration, folder, saved, None, self.plan)
            self.assertEqual(case['stop_reason'], reason)
            self.store_first_case(case)
            with self.subTest(reason=reason):
                result = verify_batch(self.root)
                self.assertTrue(result['verified'])
                self.assertFalse(result['passed'])
                case['summary']['batch_rejection_reason'] = 'invented'
                self.store_first_case(case)
                with self.assertRaisesRegex(VerificationError, 'changed the original'):
                    verify_batch(self.root)

    def test_startup_interrupt_and_partial_configuration_reasons_match_driver(self):
        from scripts.run_px4_missions import _receipt
        declaration = self.plan['cases'][0]
        folder = self.root/declaration['case_id']
        shutil.rmtree(folder)
        folder.mkdir()
        for error, provenance, reason in (
            ('KeyboardInterrupt: stopped', None, 'interrupted'),
            ('SystemExit: 1', {}, 'interrupted'),
            ('RuntimeError: recording cleanup failed', {}, 'resource_cleanup_error'),
            ('RuntimeError: startup unavailable', {}, 'runner_configuration_or_source_mismatch'),
            ('RuntimeError: startup unavailable', None, None),
        ):
            path = folder/'provenance.json'
            if provenance is None:
                path.unlink(missing_ok=True)
            else:
                write(folder, 'provenance.json', provenance)
            write(folder, 'batch_failure.json', dict(error=error, actual_px4=False))
            case = _receipt(declaration, folder, None, error, self.plan)
            self.assertEqual(case['stop_reason'], reason)
            self.store_first_case(case)
            with self.subTest(error=error, provenance=provenance):
                self.assertTrue(verify_batch(self.root)['verified'])
                case.update(stop_reason='landing_unconfirmed')
                case['summary']['batch_rejection_reason'] = 'landing_unconfirmed'
                self.store_first_case(case)
                with self.assertRaisesRegex(VerificationError, 'stop reason'):
                    verify_batch(self.root)

    def test_stop_precedence_and_exception_receipt(self):
        from scripts.run_px4_missions import _receipt
        declaration = self.plan['cases'][0]
        folder = self.root/declaration['case_id']
        events = read(folder, 'events.json')
        events = [e for e in events if e['event'] != 'landed_disarmed']
        events.extend([dict(event='failure', elapsed_s=2.5, error='KeyboardInterrupt: stopped'),
                       dict(event='cleanup_failed', elapsed_s=4., resource='recording', error='failed')])
        saved = read(folder, 'summary.json')
        saved.update(mission_score(read(folder, 'control.json'), read(folder, 'observations.json'), events,
            mission_start_s=100., mission_end_s=101., min_tracking_fraction=.8),
            passed=False, error='KeyboardInterrupt: stopped')
        write(folder, 'summary.json', saved)
        write(folder, 'events.json', events)
        write(folder, 'batch_failure.json', dict(error='KeyboardInterrupt: stopped', actual_px4=False))
        case = _receipt(declaration, folder, saved, 'KeyboardInterrupt: stopped', self.plan)
        self.assertEqual(case['stop_reason'], 'resource_cleanup_error')
        self.store_first_case(case)
        self.assertTrue(verify_batch(self.root)['verified'])
        case['stop_reason'] = case['summary']['batch_rejection_reason'] = 'interrupted'
        self.store_first_case(case)
        with self.assertRaisesRegex(VerificationError, 'stop reason'):
            verify_batch(self.root)

    def test_invented_reason_without_supporting_receipts_is_rejected(self):
        case = copy.deepcopy(self.cases[0])
        case['stop_reason'] = case['summary']['batch_rejection_reason'] = 'resource_cleanup_error'
        self.store_first_case(case)
        with self.assertRaisesRegex(VerificationError, 'stop reason'):
            verify_batch(self.root)

    def source_clock_batch(self, *, count=2):
        from scripts.run_px4_missions import _receipt
        cases = copy.deepcopy(self.cases)
        error = 'SitlError: no_fresh_preceding_source_clock_receipt'
        for index in range(count):
            declaration = self.plan['cases'][index]
            folder = self.root/declaration['case_id']
            if index:
                fixture(folder, method=declaration['method'])
            events = read(folder, 'events.json')
            events.insert(-1, dict(event='failure', elapsed_s=2.5, error=error))
            saved = read(folder, 'summary.json')
            saved.update(mission_score(read(folder, 'control.json'), read(folder, 'observations.json'), events,
                mission_start_s=100., mission_end_s=101., min_tracking_fraction=.8), passed=False, error=error)
            write(folder, 'summary.json', saved)
            write(folder, 'events.json', events)
            cases[index] = _receipt(declaration, folder, saved, None, self.plan)
            write(folder, 'batch_outcome.json', cases[index])
        reason = 'two_consecutive_source_clock_failures'
        for case in cases[count:]:
            case['reason'] = reason
        progress = copy.deepcopy(self.progress)
        progress.update(cases=cases, attempted_cases=count, failed_cases=count, skipped_cases=9-count,
            actual_px4_cases=count, stopped_reason=reason, comparison=paired_summary(cases[:count]),
            unknown_identity=dict(count=0, cases_reporting_count=count, attempted_cases_without_count=0,
                                  missing_or_unknown_is_correct=False))
        write(self.root, 'progresssummary.json', progress)
        return progress

    def test_source_clock_abort_is_recomputed_from_both_original_failures(self):
        progress = self.source_clock_batch()
        result = verify_batch(self.root)
        self.assertTrue(result['verified'])
        self.assertFalse(result['completed_all_declared_cases'])
        self.assertEqual(result['actual_px4_cases'], 2)
        for change in ('classification', 'reason', 'suffix', 'policy'):
            changed = copy.deepcopy(progress)
            if change == 'classification':
                changed['cases'][0]['source_clock_failure'] = False
                write(self.root/changed['cases'][0]['case_id'], 'batch_outcome.json', changed['cases'][0])
            elif change == 'reason':
                changed['stopped_reason'] = None
            elif change == 'suffix':
                changed['cases'][2]['reason'] = 'invented'
            else:
                plan = copy.deepcopy(self.plan)
                plan['stopping_policy']['consecutive_source_clock_failures'] = 3
                write(self.root, 'plan.json', plan)
            write(self.root, 'progresssummary.json', changed)
            with self.subTest(change=change), self.assertRaises(VerificationError):
                verify_batch(self.root)
            write(self.root/progress['cases'][0]['case_id'], 'batch_outcome.json', progress['cases'][0])
            write(self.root, 'plan.json', self.plan)

    def test_cannot_stop_after_one_or_continue_after_two_clock_failures(self):
        self.source_clock_batch(count=1)
        with self.assertRaisesRegex(VerificationError, 'lacks the original'):
            verify_batch(self.root)
        self.source_clock_batch(count=3)
        with self.assertRaisesRegex(VerificationError, 'skipped suffix|continued'):
            verify_batch(self.root)


if __name__ == '__main__':
    unittest.main()
