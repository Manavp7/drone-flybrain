"""Offline receipt/source consistency checks, not physics replay or attestation.

Only the installed, reviewed evaluator is imported. Saved runtime Python files
are hashed as bytes and never executed. A consistent failed run remains failed;
an incomplete batch remains incomplete. Studio sessions are operator demos.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.px4_evaluation import mission_score, paired_summary

PX4_COMMIT = '54f0455ffcd755534539a7cf33a09a20bf71d29d'
SCOPE = ('Saved source/receipt consistency and current reviewed gate recomputation; '
         'not physics replay, binary execution attestation, or hardware validation')
POLICY_SOURCES = ('experiments/px4_evaluation.py', 'experiments/mantis_arena.py', 'perception/pipeline.py')
REQUIRED_SOURCES = (*POLICY_SOURCES, 'experiments/px4_follow.py', 'flybrain_sim/px4_sih.py',
                    'scripts/build_px4_sih.py', 'integrations/px4/sih.px4board')


class VerificationError(ValueError):
    pass


def _require(condition, message):
    if not condition:
        raise VerificationError(message)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _path(root, relative):
    _require(isinstance(relative, str) and relative and '\\' not in relative and '\0' not in relative,
             'Invalid evidence path')
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and '..' not in pure.parts and pure.as_posix() == relative
             and relative != '.', 'Unsafe evidence path: '+relative)
    candidate = root
    for part in pure.parts:
        candidate = candidate/part
        _require(not candidate.is_symlink(), 'Symlink evidence is unsupported: '+relative)
    _require(candidate.resolve().is_relative_to(root.resolve()), 'Evidence path escapes its root')
    return candidate


def _locate(folder, name):
    path = _path(folder, name)
    return path if path.is_file() else _path(folder, name+'.gz')


def _exists(folder, name):
    return _locate(folder, name).is_file()


def _bytes(path):
    _require(path.is_file() and path.stat().st_size <= 64*1024**2, 'Missing/oversized evidence: '+path.name)
    opener = gzip.open if path.suffix == '.gz' else open
    try:
        with opener(path, 'rb') as stream:
            value = stream.read(64*1024**2+1)
    except (OSError, EOFError) as exc:
        raise VerificationError('Unreadable/compressed evidence: '+path.name) from exc
    _require(len(value) <= 64*1024**2, 'Decompressed evidence exceeds size limit')
    return value


def _read(folder, name, kind):
    path = _locate(folder, name)
    def reject(value):
        raise VerificationError('Nonfinite JSON: '+value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            _require(key not in result, 'Duplicate JSON key: '+key)
            result[key] = value
        return result
    try:
        value = json.loads(_bytes(path), parse_constant=reject, object_pairs_hook=unique)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise VerificationError('Unreadable receipt: '+name) from exc
    _require(isinstance(value, kind), 'Wrong receipt type: '+name)
    return value


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _hash(path):
    return hashlib.sha256(_bytes(path)).hexdigest()


def _run_policy_ast(body):
    """Ignore only the unused comparison function's body; retain its signature.

    Decorators/default arguments still execute on import, so they are compared.
    Every scoring helper, constant, import and other top-level statement must
    remain identical. Parsing is bounded and never imports the saved source.
    """
    _require(len(body) <= 2*1024**2, 'Evaluator source exceeds parser bound')
    try:
        tree = ast.parse(body)
        comparisons = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                       and node.name == 'paired_summary']
        _require(len(comparisons) == 1, 'Unrecognized evaluator comparison function')
        comparisons[0].body = [ast.Pass()]
        return ast.dump(tree, include_attributes=False)
    except (SyntaxError, RecursionError) as exc:
        raise VerificationError('Unparseable historical evaluator') from exc


def _sources(folder, provenance, *, check_policy=True):
    hashes = provenance.get('source_sha256')
    _require(isinstance(hashes, dict) and hashes, 'Missing source hash manifest')
    _require(all(name in hashes for name in REQUIRED_SOURCES), 'Incomplete runtime source manifest')
    snapshot = _path(folder, 'runtime')
    if not snapshot.is_dir():
        snapshot = _path(folder, 'sources')
    if not snapshot.is_dir() and _exists(folder.parent, 'plan.json'):
        plan = _read(folder.parent, 'plan.json', dict)
        _require(plan.get('source_sha256') == hashes, 'Shared runtime requires identical batch source hashes')
        snapshot = _path(folder.parent, 'runtime')
    _require(snapshot.is_dir(), 'A runtime/source snapshot is required; current-tree fallback is forbidden')
    for relative, expected in hashes.items():
        _require(_digest(expected), 'Invalid source hash: '+str(relative))
        path = _locate(snapshot, relative)
        _require(path.is_file() and _hash(path) == expected, 'Runtime source mismatch: '+relative)
    if not check_policy:
        return dict(count=len(hashes), comparison_only_source_change=None)
    # If policy changed, consistency needs a reviewed historical verifier, not
    # execution/import of the saved (potentially untrusted) implementation.
    comparison_only = False
    for relative in POLICY_SOURCES:
        if hashes[relative] == _hash(ROOT/relative):
            continue
        if relative == 'experiments/px4_evaluation.py':
            try:
                compatible = _run_policy_ast(_bytes(_locate(snapshot, relative))) == _run_policy_ast(_bytes(ROOT/relative))
            except VerificationError:
                compatible = False
            _require(compatible, 'Unsupported historical evaluator: '+relative)
            comparison_only = True
        else:
            raise VerificationError('Unsupported historical evaluator: '+relative)
    return dict(count=len(hashes), comparison_only_source_change=comparison_only)


def _error_from_events(events):
    failures = [e for e in events if e['event'] == 'failure']
    if failures:
        return failures[0].get('error')
    if any(e['event'] == 'landing_unconfirmed' for e in events):
        return 'Landing confirmation unavailable'
    cleanup = next((e for e in events if e['event'] == 'cleanup_failed'), None)
    return None if cleanup is None else f"{cleanup.get('resource')} cleanup failed: {cleanup.get('error')}"


def _batch_stop_reason(events, error, exception, errors, *, provenance_present,
                       configuration_matches, source_matches):
    """Reconstruct driver stop precedence; never trust the supplied reason."""
    _require(exception is None or isinstance(exception, str) and bool(exception),
             'Invalid runner exception')
    _require(isinstance(errors, list) and all(isinstance(item, str) and (
        item in ('runner_events_unavailable', 'runner_summary_unavailable',
                 'saved_summary_mismatch', 'runner_metrics_invalid')
        or item.startswith('unreadable_runner_evidence: ')) for item in errors),
        'Unrecognized runner evidence error')
    names = {event['event'] for event in events}
    if 'cleanup_failed' in names or 'cleanup failed' in str(error).lower():
        return 'resource_cleanup_error'
    if ('landing_unconfirmed' in names or
        bool(names & {'armed_offboard', 'stable_takeoff', 'land_requested'})
            and 'landed_disarmed' not in names):
        return 'landing_unconfirmed'
    if ((exception and exception.startswith(('KeyboardInterrupt:', 'SystemExit:')))
        or str(error).startswith(('KeyboardInterrupt:', 'SystemExit:'))):
        return 'interrupted'
    if errors and exception is None:
        return 'runner_evidence_unavailable'
    if provenance_present and (not configuration_matches or not source_matches):
        return 'runner_configuration_or_source_mismatch'
    return None


def _batch_wrapped_stop(case, wrapped, events, *, provenance_present,
                        configuration_matches, source_matches):
    reason = _batch_stop_reason(events, wrapped.get('error'), case.get('runner_exception'),
        case.get('evidence_errors'), provenance_present=provenance_present,
        configuration_matches=configuration_matches, source_matches=source_matches)
    _require('stop_reason' in case and case['stop_reason'] == reason,
             'Batch stop reason differs from original receipts')
    if reason is not None:
        wrapped.update(passed=False, batch_rejection_reason=reason)
    return wrapped


def _native_receipt(summary, provenance, events):
    _require(type(summary.get('actual_px4')) is bool, 'actual_px4 must be a boolean receipt claim')
    owned = [e for e in events if e['event'] == 'owned_px4_verified']
    _require(len(owned) == int(summary['actual_px4']), 'Actual-PX4 verification event mismatch')
    if not owned:
        _require(summary['passed'] is False and summary['error'] is not None,
                 'No native verification receipt supports this successful run')
        return False
    event, build = owned[0], owned[0].get('build')
    _require(type(event.get('pid')) is int and event['pid'] > 0, 'Invalid owned PX4 PID receipt')
    _require(event.get('px4_sensor_profile') == provenance['px4_sensor_profile'], 'Owned PX4 noise mismatch')
    _require(isinstance(build, dict) and build.get('px4_commit') == PX4_COMMIT
             and build.get('profile') == 'px4_sitl_mantis', 'Unrecognized PX4 build receipt')
    for name in ('build_recipe_sha256', 'binary_sha256', 'startup_tree_sha256',
                 'overlay_sha256', 'truth_header_sha256'):
        _require(_digest(build.get(name)), 'Invalid PX4 build digest: '+name)
    _require(build['build_recipe_sha256'] == provenance['source_sha256']['scripts/build_px4_sih.py']
        and build['overlay_sha256'] == provenance['source_sha256']['integrations/px4/sih.px4board'],
        'Build recipe/board differs from the frozen runtime')
    _require(isinstance(build.get('timestamp_patch'), str) and 'timestamp_sample' in build['timestamp_patch'],
             'Missing conservative truth timestamp patch receipt')
    patches = build.get('sensor_noise_patches')
    _require(isinstance(patches, dict) and len(patches) == 4, 'Missing sensor patch provenance')
    for name, patch in patches.items():
        _path(Path('/').resolve(), name)  # lexical path validation only; no file reads
        _require(isinstance(patch, dict) and _digest(patch.get('original_sha256'))
                 and _digest(patch.get('patched_sha256')), 'Invalid sensor patch digest')
    _require(isinstance(build.get('submodules'), list)
             and all(isinstance(s, str) for s in build['submodules']), 'Missing submodule provenance')
    return True


def _compare(summary, expected):
    for key, value in expected.items():
        _require(key in summary and json.dumps(summary[key], sort_keys=True, allow_nan=False)
                 == json.dumps(value, sort_keys=True, allow_nan=False), 'Recomputed summary mismatch: '+key)


def verify_run(folder):
    """Raise on inconsistent evidence; otherwise retain the run's passed=False."""
    folder = Path(folder).resolve()
    provenance, summary = (_read(folder, name, dict) for name in ('provenance.json', 'summary.json'))
    operator = summary.get('evaluation_scope') == 'operator_demonstration' or summary.get('mode') == 'operator_demonstration'
    source_info = _sources(folder, provenance, check_policy=not operator)
    sources = source_info['count']
    rows, observations, events = (_read(folder, name, list) for name in
        ('control.json', 'observations.json', 'events.json'))
    _require(all(isinstance(r, dict) for r in rows+observations+events), 'Invalid row/event schema')
    _require(all(isinstance(o.get('packet'), dict) and isinstance(o.get('command'), dict)
                 for o in observations), 'Invalid observation/command schema')
    _require(all(isinstance(e.get('event'), str) and _finite(e.get('elapsed_s'))
                 and e['elapsed_s'] >= 0 for e in events), 'Invalid event identity/time')
    _require(all(isinstance(e.get('error'), str) and e['error'] for e in events
                 if e['event'] in ('failure', 'landing_unconfirmed', 'cleanup_failed')),
             'Failure events require their original error text')
    _require(all(a['elapsed_s'] <= b['elapsed_s'] for a, b in zip(events, events[1:])),
             'Event clock regressed')
    _require(type(summary.get('passed')) is bool and 'error' in summary
             and (summary['error'] is None or isinstance(summary['error'], str)), 'Invalid verdict/error schema')
    _require(isinstance(provenance.get('session'), str) and bool(provenance['session'])
             and summary.get('session') == provenance['session'], 'Session mismatch')
    _require(all(o.get('packet', {}).get('session') == provenance['session'] for o in observations),
             'Observation session differs from run')
    for field in ('px4_sensor_profile', 'estimator_profile', 'mission_config'):
        _require(field in provenance and summary.get(field) == provenance[field], 'Provenance mismatch: '+field)
    _require(provenance['px4_sensor_profile'] in ('stock sensor noise', '1% stock GPS/baro/mag/IMU noise'),
             'Unknown noise profile')
    config = provenance['mission_config']
    _require(isinstance(config, dict) and type(config.get('faults')) is bool
             and type(config.get('recovery')) is bool, 'Mission fault/recovery booleans required')
    _require(summary['error'] == _error_from_events(events), 'Error is inconsistent with event receipts')
    actual = _native_receipt(summary, provenance, events)
    if provenance.get('selection', {}).get('track_id') is None or operator:
        _require(operator, 'Interactive selection requires explicit operator-demonstration scope')
        _require(summary.get('tracking_success') is not True and summary.get('benchmark_verified') is not True,
                 'Studio cannot claim a fixed-identity benchmark pass')
        return dict(verified=True, passed=summary['passed'], actual_px4_receipt_verified=actual,
            benchmark_verified=False, evaluation_scope='operator_demonstration', source_count=sources,
            comparison_only_source_change=source_info['comparison_only_source_change'],
            mission_policy_compatibility_checked=False,
            score_recomputed=False, scope=SCOPE+'; operator score is descriptive and unsupported as a benchmark')
    smoke = summary.get('mode') == 'autopilot_smoke'
    if smoke or 'metrics' not in summary:
        from experiments.px4_follow import score
        _require(provenance['source_sha256']['experiments/px4_follow.py'] == _hash(ROOT/'experiments/px4_follow.py'),
                 'Unsupported historical legacy scorer')
        expected = score(rows, observations, events, smoke=smoke)
        if smoke:
            _require(all(r.get('sent_speed') == 0 for r in rows), 'Smoke emitted forward motion')
        if smoke and rows:
            import numpy as np
            expected['hover_metrics'] = dict(duration_s=rows[-1]['elapsed_s'],
                estimated_position_error_max_m=max(float(np.linalg.norm(np.asarray(r['position'])-[0,0,1.1])) for r in rows),
                estimated_speed_p95_m_s=float(np.percentile([np.linalg.norm(r['velocity']) for r in rows], 95)),
                truth_speed_p95_m_s=float(np.percentile([np.linalg.norm(r['truth_velocity_ned']) for r in rows], 95)),
                truth_altitude_range_m=float(np.ptp([r['truth_altitude_above_start_m'] for r in rows])))
    else:
        policy = summary.get('tracking_policy')
        _require(isinstance(policy, dict) and policy.get('unknown_identity_counts_as_correct') is False,
                 'Explicit conservative tracking policy required')
        threshold = policy.get('min_correct_identity_duration_fraction')
        _require(threshold == .8 and policy.get('require_recovery') is config['recovery'],
                 'Tracking policy differs from the declared frozen threshold/recovery protocol')
        start, end = summary.get('mission_start_s'), summary.get('mission_end_s')
        _require(_finite(start) and _finite(end) and 0 <= start < end, 'Exact mission scoring window required')
        if any(e['event'] == 'mission_window_complete' for e in events):
            _require(_finite(config.get('duration_s')) and end-start >= config['duration_s'],
                     'Completed mission is shorter than its declared duration')
        expected = mission_score(rows, observations, events, mission_start_s=start, mission_end_s=end,
                                 require_recovery=config['recovery'], min_tracking_fraction=threshold)
        if config['faults']:
            faults = dict(actual_delay_injected=any(e['event'] == 'mission_delay_injected' for e in events),
                delayed_output_rejected=any(o.get('command', {}).get('reason') == 'stale_perception_result'
                    and o['packet']['completed_s']-o['packet']['capture_time_s'] >= 1 for o in observations))
            expected['checks'].update(faults)
            expected['safety_passed'] = expected['safety_passed'] and all(faults.values())
            expected['tracking_success'] = expected['tracking_success'] and all(faults.values())
            expected['passed'] = expected['passed'] and all(faults.values())
    expected['passed'] = expected['passed'] and summary['error'] is None
    _compare(summary, expected)
    return dict(verified=True, passed=summary['passed'], actual_px4_receipt_verified=actual,
        benchmark_verified=not smoke, source_count=sources, observations=len(observations),
        control_ticks=len(rows), px4_sensor_profile=summary['px4_sensor_profile'],
        comparison_only_source_change=source_info['comparison_only_source_change'],
        mission_policy_compatibility_checked=True,
        score_recomputed=True, scope=SCOPE)


def verify_batch(folder):
    """Audit every attempted case and independently reconstruct batch aggregates."""
    folder = Path(folder).resolve()
    plan = _read(folder, 'plan.json', dict)
    progress = _read(folder, 'progresssummary.json', dict)
    _require(plan.get('schema_version') == progress.get('schema_version') == 1,
             'Unsupported batch schema')
    policy = plan.get('stopping_policy', {})
    clock_error = 'SitlError: no_fresh_preceding_source_clock_receipt'
    _require(policy.get('consecutive_failed_takeoffs') == 2
             and policy.get('consecutive_source_clock_failures') == 2
             and policy.get('source_clock_error') == clock_error,
             'Batch lacks the frozen consecutive-failure stopping policy')
    declared, cases = plan.get('cases'), progress.get('cases')
    _require(isinstance(declared, list) and isinstance(cases, list) and len(declared) == len(cases),
             'Batch must retain every planned case')
    _require(all(isinstance(c, dict) for c in declared+cases), 'Malformed batch cases')
    _require(plan.get('methods') == ['neural', 'direct', 'filtered']
             and plan.get('scenes') == ['walk', 'crossing', 'occlusion']
             and type(plan.get('repeats')) is int and 1 <= plan['repeats'] <= 3,
             'Unrecognized declared comparison matrix')
    expected_matrix = {(m, r, s) for m in plan['methods'] for r in range(plan['repeats']) for s in plan['scenes']}
    observed_matrix = {(c.get('method'), c.get('repeat'), c.get('config', {}).get('trajectory')) for c in declared}
    _require(len(declared) == len(expected_matrix) and observed_matrix == expected_matrix,
             'Batch omitted/duplicated a declared method, scene or repeat')
    for repeat in range(plan['repeats']):
        for scene in plan['scenes']:
            configs = [c['config'] for c in declared if c['repeat'] == repeat
                       and c['config']['trajectory'] == scene]
            _require(all(c == configs[0] for c in configs), 'Paired methods declare different configurations/noise')
            _require(type(configs[0].get('low_noise_sensors')) is bool, 'Noise mode must be explicit boolean')
    ids = [c.get('case_id') for c in declared]
    _require(len(set(ids)) == len(ids) and all(isinstance(i, str) and i for i in ids), 'Invalid/duplicate case IDs')
    attempted, verified_runs, missing, skipped = [], [], [], []
    for declaration, case in zip(declared, cases):
        case_id = declaration['case_id']
        run_folder = _path(folder, case_id)
        _require(case.get('case_id') == case_id and case.get('method') == declaration.get('method')
                 and case.get('repeat') == declaration.get('repeat'), 'Case identity/order mismatch')
        status = case.get('status')
        _require(status in ('passed', 'failed', 'planned', 'running', 'skipped'), 'Unknown case status')
        if status not in ('passed', 'failed'):
            _require(not any(_exists(run_folder, name) for name in
                ('summary.json', 'batch_outcome.json', 'batch_failure.json')),
                'Unattempted case hides original attempt/result evidence: '+case_id)
        if status == 'skipped':
            skipped.append(case_id)
            _require(case.get('actual_px4') is False, 'Skipped case cannot claim PX4 execution')
            continue
        if status not in ('passed', 'failed'):
            missing.append(case_id)
            continue
        attempted.append(case)
        if not _exists(run_folder, 'summary.json'):
            failure = _read(run_folder, 'batch_failure.json', dict)
            error = failure.get('error')
            _require(status == 'failed' and case.get('actual_px4') is False
                     and isinstance(error, str) and error and error == case.get('runner_exception')
                     and failure.get('actual_px4') is False, 'Unsupported missing-run claim')
            _require(case.get('runner_passed') is False
                     and case.get('stable_takeoff') is False and case.get('landed_disarmed') is False,
                     'Startup exception cannot invent mission metrics or flight verdicts')
            _require(case.get('source_clock_failure') is False,
                     'Startup exception cannot claim a native source-clock failure')
            for name in ('events.json', 'control.json', 'observations.json', 'takeoff.json'):
                _require(not _exists(run_folder, name) or not _read(run_folder, name, list),
                         'Missing final summary with partial flight receipts is unsupported')
            config = declaration['config']
            provenance_present = _exists(run_folder, 'provenance.json')
            provenance = _read(run_folder, 'provenance.json', dict) if provenance_present else {}
            mission = provenance.get('mission_config', {})
            declared_config = dict(method=declaration['method'], **{key: config[key] for key in
                ('trajectory', 'target_speed', 'duration_s', 'faults', 'recovery', 'scenario_phase', 'noise_seed')})
            config_matches = (bool(provenance) and all(mission.get(k) == v for k, v in declared_config.items())
                and provenance.get('estimator_profile') == config['estimator_profile']
                and provenance.get('estimator_parameters') == config['estimator_parameters']
                and provenance.get('px4_sensor_profile') == config['sensor_profile']
                and provenance.get('selection', {}).get('track_id') == config['selected_track'])
            source_matches = bool(provenance) and provenance.get('source_sha256') == plan['source_sha256']
            if source_matches:
                _sources(run_folder, provenance, check_policy=False)
            comparable = dict(config)
            if not config_matches or not source_matches:
                comparable['unverified_configuration_case'] = case_id
            _require(case.get('declared_config') == config and case.get('config') == comparable,
                     'Startup exception configuration differs from plan')
            _require(case.get('protocol_checks') == dict(declared_configuration_matches=config_matches,
                frozen_sources_match=source_matches, actual_px4_verified=False, declared_delay_submitted=False),
                'Startup protocol claims exceed available receipts')
            wrapped = _batch_wrapped_stop(case,
                dict(passed=False, actual_px4=False, error=error, metrics={}), [],
                provenance_present=provenance_present,
                configuration_matches=config_matches, source_matches=source_matches)
            _require(case.get('summary') == wrapped,
                     'Startup exception cannot invent mission metrics or flight verdicts')
            _require(_read(run_folder, 'batch_outcome.json', dict) == case,
                     'Startup exception outcome/progress mismatch')
            verified_runs.append(dict(case_id=case_id, verified=True, passed=False,
                actual_px4_receipt_verified=False, scope='Preserved startup/runner exception; no flight evidence'))
            continue
        verification = verify_run(run_folder)
        verified_runs.append(dict(case_id=case_id, **verification))
        provenance = _read(run_folder, 'provenance.json', dict)
        saved = _read(run_folder, 'summary.json', dict)
        config = declaration['config']
        _require(provenance['source_sha256'] == plan.get('source_sha256'), 'Batch runtime source changed')
        _require(case.get('declared_config') == config and case.get('config') == config, 'Case configuration changed')
        for key in ('trajectory', 'target_speed', 'duration_s', 'faults', 'recovery', 'scenario_phase', 'noise_seed'):
            _require(provenance['mission_config'].get(key) == config.get(key), 'Run/plan mismatch: '+key)
        _require(provenance['mission_config'].get('method') == declaration['method']
            and provenance['estimator_profile'] == config.get('estimator_profile')
            and provenance.get('estimator_parameters') == config.get('estimator_parameters')
            and provenance['px4_sensor_profile'] == config.get('sensor_profile')
            and provenance.get('selection', {}).get('track_id') == config.get('selected_track'),
            'Run/plan method, estimator, selection or sensor mismatch')
        _require(config.get('sensor_profile') == ('1% stock GPS/baro/mag/IMU noise'
            if config.get('low_noise_sensors') is True else 'stock sensor noise'), 'Noise flag/profile mismatch')
        events = _read(run_folder, 'events.json', list)
        delays = [e for e in events if e['event'] == 'mission_delay_injected']
        checks = dict(declared_configuration_matches=True, frozen_sources_match=True,
            actual_px4_verified=verification['actual_px4_receipt_verified'],
            declared_delay_submitted=len(delays) == 1 and delays[0].get('delay_s') == 1.)
        _require(case.get('protocol_checks') == checks, 'Batch protocol checks do not match receipts')
        _require(case.get('runner_passed') is saved['passed'] and case.get('actual_px4') is saved['actual_px4'],
                 'Batch replaced original verdict/native execution claim')
        source_clock_failure = bool(verification['actual_px4_receipt_verified']
            and isinstance(saved.get('error'), str) and clock_error in saved['error'])
        _require(case.get('source_clock_failure') is source_clock_failure,
                 'Source-clock failure classification differs from original receipts')
        names = {event['event'] for event in events}
        _require(case.get('stable_takeoff') is ('stable_takeoff' in names)
                 and case.get('landed_disarmed') is ('landed_disarmed' in names),
                 'Batch flight milestones differ from original receipts')
        wrapped = dict(saved, passed=saved['passed'] and case.get('runner_exception') is None
                       and not case.get('evidence_errors') and all(checks.values()))
        if case.get('runner_exception') is not None:
            failure = _read(run_folder, 'batch_failure.json', dict)
            _require(failure.get('error') == case['runner_exception'], 'Runner exception mismatch')
            wrapped['error'] = case['runner_exception']
        wrapped = _batch_wrapped_stop(case, wrapped, events, provenance_present=True,
            configuration_matches=True, source_matches=True)
        _require(case.get('summary') == wrapped and (status == 'passed') == wrapped['passed'],
                 'Batch summary changed the original result')
        _require(_read(run_folder, 'batch_outcome.json', dict) == case, 'Batch outcome/progress mismatch')
    clock_failures = takeoff_failures = 0
    derived_stop = None
    for index, case in enumerate(cases):
        if case['status'] not in ('passed', 'failed'):
            continue
        _require(derived_stop is None, 'Batch continued after a mandatory stop')
        clock_failures = clock_failures+1 if case['source_clock_failure'] else 0
        takeoff_failures = 0 if case['stable_takeoff'] else takeoff_failures+1
        derived_stop = case['stop_reason']
        if derived_stop is None and takeoff_failures >= 2:
            derived_stop = 'two_consecutive_failed_takeoffs'
        if derived_stop is None and clock_failures >= 2:
            derived_stop = 'two_consecutive_source_clock_failures'
        if derived_stop is not None:
            _require(progress.get('stopped_reason') == derived_stop,
                     'Batch stopping reason differs from original receipts')
            _require(all(c['status'] == 'skipped' and c.get('reason') == derived_stop
                         for c in cases[index+1:]), 'Mandatory stop requires an explicitly skipped suffix')
    if progress.get('stopped_reason') in ('two_consecutive_source_clock_failures',
                                         'two_consecutive_failed_takeoffs'):
        _require(derived_stop == progress['stopped_reason'],
                 'Consecutive-failure stop lacks the original qualifying attempts')
    counts = dict(declared_cases=len(cases), attempted_cases=len(attempted),
        passed_cases=sum(c['status'] == 'passed' for c in cases),
        failed_cases=sum(c['status'] == 'failed' for c in cases), skipped_cases=len(skipped),
        actual_px4_cases=sum(v['actual_px4_receipt_verified'] for v in verified_runs),
        completed_all_declared_cases=len(attempted) == len(cases),
        finished=all(c['status'] in ('passed', 'failed', 'skipped') for c in cases),
        passed=bool(cases) and all(c['status'] == 'passed' for c in cases))
    _compare(progress, counts)
    unknown = [c['summary'].get('metrics', {}).get('identity_unknown_count') for c in attempted
               if isinstance(c.get('summary', {}).get('metrics'), dict)]
    unknown = [v for v in unknown if type(v) is int and v >= 0]
    _require(progress.get('unknown_identity') == dict(count=sum(unknown) if unknown else None,
        cases_reporting_count=len(unknown), attempted_cases_without_count=len(attempted)-len(unknown),
        missing_or_unknown_is_correct=False), 'Unknown identity aggregate changed')
    _require(progress.get('comparison') == paired_summary(attempted), 'Paired comparison differs from raw case results')
    return dict(verified=True, **counts, missing_cases=missing, skipped_case_ids=skipped,
                verified_runs=verified_runs, scope=SCOPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    try:
        result = verify_batch(args.path) if _exists(args.path, 'plan.json') else verify_run(args.path)
    except (VerificationError, KeyError, TypeError, OSError, ValueError) as exc:
        print(json.dumps(dict(verified=False, error=str(exc), scope=SCOPE), indent=2))
        raise SystemExit(1)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
