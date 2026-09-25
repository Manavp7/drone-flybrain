"""Run a frozen, serialized PX4 mission comparison; preserve every failure.

This starts actual local PX4 through experiments.px4_follow.run. An existing
output directory is always refused: there is no implicit resume or retry.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.px4_evaluation import paired_summary

METHODS = ('neural', 'direct', 'filtered')
SCENES = ('walk', 'crossing', 'occlusion')
STORAGE_BUDGET_BYTES = 512 * 1024**2
RUN_ADMISSION_BYTES = 64 * 1024**2
METADATA_RESERVE_BYTES = 1024**2
SOURCE_CLOCK_ERROR = 'SitlError: no_fresh_preceding_source_clock_receipt'


def runtime_hashes():
    """Match the actual runner's frozen files; it owns per-run source copies."""
    files = [path for directory in ('experiments', 'perception', 'flybrain_sim')
             for path in (ROOT/directory).glob('*.py')]
    files += [ROOT/'scripts/build_px4_sih.py', ROOT/'integrations/px4/sih.px4board',
              ROOT/'models/flyvis_0000_000.manifest.json']
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}


def estimator_profiles():
    from flybrain_sim.px4_sih import ESTIMATOR_PROFILES
    return deepcopy(ESTIMATOR_PROFILES)


def make_plan(*, repeats=2, duration_s=60., low_noise_sensors=False,
              estimator_profile='stock', source_hashes=None):
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError('Repeats must be an integer from 1 to 3')
    if (isinstance(duration_s, bool) or not isinstance(duration_s, (int, float))
            or not math.isfinite(duration_s) or not 30 <= duration_s <= 120):
        raise ValueError('Mission duration must be between 30 and 120 seconds')
    profiles = estimator_profiles()
    if estimator_profile not in profiles or type(low_noise_sensors) is not bool:
        raise ValueError('An explicit supported estimator and boolean noise profile are required')
    sources = runtime_hashes() if source_hashes is None else deepcopy(source_hashes)
    if (not isinstance(sources, dict) or not sources or any(
            not isinstance(key, str) or not isinstance(value, str)
            or len(value) != 64 or any(c not in '0123456789abcdef' for c in value)
            for key, value in sources.items())):
        raise ValueError('Frozen runtime source hashes are required')
    common = dict(duration_s=float(duration_s), target_speed=.08, faults=True,
        fault_delay_s=1., fault_time_fraction=.5, recovery=False, selected_track=1,
        low_noise_sensors=low_noise_sensors, estimator_profile=estimator_profile,
        estimator_parameters=profiles[estimator_profile],
        sensor_profile='1% stock GPS/baro/mag/IMU noise' if low_noise_sensors else 'stock sensor noise',
        scenario_phase='starts at stable takeoff using SIH source time',
        noise_seed='upstream SIH srand(1234); asynchronous sensor scheduling may vary')
    cases = []
    for repeat in range(repeats):
        order = METHODS if repeat % 2 == 0 else tuple(reversed(METHODS))
        for scene in SCENES:
            for method in order:
                cases.append(dict(case_id=f'r{repeat+1:02d}-{scene}-{method}',
                    method=method, repeat=repeat, config=dict(deepcopy(common), trajectory=scene)))
    return dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
        repeats=repeats, methods=list(METHODS), scenes=list(SCENES), cases=cases,
        source_sha256=sources,
        batch_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        storage=dict(budget_bytes=STORAGE_BUDGET_BYTES, next_run_reserve_bytes=RUN_ADMISSION_BYTES,
            metadata_reserve_bytes=METADATA_RESERVE_BYTES,
            policy='Stop admission; never delete evidence. Recorder is capped at 32 MiB per run. '
                'The 64 MiB admission reserve also covers source copies, logs and JSON. '
                'Those non-video files are not hard-capped during an admitted run.'),
        stopping_policy=dict(consecutive_failed_takeoffs=2, consecutive_source_clock_failures=2,
            source_clock_error=SOURCE_CLOCK_ERROR, landing_unconfirmed='stop',
            resource_cleanup_error='stop', runtime_source_change='stop', overwrite_or_resume='refuse'),
        interpretation='Predeclared paired configurations, not identical camera paths or guaranteed '
            'identical noise samples. Unknown identity is not correct identity. No automatic superiority claim.')


def _json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()


def _atomic_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    with temporary.open('xb') as stream:
        stream.write(_json_bytes(value))
    temporary.replace(path)


def directory_bytes(folder):
    """Logical bytes owned by this new batch; do not follow external symlinks."""
    total = 0
    with os.scandir(folder) as entries:
        for entry in entries:
            total += (directory_bytes(entry.path) if entry.is_dir(follow_symlinks=False)
                      else entry.stat(follow_symlinks=False).st_size)
    return total


def _read_json(path):
    if not path.exists():
        return None
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError('Evidence receipt exceeds its bounded size: '+path.name)
    return json.loads(path.read_text(), parse_constant=lambda value:
                     (_ for _ in ()).throw(ValueError('Nonfinite JSON evidence')))


def _native_run(folder, **kwargs):
    # Delay native/runtime imports until the complete plan is saved.
    from experiments.px4_follow import run
    return run(folder, **kwargs)


def _run_arguments(case):
    config = case['config']
    return dict(smoke=False, selected_track=config['selected_track'],
        low_noise_sensors=config['low_noise_sensors'], estimator_profile=config['estimator_profile'],
        method=case['method'], trajectory=config['trajectory'], target_speed=config['target_speed'],
        duration_s=config['duration_s'], mission=True, faults=True, recovery=False)


def _receipt(case, folder, returned, exception, plan):
    """Require original receipts before labeling an attempted case actual PX4."""
    errors = []
    try:
        events = _read_json(folder/'events.json')
        provenance = _read_json(folder/'provenance.json')
        saved = _read_json(folder/'summary.json')
    except (OSError, ValueError, TypeError) as exc:
        events = provenance = saved = None
        errors.append('unreadable_runner_evidence: '+str(exc))
    valid_events = isinstance(events, list) and all(isinstance(event, dict) for event in events)
    if not valid_events:
        events = []
        errors.append('runner_events_unavailable')
    summary = deepcopy(returned if isinstance(returned, dict) else saved if isinstance(saved, dict) else {})
    if not isinstance(returned, dict) and exception is None:
        errors.append('runner_summary_unavailable')
    if isinstance(returned, dict) and saved != returned:
        errors.append('saved_summary_mismatch')
    config, mission = case['config'], provenance.get('mission_config', {}) if isinstance(provenance, dict) else {}
    selection = provenance.get('selection', {}) if isinstance(provenance, dict) else {}
    declared = dict(method=case['method'], **{key: config[key] for key in
        ('trajectory', 'target_speed', 'duration_s', 'faults', 'recovery', 'scenario_phase', 'noise_seed')})
    provenance_matches = (isinstance(provenance, dict) and isinstance(mission, dict)
        and all(mission.get(key) == value for key, value in declared.items())
        and provenance.get('estimator_profile') == config['estimator_profile']
        and provenance.get('estimator_parameters') == config['estimator_parameters']
        and provenance.get('px4_sensor_profile') == config['sensor_profile']
        and isinstance(selection, dict) and selection.get('track_id') == config['selected_track'])
    source_matches = isinstance(provenance, dict) and provenance.get('source_sha256') == plan['source_sha256']
    owned = [event for event in events if event.get('event') == 'owned_px4_verified']
    actual = (summary.get('actual_px4') is True and len(owned) == 1
        and isinstance(provenance, dict) and isinstance(summary.get('session'), str)
        and bool(summary['session']) and summary['session'] == provenance.get('session'))
    delays = [event for event in events if event.get('event') == 'mission_delay_injected']
    checks = dict(declared_configuration_matches=provenance_matches,
        frozen_sources_match=source_matches, actual_px4_verified=actual,
        declared_delay_submitted=len(delays) == 1 and delays[0].get('delay_s') == 1.)
    runner_passed = summary.get('passed') is True
    if not isinstance(summary.get('metrics', {}), dict):
        errors.append('runner_metrics_invalid')
        summary['metrics'] = {}
    summary.update(passed=runner_passed and exception is None and not errors and all(checks.values()),
                   actual_px4=actual)
    if exception is not None:
        summary['error'] = exception
    summary.setdefault('passed', False)
    summary.setdefault('metrics', {})
    names = {event.get('event') for event in events}
    stop = None
    if ('cleanup_failed' in names or 'cleanup failed' in str(summary.get('error', '')).lower()):
        stop = 'resource_cleanup_error'
    elif ('landing_unconfirmed' in names or
          bool(names & {'armed_offboard', 'stable_takeoff', 'land_requested'}) and 'landed_disarmed' not in names):
        stop = 'landing_unconfirmed'
    elif ((exception and exception.startswith(('KeyboardInterrupt:', 'SystemExit:'))) or str(
            summary.get('error', '')).startswith(('KeyboardInterrupt:', 'SystemExit:'))):
        stop = 'interrupted'
    elif errors and exception is None:
        stop = 'runner_evidence_unavailable'
    elif (not provenance_matches or not source_matches) and isinstance(provenance, dict):
        stop = 'runner_configuration_or_source_mismatch'
    if stop is not None:
        summary.update(passed=False, batch_rejection_reason=stop)
    comparable_config = deepcopy(config)
    if not provenance_matches or not source_matches:
        # An unverified/mismatched run remains in the comparison, but cannot be
        # grouped as if it had actually used the other methods' configuration.
        comparable_config['unverified_configuration_case'] = case['case_id']
    return dict(case_id=case['case_id'], method=case['method'], repeat=case['repeat'],
        config=comparable_config, declared_config=deepcopy(config),
        status='passed' if summary['passed'] else 'failed',
        actual_px4=actual, runner_passed=runner_passed, summary=summary,
        protocol_checks=checks, evidence_errors=errors, runner_exception=exception,
        source_clock_failure=bool(actual and isinstance(saved, dict)
            and isinstance(saved.get('error'), str) and SOURCE_CLOCK_ERROR in saved['error']),
        stable_takeoff='stable_takeoff' in names, landed_disarmed='landed_disarmed' in names,
        stop_reason=stop, folder=case['case_id'])


def _progress(plan, cases, stopped, measured_bytes):
    attempted = [case for case in cases if case['status'] in ('passed', 'failed')]
    measured_identity = [case['summary'].get('metrics', {}).get('identity_unknown_count')
                         for case in attempted if isinstance(case['summary'].get('metrics'), dict)]
    measured_identity = [value for value in measured_identity if type(value) is int and value >= 0]
    return dict(schema_version=1, updated_utc=datetime.now(timezone.utc).isoformat(),
        plan='plan.json', declared_cases=len(plan['cases']),
        finished=all(case['status'] in ('passed', 'failed', 'skipped') for case in cases),
        completed_all_declared_cases=len(attempted) == len(cases),
        passed=bool(cases) and all(case['status'] == 'passed' for case in cases),
        stopped_reason=stopped, attempted_cases=len(attempted),
        passed_cases=sum(case['status'] == 'passed' for case in cases),
        failed_cases=sum(case['status'] == 'failed' for case in cases),
        skipped_cases=sum(case['status'] == 'skipped' for case in cases),
        actual_px4_cases=sum(case['actual_px4'] is True for case in attempted),
        unknown_identity=dict(count=sum(measured_identity) if measured_identity else None,
            cases_reporting_count=len(measured_identity), attempted_cases_without_count=len(attempted)-len(measured_identity),
            missing_or_unknown_is_correct=False),
        storage=dict(plan['storage'], last_measured_bytes=measured_bytes,
                     over_budget=measured_bytes > STORAGE_BUDGET_BYTES),
        cases=deepcopy(cases), comparison=paired_summary(attempted))


def run_batch(output, *, repeats=2, duration_s=60., low_noise_sensors=False,
              estimator_profile='stock', runner=None, source_reader=runtime_hashes,
              byte_counter=directory_bytes):
    """Serialize each declared case exactly once; injected runners are for tests."""
    plan = make_plan(repeats=repeats, duration_s=duration_s, low_noise_sensors=low_noise_sensors,
                     estimator_profile=estimator_profile, source_hashes=source_reader())
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output/'plan.json').open('xb') as stream:
        stream.write(_json_bytes(plan))
    cases = [dict(deepcopy(case), status='planned') for case in plan['cases']]
    runner = _native_run if runner is None else runner
    stopped, consecutive_failed_takeoffs, consecutive_source_clock_failures = None, 0, 0

    def publish():
        result = _progress(plan, cases, stopped, byte_counter(output))
        _atomic_json(output/'progresssummary.json', result)
        return result

    publish()
    for index, case in enumerate(plan['cases']):
        if source_reader() != plan['source_sha256']:
            stopped = 'runtime_source_changed'
        elif byte_counter(output)+RUN_ADMISSION_BYTES+METADATA_RESERVE_BYTES > STORAGE_BUDGET_BYTES:
            stopped = 'storage_admission_budget_exhausted'
        elif (output/case['case_id']).exists():
            stopped = 'run_folder_already_exists'
        if stopped:
            break
        cases[index]['status'] = 'running'
        publish()
        folder = output/case['case_id']
        returned = exception = None
        try:
            returned = runner(folder, **_run_arguments(case))
        except BaseException as exc:
            exception = f'{type(exc).__name__}: {exc}'
            # Source-copy/model/startup errors can occur before the run creates
            # a result. Preserve that attempt without inventing flight evidence.
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'batch_failure.json').open('xb') as stream:
                stream.write(_json_bytes(dict(error=exception, actual_px4=False,
                    interpretation='Runner raised; native execution is established only by original run receipts.')))
        if not folder.exists():
            folder.mkdir()
            exception = exception or 'RuntimeError: Runner did not create its output directory'
        receipt = _receipt(case, folder, returned, exception, plan)
        with (folder/'batch_outcome.json').open('xb') as stream:
            stream.write(_json_bytes(receipt))
        cases[index] = receipt
        consecutive_failed_takeoffs = 0 if receipt['stable_takeoff'] else consecutive_failed_takeoffs+1
        consecutive_source_clock_failures = (consecutive_source_clock_failures+1
            if receipt['source_clock_failure'] else 0)
        stopped = receipt['stop_reason']
        if stopped is None and consecutive_failed_takeoffs >= 2:
            stopped = 'two_consecutive_failed_takeoffs'
        if stopped is None and consecutive_source_clock_failures >= 2:
            stopped = 'two_consecutive_source_clock_failures'
        if stopped is None and byte_counter(output)+METADATA_RESERVE_BYTES > STORAGE_BUDGET_BYTES:
            stopped = 'storage_budget_reached_after_run'
        publish()
        if stopped:
            break
    if stopped:
        for case in cases:
            if case['status'] == 'planned':
                case.update(status='skipped', reason=stopped, actual_px4=False)
    return publish()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New batch directory; existing paths are refused')
    parser.add_argument('--repeats', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--duration', type=float, default=60., help='Per-mission seconds, from 30 to 120')
    parser.add_argument('--estimator-profile', choices=estimator_profiles(), default='stock')
    parser.add_argument('--low-noise-sensors', action='store_true',
                        help='Explicit diagnostic: 1%% stock noise, preserved identically across all methods')
    args = parser.parse_args()
    if not math.isfinite(args.duration) or not 30 <= args.duration <= 120:
        parser.error('Duration must be between 30 and 120 seconds')
    summary = run_batch(args.output, repeats=args.repeats, duration_s=args.duration,
                        estimator_profile=args.estimator_profile, low_noise_sensors=args.low_noise_sensors)
    print(json.dumps({key: summary[key] for key in ('passed', 'attempted_cases', 'passed_cases',
        'failed_cases', 'skipped_cases', 'actual_px4_cases', 'stopped_reason')}, indent=2))
    print('Full declared plan and preserved results:', args.output.resolve())
    raise SystemExit(0 if summary['passed'] else 1)


if __name__ == '__main__':
    main()
