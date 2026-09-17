"""Reproducible local Studio development checks, separate from historical scores.

The harness selects one current camera detection nearest image centre once.
Known fixture geometry is used only by the evaluator, never to select or steer.
The definition is saved before any new run. No models or recordings are copied.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from experiments.flight_contracts import ROOT, SAFETY_MARGIN_M
from experiments.mantis_session import motion_brake_scale


def definition():
    return dict(schema='mantis-studio-remediation-v1',
        scope='Controlled integration checks; not general navigation or identity validation',
        selection='One explicit current detection nearest image centre; no automatic reselection',
        cases=[dict(name='detour_neural_a', scenario='detour', duration_s=25., method='mantis_neural', detours=True),
               dict(name='detour_neural_b', scenario='detour', duration_s=25., method='mantis_neural', detours=True),
               dict(name='detour_direct', scenario='detour', duration_s=25., method='direct_yolo', detours=True),
               dict(name='motion_observe', scenario='stationary', duration_s=10., motion_mode='observe'),
               dict(name='motion_brake', scenario='stationary', duration_s=10., motion_mode='brake')],
        thresholds=dict(detour_center_past_x_m=2.1, resumed_follow_translation_m=.15,
                        resumed_follow_observations=3, minimum_hull_clearance_m=SAFETY_MARGIN_M,
                        minimum_fresh_motion_observations=10, steady_motion_fresh_fraction=.8),
        limitations=['A neural accuracy or control advantage is not inferred from execution or braking.',
                     'Timing is measured on the current machine and can vary.'])


def score(receipt, rows, config, thresholds=None):
    thresholds = thresholds or definition()['thresholds']
    summary = receipt['summary']
    stats = summary['statistics']
    sequences = [r['frame']['sequence'] for r in rows]
    captures = [r['frame']['capture_time_s'] for r in rows]
    ordered = bool(rows) and all(b > a for a,b in zip(sequences,sequences[1:])) and all(
        b > a for a,b in zip(captures,captures[1:]))
    def selected(row, track):
        frame, selection, guidance = row['frame'], row.get('selection', {}), row['guidance']
        return bool(track is not None and selection.get('track_id') == track
            and selection.get('held') is False and selection.get('reason') == 'observed'
            and selection.get('sequence') == frame['sequence']
            and selection.get('capture_time_s') == frame['capture_time_s']
            and guidance.get('track_id') == track and guidance.get('valid') is True
            and guidance.get('capture_time_s') == frame['capture_time_s']
            and 0 <= row['completed_time_s']-frame['capture_time_s'] <= .65
            and row['evaluation']['wrong_person'] is False
            and row['evaluation'].get('actor_id') == stats.get('selected_actor_reference')
            and stats.get('selected_actor_reference') is not None)
    gates = dict(ordered_observations=ordered,
                 completed=receipt['status'] == 'completed' and summary.get('failure') is None,
                 contacts_absent=stats['contacts'] == 0,
                 no_wrong_person=stats['wrong_person_observations'] == 0,
                 selected_observations=stats['evaluated_selected_observations'] >= 10,
                 actual_models=stats['actual_yolo_calls'] >= 10 and stats['actual_flyvis_observations'] >= 10)
    metrics = {}
    if config['scenario'] == 'detour':
        events = stats.get('detour_completion_events', [])
        resumed = []
        first = events[0] if events else None
        authority = False
        if first:
            sources = [r for r in rows if r['frame'].get('sequence') == first['command_sequence']
                       and abs(r['frame']['capture_time_s']-first['source_capture_time_s']) < 1e-8]
            source = sources[0] if len(sources) == 1 else None
            authority = bool(source and selected(source, first.get('selected_track'))
                and source['completed_time_s'] <= first['sim_s'] < first['source_capture_time_s']+.9)
            resumed = [r for r in rows if r['frame']['capture_time_s'] > first['sim_s']
                       and r['navigation']['reason'] == 'following'
                       and r['safety']['forward_speed'] > .01
                       and selected(r, first.get('selected_track'))]
        displacement = max((float(np.linalg.norm(np.array(r['position'][:2])-np.array(first['position'][:2])))
                            for r in resumed), default=0.)
        max_x = max((r['position'][0] for r in resumed), default=0.)
        clearance = stats.get('minimum_obstacle_hull_clearance_m')
        gates.update(detour_recorded=bool(first and stats['detours_completed'] > 0),
            center_passed_obstacle=max_x > thresholds['detour_center_past_x_m'],
            clearance=clearance is not None and clearance >= thresholds['minimum_hull_clearance_m']-1e-6,
            following_resumed=len(resumed) >= thresholds['resumed_follow_observations']
                and displacement >= thresholds['resumed_follow_translation_m'],
            observed_authority=authority)
        metrics.update(maximum_center_x_m=max_x, minimum_hull_clearance_m=clearance,
                       resumed_observations=len(resumed), resumed_translation_m=displacement,
                       detour_completion_events=events)
    else:
        observations = [r for r in rows if r.get('motion')]
        fresh = [r for r in observations if motion_brake_scale(r['motion'], r['completed_time_s']) > 0]
        first_capture = observations[0]['motion']['capture_time_s'] if observations else 0.
        steady = [r for r in observations if r['motion']['capture_time_s'] >= first_capture+1.]
        fraction = sum(motion_brake_scale(r['motion'], r['completed_time_s']) > 0 for r in steady)/max(1,len(steady))
        gates.update(fresh_motion=len(fresh) >= thresholds['minimum_fresh_motion_observations'],
                     steady_freshness=bool(steady) and fraction >= thresholds['steady_motion_fresh_fraction'])
        if config.get('motion_mode') == 'brake':
            pairs = stats.get('motion_brake_pairs', [])
            valid_pairs = []
            for pair in pairs:
                sources = [r for r in observations if r['motion']['capture_time_s'] == pair['motion_capture_time_s']
                           and r['motion']['available_time_s'] == pair['motion_available_time_s']]
                if (len(sources) == 1 and motion_brake_scale(sources[0]['motion'],pair['sim_s']) > 0
                        and 0 <= pair['sim_s']-pair['command_capture_time_s'] < .9
                        and 0 <= pair['sim_s']-pair['depth_capture_time_s'] <= .1+1e-9
                        and pair['depth_approved_unscaled_speed'] > pair['actual_forward_speed'] >= 0
                        and 0 < pair['scale'] < 1):
                    valid_pairs.append(pair)
            gates.update(valid_motion_changes_request=stats.get('motion_valid_speed_reductions', 0) > 0,
                         independent_depth_approved_reduction=bool(valid_pairs),
                         translation_remains_possible=stats.get('positive_forward_ticks', 0) > 0)
        metrics.update(motion_observations=len(observations), fresh_motion_observations=len(fresh),
                       steady_fresh_fraction=fraction, gap_resets=stats.get('motion_gap_resets', 0),
                       valid_speed_reduction_ticks=stats.get('motion_valid_speed_reductions', 0),
                       latency_s=[r['inference_wall_s'] for r in observations])
    return dict(passed=all(gates.values()), gates=gates, metrics=metrics)


def run_checks(output, base_url, case_names=None, runs_root=ROOT/'results/studio'):
    parsed = urlsplit(base_url)
    if (parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost')
            or parsed.username or parsed.password or parsed.path not in ('', '/')
            or parsed.query or parsed.fragment):
        raise ValueError('Checks require a local loopback Studio URL')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    frozen = definition()
    cases = frozen['cases']
    if case_names:
        unknown = set(case_names)-{c['name'] for c in cases}
        if unknown:
            raise ValueError('Unknown check names: '+str(sorted(unknown)))
        cases = [c for c in cases if c['name'] in case_names]
    frozen['selected_cases'] = [c['name'] for c in cases]
    frozen['defined_at'] = datetime.now(timezone.utc).isoformat()
    frozen['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted((ROOT/'experiments').glob('mantis_*.py'))}
    (output/'definition.json').write_text(json.dumps(frozen,indent=2)+'\n')
    base_url = base_url.rstrip('/')
    page = urlopen(base_url, timeout=5).read().decode()
    token = re.search("const token='([^']+)'", page).group(1)

    def call(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {'Content-Type':'application/json', 'X-Mantis-Token':token}
        with urlopen(Request(base_url+path, data=data, headers=headers), timeout=10) as response:
            return json.load(response)

    outcomes = []
    for case in cases:
        if call('/api/state')['running']:
            raise RuntimeError('Another Studio session is active')
        config = dict(method='mantis_neural', follow_distance_m=2.5, max_recording_mb=16)
        config.update({k:v for k,v in case.items() if k != 'name'})
        snapshot = call('/api/start', config)
        identity, selected = snapshot['active_id'], False
        print(json.dumps(dict(case=case['name'], run_id=identity, event='started')),flush=True)
        try:
            deadline = time.monotonic()+300
            while time.monotonic() < deadline:
                snapshot = call('/api/state')
                if snapshot['active_id'] != identity:
                    raise RuntimeError('Another client changed the active run')
                state = snapshot['state']
                if not snapshot['running']:
                    break
                if state['phase'] == 'awaiting-selection' and not selected:
                    people = [p for p in state['selection']['people'] if p['selectable']]
                    width = state['frame']['width']
                    person = min(people, key=lambda p: abs(sum(p['bbox_xyxy'][::2])/2-width/2))
                    call('/api/control', dict(run_id=identity, selection=dict(
                        track_id=person['track_id'], sequence=state['selection']['sequence'])))
                    selected = True
                time.sleep(.2)
            else:
                raise TimeoutError('Development check exceeded five wall-clock minutes')
        except BaseException:
            snapshot = call('/api/state')
            if snapshot['active_id'] == identity and snapshot['running']:
                call('/api/control',dict(run_id=identity,operation='stop'))
            raise
        capture = Path(runs_root)/identity/'capture'
        receipt = json.loads((capture/'summary.json').read_text())
        rows = [json.loads(line)['telemetry'] for line in (capture/'telemetry.jsonl').read_text().splitlines()]
        result = dict(case=case['name'],run_id=identity,config=state['config'],
                      outcome=score(receipt,rows,state['config'],frozen['thresholds']),
                      summary=receipt, source_artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [capture/'summary.json',capture/'telemetry.jsonl',capture/'provenance.json']})
        (output/(case['name']+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        outcomes.append(result)
        print(json.dumps(dict(case=case['name'],run_id=identity,**result['outcome'])),flush=True)
    status = dict(passed=all(r['outcome']['passed'] for r in outcomes), cases=len(outcomes),
                  outcomes=[dict(case=r['case'],run_id=r['run_id'],**r['outcome']) for r in outcomes])
    (output/'summary.json').write_text(json.dumps(status,indent=2,allow_nan=False)+'\n')
    return status


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-url',default='http://127.0.0.1:8875')
    parser.add_argument('--case',action='append')
    args = parser.parse_args()
    result = run_checks(args.output,args.base_url,args.case)
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
