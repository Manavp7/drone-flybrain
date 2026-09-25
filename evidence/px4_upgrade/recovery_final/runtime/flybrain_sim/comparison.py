"""Compare frozen software releases on identical reduced-order scenarios."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import statistics

from .evaluation import aggregate, write_json

INVARIANT_FILES = ('contracts.py', 'physics.py', 'sensors.py', 'scenarios.py', 'geometry.py', 'runner.py')


def _load_run(directory: Path) -> tuple[dict, list[dict], list[dict]]:
    summary = json.loads((directory / 'summary.json').read_text())
    rows = [json.loads(line) for line in (directory / 'episodes.jsonl').read_text().splitlines() if line]
    if summary['benchmark']['variants'] != ['baseline']:
        raise ValueError('Version comparison requires exactly the baseline variant per release')
    if len(rows) != summary['benchmark']['episode_count']:
        raise ValueError('Raw episode count does not match summary')
    seeds = [row['seed'] for row in rows]
    if len(set(seeds)) != len(rows):
        raise ValueError('Duplicate scenario seeds')
    expected = set(range(summary['benchmark']['seed_start'], summary['benchmark']['seed_end_inclusive'] + 1))
    if set(seeds) != expected or len(rows) != summary['benchmark']['scenario_count']:
        raise ValueError('Scenario seed range is incomplete')
    if any(row['variant'] != 'baseline' for row in rows):
        raise ValueError('Unexpected controller variant in raw records')
    actual = aggregate(rows)
    for key in ('runs', 'mission_successes', 'collision_count', 'geofence_count', 'outcomes'):
        if summary['overall'][0][key] != actual[key]:
            raise ValueError(f'Aggregate mismatch: {key}')
    replays = json.loads((directory / 'replays.json').read_text())
    indexed = {row['seed']: row for row in rows}
    for replay in replays:
        result = replay['result']
        if result['seed'] not in indexed:
            raise ValueError('Replay seed missing from raw results')
        for key, value in indexed[result['seed']].items():
            if key != 'wall_seconds' and result[key] != value:
                raise ValueError(f'Replay differs from raw result: {key}')
    return summary, rows, replays


def compare_releases(first: Path, second: Path, first_source: Path, second_source: Path,
                     output: Path, integration_status: list[dict] | None = None) -> dict:
    """Require identical physics, faults and scoring; preserve every paired outcome."""
    invariant_hashes = {}
    for name in INVARIANT_FILES:
        left = (first_source / 'flybrain_sim' / name).read_bytes()
        right = (second_source / 'flybrain_sim' / name).read_bytes()
        if left != right:
            raise ValueError(f'Benchmark environment/scoring changed: {name}')
        invariant_hashes[name] = hashlib.sha256(left).hexdigest()
    a, arows, areplays = _load_run(first)
    b, brows, breplays = _load_run(second)
    for source, record in ((first_source, a), (second_source, b)):
        digest = hashlib.sha256()
        for path in sorted((source / 'flybrain_sim').glob('*.py')):
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
        if digest.hexdigest() != record['source_hash']:
            raise ValueError('Provided source does not match the evaluated release')
    left = {r['seed']: r for r in arows}
    right = {r['seed']: r for r in brows}
    if left.keys() != right.keys() or any(left[k]['category'] != right[k]['category'] for k in left):
        raise ValueError('Both releases must run identical seeds and categories')
    for key in ('categories', 'dt_seconds', 'maximum_simulated_seconds'):
        if a['benchmark'][key] != b['benchmark'][key]:
            raise ValueError(f'Benchmark settings differ: {key}')
    variants = ['v1_baseline', 'v2_baseline']
    rows = []
    replays = []
    for tag, source_rows, source_replays in zip(variants, (arows, brows), (areplays, breplays)):
        rows.extend([{**row, 'variant': tag, 'original_variant': row['variant']} for row in source_rows])
        for replay in source_replays:
            item = deepcopy(replay)
            item['result']['original_variant'] = item['result']['variant']
            item['result']['variant'] = tag
            item['id'] = f"{item['result']['seed']}-{tag}"
            replays.append(item)
    paired = Counter({'first_only_success': 0, 'second_only_success': 0, 'both_success': 0, 'both_failed': 0})
    energy_difference = []
    transitions = []
    for seed in sorted(left):
        old, new = left[seed], right[seed]
        p, q = old['mission_complete'], new['mission_complete']
        key = 'both_success' if p and q else 'first_only_success' if p else 'second_only_success' if q else 'both_failed'
        paired[key] += 1
        if p and q:
            energy_difference.append(new['energy_wh'] - old['energy_wh'])
        transitions.append({'seed': seed, 'category': old['category'], 'original_outcome': old['outcome'],
                            'revised_outcome': new['outcome'], 'original_success': p, 'revised_success': q})
    count = len(left)
    pair_result = dict(paired)
    pair_result.update({'paired_scenario_count': count,
                       'success_difference_percentage_points': 100 * (paired['second_only_success'] - paired['first_only_success']) / count,
                       'mean_energy_difference_on_joint_success_wh': statistics.fmean(energy_difference) if energy_difference else None})
    summary = {'schema_version': 1, 'software_version': b['software_version'], 'generated_at': b['generated_at'],
               'source_hash': dict(zip(variants, (a['source_hash'], b['source_hash']))),
               'invariant_environment_hashes': invariant_hashes,
               'comparison': {'labels': ['Original controller', 'Revised controller'],
                              'description': 'Frozen release comparison on identical generated scenarios.',
                              'note': 'Only conventional navigation changes are measured. Flyvis and PX4 are separate integration work and are not used in this benchmark.'},
               'benchmark': {**b['benchmark'], 'episode_count': 2 * count, 'variants': variants,
                             'paired': True, 'sampled_replay_count': len(replays),
                             'wall_seconds': a['benchmark']['wall_seconds'] + b['benchmark']['wall_seconds']},
               'overall': [aggregate([r for r in rows if r['variant'] == v]) for v in variants],
               'by_category': [{**aggregate([r for r in rows if r['variant'] == v and r['category'] == c]), 'category': c}
                               for c in a['benchmark']['categories'] for v in variants],
               'paired': pair_result, 'limitations': b['limitations'],
               'integration_status': integration_status or [],
               'model_status': {'navigation': {'uses_connectome': False, 'uses_camera_pixels': False,
                                               'description': 'Both navigation versions use geometric observations; research-model inference is separate.'}}}
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to replace comparison files in {output}')
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'summary.json', summary)
    write_json(output / 'replays.json', replays)
    write_json(output / 'transitions.json', transitions)
    with (output / 'episodes.jsonl').open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    with (output / 'episodes.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / 'manifest.json', {'source_hashes': summary['source_hash'],
               'invariant_environment_hashes': invariant_hashes, 'scenario_count': count,
               'episode_count': count * 2, 'source_results': [first.name, second.name],
               'labels': summary['comparison']['labels'], 'benchmark': summary['benchmark']})
    return summary
