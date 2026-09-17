"""Verify copied evidence and independently recompute vector errors; no models."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise SystemExit(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_scores(rows, metrics):
    for method in ('neural', 'conventional', 'zero'):
        squared = []
        for row in rows:
            prediction = ([0., 0.] if method == 'zero' else
                          row['response'][method]['nominal_velocity_px_s'])
            squared.append(sum((value-target)**2 for value, target in
                               zip(prediction, row['truth_velocity_px_s'])))
        rmse = math.sqrt(math.fsum(squared)/len(squared))
        require(math.isclose(rmse, metrics[method]['vector_rmse_px_s'], rel_tol=1e-12, abs_tol=1e-12),
                f'{method} vector RMSE differs from compact observations')
        require(metrics[method]['rows'] == len(rows), f'{method} scored row count differs')


def main():
    definition = json.loads((ROOT/'definition.json').read_text())
    provenance = json.loads((ROOT/'provenance.json').read_text())
    summary = json.loads((ROOT/'summary.json').read_text())
    for name, expected in summary['artifacts'].items():
        require(Path(name).name == name, 'Unexpected artifact path')
        require(digest(ROOT/name) == expected, f'Artifact digest mismatch: {name}')
    require(digest(ROOT/'definition.json') == provenance['definition_sha256'], 'Definition changed')
    rows = [json.loads(line) for line in (ROOT/'observations.jsonl').read_text().splitlines()]
    require(len(rows) == 156, 'Expected 156 observations')
    require(len(definition['cases']) == 12 and len(summary['cases']) == 12, 'Expected 12 cases')
    require(set(definition['train_seeds']).isdisjoint(definition['heldout_seeds']), 'Texture seeds overlap')
    require(all(count == 13 for count in Counter(row['case'] for row in rows).values()), 'Expected 13 observations per case')
    require(all(row['response']['response_time_s'] > row['response']['stimulus_time_s'] >=
                row['response']['capture_time_s']-1e-10 for row in rows), 'Noncausal response timestamps')
    scored = [row for row in rows if row['response']['valid']]
    require(len(scored) == 96, 'Expected 96 post-warmup scored observations')
    for case in summary['cases']:
        check_scores([row for row in scored if row['case'] == case['name']], case['metrics'])
    for split in ('train', 'heldout'):
        check_scores([row for row in scored if row['split'] == split], summary['splits'][split])
    heldout = summary['splits']['heldout']
    expected_win = heldout['neural']['vector_rmse_px_s'] < min(
        heldout['conventional']['vector_rmse_px_s'], heldout['zero']['vector_rmse_px_s'])
    require(summary['neural_outperformed_both_heldout_baselines'] == expected_win, 'Conclusion differs from scores')
    implementation = ROOT.parents[2]/'experiments'/'mantis_motion.py'
    print(json.dumps(dict(status='verified', observations=len(rows), scored=len(scored),
        current_implementation_matches_recorded_hash=(digest(implementation) == provenance['implementation_sha256']
                                                     if implementation.is_file() else None),
        heldout_vector_rmse_px_s={name: values['vector_rmse_px_s'] for name, values in heldout.items()},
        neural_outperformed_both_heldout_baselines=expected_win), indent=2))


if __name__ == '__main__':
    main()
