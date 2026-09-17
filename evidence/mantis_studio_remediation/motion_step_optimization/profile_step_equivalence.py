"""Serialized exact-state audit of frozen official-step optimization."""
import argparse
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
import numpy as np

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', type=Path, default=ROOT/'models/flyvis_0000_000.manifest.json')
parser.add_argument('--output', type=Path, required=True, help='New output directory; never overwrite evidence')
args = parser.parse_args()
OUTPUT = args.output.resolve()
manifest = args.manifest.resolve()
packaging = json.loads((BUNDLE/'packaging.json').read_text())
for filename in ('mantis_motion_before.py', 'mantis_motion_stage1.py', 'mantis_motion_stage2.py'):
    assert hashlib.sha256((BUNDLE/filename).read_bytes()).hexdigest() == packaging['files'][filename], filename
for filename, expected in packaging['supporting_source_hashes'].items():
    assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == expected, filename
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == packaging['model_manifest_sha256'], 'model manifest changed'
OUTPUT.mkdir(parents=True, exist_ok=False)
sys.path.insert(0, str(ROOT))


def load_snapshot(label, filename):
    spec = importlib.util.spec_from_file_location(label, BUNDLE/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

original = load_snapshot('motion_original', 'mantis_motion_before.py')
stage1 = load_snapshot('motion_stage1', 'mantis_motion_stage1.py')
current = load_snapshot('motion_stage2', 'mantis_motion_stage2.py')
TIMES = [0., .10, .55, 1.10, 1.55, 2.10, 2.55, 3.10, 3.55, 4.10, 4.80, 5.25, 5.80]
SEED = 20271
VELOCITY = [6., -4.]
frames = [current.translated_texture(SEED, VELOCITY, t) for t in TIMES]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_arrays(experiment):
    state = experiment.backend.state
    return {f'{kind}.{key}': getattr(state, kind)[key].detach().cpu().numpy().copy()
            for kind in ('nodes', 'edges') for key in getattr(state, kind)}


def digest(array):
    return dict(shape=list(array.shape), dtype=str(array.dtype), sha256=hashlib.sha256(array.tobytes()).hexdigest())


def run(module, label):
    begin = time.perf_counter()
    experiment = module.RawMotionExperiment(manifest)
    initialization_s = time.perf_counter()-begin
    reset_arrays = [state_arrays(experiment)]
    responses, captured, flows = [], [], []
    for t, rgb in zip(TIMES, frames):
        result = experiment.step(rgb, t)
        responses.append(result)
        arrays = state_arrays(experiment)
        captured.append(arrays)
        flows.append(experiment.backend.decode(arrays['nodes.activity'][0]).copy())
    reset_times = []
    for _ in range(4):
        begin = time.perf_counter()
        experiment.reset()
        reset_times.append(time.perf_counter()-begin)
        reset_arrays.append(state_arrays(experiment))
    long_results = [r for i, r in enumerate(responses) if i > 0 and not r['gap_reset'] and TIMES[i]-TIMES[i-1] > .4]
    metrics = dict(initialization_s=initialization_s, first_capture_s=responses[0]['elapsed_wall_s'],
        ordinary_capture_median_s=statistics.median(r['elapsed_wall_s'] for r in responses[1:] if not r['gap_reset']),
        long_gap_capture_median_s=statistics.median(r['elapsed_wall_s'] for r in long_results),
        long_gap_neural_median_s=statistics.median(r['neural_wall_s'] for r in long_results),
        gap_reset_capture_s=[r['elapsed_wall_s'] for r in responses if r['gap_reset']],
        explicit_reset_s=reset_times, explicit_reset_median_s=statistics.median(reset_times),
        all_capture_s=[r['elapsed_wall_s'] for r in responses],
        neural_capture_s=[r['neural_wall_s'] for r in responses],
        conventional_capture_s=[r['conventional_wall_s'] for r in responses],
        valid_observations=sum(r['valid'] for r in responses), gap_resets=sum(r['gap_reset'] for r in responses))
    print(json.dumps(dict(phase=label, metrics=metrics)), flush=True)
    provenance = experiment.provenance
    del experiment
    gc.collect()
    return metrics, responses, captured, flows, reset_arrays, provenance


runs = {label: run(module, label) for label, module in [('original', original), ('stage1', stage1), ('stage2', current)]}
comparison = {}
for label in ['original', 'stage1']:
    pre, post = runs[label], runs['stage2']
    max_state = max_flow = max_reset = 0.
    for a, b in zip(pre[2], post[2]):
        assert a.keys() == b.keys()
        for key in a:
            max_state = max(max_state, float(np.max(abs(a[key]-b[key]))))
            np.testing.assert_array_equal(a[key], b[key])
    for a, b in zip(pre[3], post[3]):
        max_flow = max(max_flow, float(np.max(abs(a-b))))
        np.testing.assert_array_equal(a, b)
    for a, b in zip(pre[4], post[4]):
        assert a.keys() == b.keys()
        for key in a:
            max_reset = max(max_reset, float(np.max(abs(a[key]-b[key]))))
            np.testing.assert_array_equal(a[key], b[key])
    for a, b in zip(pre[1], post[1]):
        for key in ['capture_time_s', 'stimulus_time_s', 'response_time_s', 'valid', 'gap_reset', 'status', 'neural', 'conventional', 'populations']:
            assert a[key] == b[key], key
    comparison[label+'_vs_stage2'] = dict(full_dynamic_states_exact=True, full_decoder_fields_exact=True,
        restored_initial_states_exact=True, max_activity_abs_difference=max_state,
        max_flow_abs_difference=max_flow, max_reset_abs_difference=max_reset,
        clock_validity_population_summaries_exact=True)

summary = dict(status='passed', definitions=dict(times_s=TIMES, texture_seed=SEED, velocity_px_s=VELOCITY,
    frame_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in frames], heldout_tuning=False,
    purpose='Exact recurrence/decoder preservation and steady compute latency; not accuracy evaluation'),
    source_hashes=dict(original=sha(BUNDLE/'mantis_motion_before.py'), stage1=sha(BUNDLE/'mantis_motion_stage1.py'),
                       stage2=sha(current.__file__)),
    current_implementation_sha256=sha(ROOT/'experiments/mantis_motion.py'),
    current_implementation_matches_stage2=sha(ROOT/'experiments/mantis_motion.py') == sha(current.__file__),
    measurements={label: value[0] for label, value in runs.items()}, equivalence=comparison,
    output_digests=[dict(capture_time_s=t, states={k: digest(v) for k, v in states.items()}, flow=digest(flow))
                    for t, states, flow in zip(TIMES, runs['stage2'][2], runs['stage2'][3])],
    clock_status=[{k: r[k] for k in ['capture_time_s', 'stimulus_time_s', 'response_time_s', 'valid', 'gap_reset', 'status']}
                  for r in runs['stage2'][1]],
    provenance=runs['stage2'][5],
    limitations=['One serialized local CPU sequence, original then stage1 then stage2; cache/order affects timings.',
                 'Intervals are supplied camera timestamps; timings exclude YOLO, target guidance, rendering, recording, and integration scheduling.',
                 'Initialization moves decoder setup out of capture time and must finish before accepting camera frames.',
                 'Exact output preservation cannot improve prior negative neural accuracy scores.',
                 'Raw arrays are checked in memory and omitted from compact receipt; rerunner recreates and compares all arrays.'])
path = OUTPUT/'profile_step_equivalence.json'
with path.open('x') as stream:
    json.dump(summary, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(summary, indent=2), flush=True)
