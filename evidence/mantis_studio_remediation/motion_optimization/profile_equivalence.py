"""One serialized original/optimized checkpoint equivalence and timing audit."""
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

BUNDLE=Path(__file__).resolve().parent
ROOT=BUNDLE.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest',type=Path,default=ROOT/'models/flyvis_0000_000.manifest.json')
parser.add_argument('--output',type=Path,required=True,help='New directory for the repeated measurement')
args=parser.parse_args()
OUTPUT=args.output.expanduser().resolve()
OUTPUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT))
from experiments import mantis_motion as current

old_path=BUNDLE/'mantis_motion_before.py'
spec=importlib.util.spec_from_file_location('mantis_motion_before',old_path)
old=importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
TIMES=[0.,.10,.23,.34,.48,.60,.74,.85,1.02,1.70,1.82,1.94,2.06,2.18,2.30]
frames=[current.translated_texture(9103,[12.,-8.],t) for t in TIMES]


def state_arrays(experiment):
    state=experiment.backend.state
    return {f'{kind}.{key}':getattr(state,kind)[key].detach().cpu().numpy().copy()
            for kind in ('nodes','edges') for key in getattr(state,kind)}


def run(module,label):
    begin=time.perf_counter()
    experiment=module.RawMotionExperiment(args.manifest.expanduser().resolve())
    initialization_s=time.perf_counter()-begin
    reset_arrays=[state_arrays(experiment)]
    responses=[]; captured=[]; flows=[]
    for t,rgb in zip(TIMES,frames):
        result=experiment.step(rgb,t)
        responses.append(result)
        arrays=state_arrays(experiment)
        captured.append(arrays)
        flows.append(experiment.backend.decode(arrays['nodes.activity'][0]).copy())
    reset_times=[]
    for _ in range(4):
        begin=time.perf_counter();experiment.reset();reset_times.append(time.perf_counter()-begin)
        reset_arrays.append(state_arrays(experiment))
    metrics=dict(initialization_s=initialization_s, first_capture_s=responses[0]['elapsed_wall_s'],
        ordinary_capture_median_s=statistics.median(r['elapsed_wall_s'] for r in responses[1:] if not r['gap_reset']),
        gap_reset_capture_s=[r['elapsed_wall_s'] for r in responses if r['gap_reset']],
        explicit_reset_s=reset_times, explicit_reset_median_s=statistics.median(reset_times),
        neural_capture_s=[r['neural_wall_s'] for r in responses],
        conventional_capture_s=[r['conventional_wall_s'] for r in responses],
        valid_observations=sum(r['valid'] for r in responses), gap_resets=sum(r['gap_reset'] for r in responses))
    print(json.dumps(dict(phase=label,metrics=metrics)),flush=True)
    del experiment;gc.collect()
    return metrics,responses,captured,flows,reset_arrays

pre=run(old,'before')
post=run(current,'after')
max_state=max_flow=max_reset=0.
state_equal=flow_equal=reset_equal=True
for a,b in zip(pre[2],post[2]):
    assert a.keys()==b.keys()
    for key in a:
        max_state=max(max_state,float(np.max(abs(a[key]-b[key]))))
        state_equal &= bool(np.array_equal(a[key],b[key]))
        np.testing.assert_allclose(a[key],b[key],rtol=0,atol=1e-7)
for a,b in zip(pre[3],post[3]):
    max_flow=max(max_flow,float(np.max(abs(a-b))))
    flow_equal &= bool(np.array_equal(a,b))
    np.testing.assert_allclose(a,b,rtol=0,atol=1e-7)
for a,b in zip(pre[4],post[4]):
    for key in a:
        max_reset=max(max_reset,float(np.max(abs(a[key]-b[key]))))
        reset_equal &= bool(np.array_equal(a[key],b[key]))
        np.testing.assert_array_equal(a[key],b[key])
for a,b in zip(pre[1],post[1]):
    for key in ['capture_time_s','stimulus_time_s','response_time_s','valid','gap_reset','status','neural','conventional','populations']:
        assert a[key]==b[key],key
summary=dict(status='passed',definitions=dict(times_s=TIMES,texture_seed=9103,velocity_px_s=[12.,-8.],
    frame_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in frames],heldout_tuning=False),
    source_hashes=dict(before=hashlib.sha256(old_path.read_bytes()).hexdigest(),
        after=hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest()),
    before=pre[0],after=post[0],equivalence=dict(full_dynamic_states_exact=state_equal,
    full_decoder_fields_exact=flow_equal,restored_initial_states_exact=reset_equal,
    max_activity_abs_difference=max_state,max_flow_abs_difference=max_flow,max_reset_abs_difference=max_reset,
    clock_validity_population_summaries_exact=True),
    limitations=['One serialized local CPU verification sequence, not a real-time guarantee.',
                 'This optimization preserves neural outputs; it cannot improve prior accuracy scores.'])
(OUTPUT/'profile_equivalence.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
print(json.dumps(summary,indent=2),flush=True)
