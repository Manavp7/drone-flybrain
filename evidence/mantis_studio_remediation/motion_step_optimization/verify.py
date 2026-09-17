"""Model-free integrity check for the frozen step-optimization evidence."""
import hashlib
import json
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[2]
packaging = json.loads((BUNDLE/'packaging.json').read_text())
for name, expected in packaging['files'].items():
    actual = hashlib.sha256((BUNDLE/name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit('Evidence digest mismatch: '+name)
receipt = json.loads((BUNDLE/'profile_step_equivalence.json').read_text())
assert receipt['status'] == 'passed'
assert len(receipt['definitions']['times_s']) == 13
assert len(receipt['output_digests']) == 13
for key, name in [('original', 'mantis_motion_before.py'), ('stage1', 'mantis_motion_stage1.py'), ('stage2', 'mantis_motion_stage2.py')]:
    assert receipt['source_hashes'][key] == packaging['files'][name]
    assert receipt['measurements'][key]['gap_resets'] == 1
    assert receipt['measurements'][key]['valid_observations'] == 9
for result in receipt['equivalence'].values():
    for key in ['full_dynamic_states_exact', 'full_decoder_fields_exact', 'restored_initial_states_exact', 'clock_validity_population_summaries_exact']:
        assert result[key] is True
    for key in ['max_activity_abs_difference', 'max_flow_abs_difference', 'max_reset_abs_difference']:
        assert result[key] == 0.
current = ROOT/'experiments/mantis_motion.py'
print(json.dumps(dict(status='passed', files_checked=len(packaging['files']), captures=13,
    current_implementation_matches_recorded_stage2=hashlib.sha256(current.read_bytes()).hexdigest() == receipt['source_hashes']['stage2']), indent=2))
