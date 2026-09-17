"""Check portable development receipts, saved metrics and causal timestamps."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT/'live_checks.json').read_text())
for trial in report['trials']:
    folder = ROOT/'live'/trial['run_id']
    receipt = json.loads((folder/'summary.json').read_text())
    rows = [json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
    for name in ('summary.json', 'provenance.json'):
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == trial['artifact_sha256'][name]
    assert receipt['bytes'] == trial['bytes'] <= receipt['max_bytes'] == trial['max_bytes']
    assert receipt['summary']['statistics'] == trial['statistics']
    assert len(rows) == trial['saved_samples']
    assert all(row['capture_s'] <= row['available_s']+1e-8 for row in rows)
    assert all(b['capture_s'] > a['capture_s'] and b['available_s'] > a['available_s']
               for a, b in zip(rows, rows[1:]))
    scored = [row for row in rows if row['evaluation']['wrong_person'] is not None]
    wrong = sum(row['evaluation']['wrong_person'] for row in scored)
    missing = trial['statistics']['observations']-len(rows)
    assert missing == 0 or receipt['status'] == 'budget-exhausted' and missing == 1
    assert 0 <= trial['statistics']['evaluated_selected_observations']-len(scored) <= missing
    assert 0 <= trial['statistics']['wrong_person_observations']-wrong <= missing
    assert max(row['speed_m_s'] for row in rows) == trial['max_saved_speed_m_s']
print(f"Verified {len(report['trials'])} development receipts, hashes, saved scores and causal clocks")
