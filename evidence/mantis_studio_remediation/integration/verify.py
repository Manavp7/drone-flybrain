"""Verify compact artifacts and recompute frozen final integration scores.

This reuses the frozen scorer and recorded physical endpoint/clearance receipts.
It does not rerun learned models or independently replay motor/depth physics.
"""
from pathlib import Path
import hashlib
import importlib.util
import json

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'manifest.json').read_text())
for name,digest in manifest['files'].items():
 path=ROOT/name
 assert path.is_relative_to(ROOT) and '..' not in Path(name).parts
 assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
spec=importlib.util.spec_from_file_location('frozen_studio_score',ROOT/'frozen_score.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
batch=ROOT/'studio_remediation_validation01'
definition=json.loads((batch/'definition.json').read_text())
count=0
for case in definition['selected_cases']:
 result=json.loads((batch/(case+'.json')).read_text())
 rows=[json.loads(line) for line in (ROOT/'live'/result['run_id']/'observations.jsonl').read_text().splitlines()]
 recomputed=module.score(result['summary'],rows,result['config'],definition['thresholds'])
 assert recomputed==result['outcome'],case
 count+=1
print(json.dumps(dict(status='verified',artifacts=len(manifest['files']),final_cases=count,
                     passed=sum(json.loads((batch/(case+'.json')).read_text())['outcome']['passed']
                                for case in definition['selected_cases'])),indent=2))
