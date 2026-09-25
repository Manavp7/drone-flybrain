"""Recompute a frozen SIH run's gates and audit command deadlines (not replay)."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.px4_follow import score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    folder = args.run
    def read(name):
        return json.loads((folder/name).read_text(), parse_constant=lambda value:
                          (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    provenance = read('provenance.json')
    source_root = (folder/'sources').resolve() if (folder/'sources').is_dir() else ROOT
    for relative, expected in provenance['source_sha256'].items():
        file = (source_root/relative).resolve()
        if not file.is_relative_to(source_root) or hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Runtime source changed: {relative}')
    rows, observations, events = (read(name) for name in
                                  ('control.json', 'observations.json', 'events.json'))
    summary = read('summary.json')
    owned = [event for event in events if event['event'] == 'owned_px4_verified']
    if (len(owned) != 1 or not summary['actual_px4']
            or summary['session'] != provenance['session']
            or summary['px4_sensor_profile'] != provenance['px4_sensor_profile']
            or summary['px4_sensor_profile'] != owned[0]['px4_sensor_profile']):
        raise ValueError('Session or simulation profile mismatch')
    actual = score(rows, observations, events, smoke=summary['mode']=='autopilot_smoke')
    if summary['error'] is not None or not actual['passed']:
        raise ValueError('Run did not pass all acceptance gates')
    if any(summary[key] != value for key, value in actual.items()):
        raise ValueError('Recomputed summary mismatch')
    commands = {o['command']['sequence']: o['command'] for o in observations}
    advancing = 0
    for row in rows:
        if row['sent_speed'] <= 0:
            continue
        command = commands[row['command_sequence']]
        if (not command['valid'] or not command['issued_at_s'] <= row['sent_at_s'] < command['valid_until_s']
                or not 0 < row['sent_speed'] <= .45 or not 0 <= row['depth_age_s'] <= .1
                or not 0 <= row['state_age_s'] <= .1 or not 0 <= row['tick_s'] <= .1):
            raise ValueError('Forward command violates authority/deadline limits')
        advancing += 1
    print(json.dumps(dict(verified=True, sources=len(provenance['source_sha256']),
        observations=len(observations), control_ticks=len(rows), advancing_ticks=advancing,
        px4_sensor_profile=summary['px4_sensor_profile'],
        checks=actual['checks'], scope='artifact consistency and gate recomputation; not physics replay'), indent=2))


if __name__ == '__main__':
    main()
