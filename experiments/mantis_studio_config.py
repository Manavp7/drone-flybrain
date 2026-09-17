"""Strict, bounded inputs shared by the local studio server and simulator."""
import math

DEFAULTS = dict(scenario='walk', method='mantis_neural', duration_s=30., follow_distance_m=3.5,
                target_speed=.08, detours=False, motion_mode='off', recording='compact',
                max_recording_mb=64)
CHOICES = dict(scenario=('walk', 'crossing', 'occlusion', 'stationary', 'detour'),
               method=('mantis_neural', 'direct_yolo', 'alpha_beta'),
               motion_mode=('off', 'observe', 'brake'), recording=('compact', 'full-research'))
RANGES = dict(duration_s=(2., 90.), follow_distance_m=(2.5, 5.), target_speed=(0., .2),
              max_recording_mb=(16., 128.))
LIVE_KEYS = {'follow_distance_m', 'target_speed', 'scenario', 'detours'}


def validate_config(value, *, base=None, live=False):
    if not isinstance(value, dict) or set(value)-set(DEFAULTS):
        raise ValueError('Unknown configuration fields')
    if live and set(value)-LIVE_KEYS:
        raise ValueError('Only target movement, follow distance and detours can change during a run')
    result = dict(DEFAULTS if base is None else base)
    result.update(value)
    for key, choices in CHOICES.items():
        if result[key] not in choices or not isinstance(result[key], str):
            raise ValueError('Invalid '+key)
    for key, (low, high) in RANGES.items():
        number = result[key]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f'{key} must be between {low} and {high}')
        result[key] = float(number)
    if type(result['detours']) is not bool:
        raise ValueError('detours must be boolean')
    if int(result['max_recording_mb']) != result['max_recording_mb']:
        raise ValueError('Recording budget must be an integer number of MB')
    result['max_recording_mb'] = int(result['max_recording_mb'])
    return result
