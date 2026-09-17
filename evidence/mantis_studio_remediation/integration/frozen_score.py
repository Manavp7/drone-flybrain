import numpy as np
MAX_OBSERVATION_AGE_S=.65
def motion_brake_scale(result, now_s):
    """Explicit experimental speed reduction only; not a collision detector.

    Frozen transfer-unit threshold, not fitted to these flights. Image motion
    includes ego-motion and can cause unnecessary slowing. Unknown data holds.
    """
    if not result or result.get('valid') is not True:
        return 0.
    times = [now_s, result.get('capture_time_s'), result.get('response_time_s'), result.get('available_time_s')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0 for v in times):
        return 0.
    if now_s < max(times[2:]):
        return 0.
    age = now_s-times[1]
    magnitude = result.get('neural', {}).get('rms_decoder_magnitude')
    if age < 0 or age > MAX_OBSERVATION_AGE_S or isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)) or not np.isfinite(magnitude) or magnitude < 0:
        return 0.
    return float(np.clip(1.-magnitude/.10, .2, 1.))

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
