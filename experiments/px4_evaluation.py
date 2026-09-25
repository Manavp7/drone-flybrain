"""Offline PX4 mission evaluation. Actor annotations never authorize control.

Mission windows and completion/send times use absolute host monotonic seconds.
Each observation is ``{packet, command, evaluation}``; the evaluator annotation
must repeat the packet's exact sequence and capture timestamp. Missing or
ambiguous identity evidence is unknown, never a correct association.
"""
from __future__ import annotations

import copy
import json
import math
from numbers import Real
from statistics import mean


RESULT_AGE_S = .65
COMMAND_AGE_S = .9
SENSOR_AGE_S = .1
RECOVERY_AGE_S = 1.2


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


def _window(start, end):
    if not _finite(start) or not _finite(end) or not 0 <= start < end:
        raise ValueError('An increasing, finite absolute mission window is required')


def _age(value, limit):
    return _finite(value) and 0 <= value <= limit + 1e-9


def _percentiles(values):
    values = sorted(float(v) for v in values if _finite(v))
    def at(fraction):
        if not values:
            return None
        coordinate = (len(values)-1)*fraction
        lower = int(coordinate)
        return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(coordinate-lower)
    return dict(p50=at(.5), p95=at(.95), maximum=max(values) if values else None)


def _yaw_span(rows):
    angles = [r['yaw'] for r in rows if _finite(r.get('yaw'))]
    if not angles:
        return None
    unwrapped = [float(angles[0])]
    for before, after in zip(angles, angles[1:]):
        unwrapped.append(unwrapped[-1] + math.atan2(math.sin(after-before), math.cos(after-before)))
    return max(unwrapped)-min(unwrapped)


def _records(observations, end_s=math.inf):
    # Importing this evaluator creates no renderer or inference runtime.
    from experiments.mantis_arena import evaluate_selection
    records, errors = [], []
    intended = selected = None
    previous_sequence, previous_capture, previous_completion = -1, -math.inf, -math.inf
    session = None
    for index, item in enumerate(observations):
        packet, command = item.get('packet', {}), item.get('command', {})
        sequence, capture, completed = (packet.get(key) for key in
                                        ('sequence', 'capture_time_s', 'completed_s'))
        packet_session = packet.get('session')
        if (type(sequence) is not int or sequence <= previous_sequence
                or not _finite(capture) or not _finite(completed)
                or not 0 <= capture <= completed or capture <= previous_capture
                or completed < previous_completion or not isinstance(packet_session, str)
                or not packet_session or session is not None and packet_session != session):
            errors.append(dict(observation_index=index, reason='invalid_packet_order_or_time'))
            continue
        previous_sequence, previous_capture, previous_completion = sequence, capture, completed
        session = packet_session
        if completed > end_s:
            continue
        action = packet.get('selection_action') or {}
        request = action.get('request') or {}
        explicit = (action.get('accepted') is True and type(request.get('track_id')) is int
                    and request['track_id'] > 0 and type(request.get('sequence')) is int
                    and 0 <= request['sequence'] <= sequence)
        observation, selection = packet.get('observation', {}), packet.get('selection', {})
        if observation.get('valid') is True and (type(observation.get('sequence')) is not int
                or observation['sequence'] != sequence or observation.get('capture_time_s') != capture):
            errors.append(dict(observation_index=index, reason='observation_capture_or_sequence_mismatch'))
            continue
        evaluation = item.get('evaluation')
        matched = (isinstance(evaluation, dict) and evaluation.get('sequence') == sequence
                   and type(evaluation.get('sequence')) is int
                   and evaluation.get('capture_time_s') == capture
                   and isinstance(evaluation.get('projections'), list))
        identity = dict(actor_id=None, wrong_person=None, ambiguous=False, best_iou=None)
        if selected is None and explicit:
            # The first selected detector box defines intent, even when guidance
            # in that packet predates selection and is therefore invalid. Never
            # bind to a later occupant of the same temporary track ID.
            selected = request['track_id']
            original = (dict(packet=packet, evaluation=evaluation if matched else None)
                if request['sequence'] == sequence else next(
                    (r for r in records if r['sequence'] == request['sequence']), None))
            if original is not None and original.get('evaluation') is not None:
                original_packet = original['packet']
                detections = original_packet.get('detections', {}).get('detections', [])
                boxes = [d for d in detections if d.get('track_id') == selected and d.get('class_id') == 0]
                observed = original_packet.get('observation', {})
                if not detections and observed.get('valid') is True and observed.get('track_id') == selected:
                    boxes = [observed]
                if len(boxes) == 1:
                    try:
                        selected_identity = evaluate_selection(dict(valid=True,
                            bbox_xyxy=boxes[0].get('bbox_xyxy')), original['evaluation']['projections'])
                        intended = selected_identity['actor_id']
                    except (KeyError, TypeError, ValueError, IndexError):
                        pass  # Unknown original selection stays unqualified.
        available = (selected is not None and observation.get('valid') is True
                     and observation.get('track_id') == selected
                     and selection.get('track_id') == selected and selection.get('held') is False
                     and _age(completed-capture, RESULT_AGE_S))
        if matched:
            try:
                identity = evaluate_selection(observation, evaluation['projections'], intended)
            except (KeyError, TypeError, ValueError, IndexError):
                matched = False
                identity = dict(actor_id=None, wrong_person=None, ambiguous=False, best_iou=None)
        command_consistent = (command.get('sequence') == sequence
            and type(command.get('sequence')) is int and command.get('capture_time_s') == capture
            and _finite(command.get('issued_at_s')) and completed <= command['issued_at_s']
            and _finite(command.get('valid_until_s'))
            and abs(command['valid_until_s']-(capture+COMMAND_AGE_S)) <= 1e-8)
        valid = (command_consistent and command.get('valid') is True and available
                 and _age(command['issued_at_s']-capture, RESULT_AGE_S)
                 and packet.get('candidate', {}).get('valid') is True
                 and packet['candidate'].get('capture_time_s') == capture
                 and packet['candidate'].get('track_id') == selected)
        records.append(dict(sequence=sequence, capture=capture, completed=completed,
            packet=packet, command=command, evaluation=evaluation if matched else None,
            annotation_matched=matched, identity=identity, selected_track=selected,
            tracking_available=available, command_valid=bool(valid)))
    return records, selected, intended, errors


def _intervals(records, start, end, predicate):
    """Integrate causal completed-result state, including startup and expiry gaps."""
    boundaries = sorted({start, end, *[min(end, max(start, t)) for r in records
                        for t in (r['completed'], r['capture']+RESULT_AGE_S)]})
    intervals, cursor, latest = [], 0, None
    for left, right in zip(boundaries, boundaries[1:]):
        while cursor < len(records) and records[cursor]['completed'] <= left:
            latest = records[cursor]
            cursor += 1
        value = (latest is not None and left < latest['capture']+RESULT_AGE_S
                 and predicate(latest))
        if not value and right > left:
            if intervals and abs(intervals[-1]['end_s']-left) <= 1e-9:
                intervals[-1]['end_s'] = right
            else:
                intervals.append(dict(start_s=left, end_s=right))
    for interval in intervals:
        interval['duration_s'] = interval['end_s']-interval['start_s']
    return intervals


def _positive_send(row, records):
    """Reconstruct the complete authority chain for one forward send."""
    record = records.get(row.get('command_sequence'))
    if type(row.get('command_sequence')) is not int or record is None or not record['command_valid']:
        return False
    command, at, speed = record['command'], row.get('sent_at_s'), row.get('sent_speed')
    return bool(_finite(at) and _finite(speed) and 0 < speed <= .45
        and command['issued_at_s'] <= at < command['valid_until_s']
        and row.get('command_expiry_s') == command['valid_until_s']
        and _finite(command.get('forward_speed')) and speed <= command['forward_speed']+1e-9
        and _finite(row.get('requested_speed')) and speed <= row['requested_speed']+1e-9
        and row.get('depth_reason') == 'clear'
        and _age(row.get('depth_age_s'), SENSOR_AGE_S)
        and _age(row.get('state_age_s'), SENSOR_AGE_S)
        and _age(row.get('tick_s'), SENSOR_AGE_S))


def mission_metrics(rows, observations, *, mission_start_s, mission_end_s):
    """Return descriptive metrics; no success threshold is silently selected."""
    _window(mission_start_s, mission_end_s)
    records, selected, intended, errors = _records(observations, mission_end_s)
    within = [r for r in records if mission_start_s <= r['completed'] <= mission_end_s]
    sends = [r for r in rows if _finite(r.get('sent_at_s'))
             and mission_start_s <= r['sent_at_s'] <= mission_end_s]
    losses = _intervals(records, mission_start_s, mission_end_s, lambda r: r['tracking_available'])
    unconfirmed = _intervals(records, mission_start_s, mission_end_s,
        lambda r: r['tracking_available'] and r['identity']['wrong_person'] is False)
    duration = mission_end_s-mission_start_s
    wrong_frames = [r for r in within if r['identity']['wrong_person'] is True]
    wrong_episodes = sum(r['identity']['wrong_person'] is True
        and (i == 0 or within[i-1]['identity']['wrong_person'] is not True)
        for i, r in enumerate(within))
    positives = [r for r in sends if _finite(r.get('sent_speed')) and r['sent_speed'] > 0]
    by_sequence = {r['sequence']: r for r in records}
    bad_sends = [i for i, r in enumerate(sends) if not _finite(r.get('sent_speed'))
                 or r['sent_speed'] < 0 or r['sent_speed'] > 0 and not _positive_send(r, by_sequence)]
    malformed_rows = sum(not _finite(r.get('sent_at_s')) for r in rows)
    chronological = all(a['sent_at_s'] <= b['sent_at_s'] for a, b in zip(sends, sends[1:]))
    correct = sum(r['identity']['wrong_person'] is False for r in within)
    assessable = correct+len(wrong_frames)
    ambiguous = sum(r['identity']['ambiguous'] is True for r in within)
    reported_calls = [r['packet'].get('flyvis_observations') for r in within]
    calls = sum(r['packet'].get('flyvis_used') is True for r in within)
    return dict(observation_count=len(within), input_observation_count=len(observations),
        selected_track=selected, intended_actor=intended,
        mission_duration_s=duration, tracking_loss_episodes=len(losses), tracking_losses=losses,
        tracking_loss_duration_s=sum(i['duration_s'] for i in losses),
        tracking_available_fraction=1-sum(i['duration_s'] for i in losses)/duration,
        correct_identity_duration_fraction=1-sum(i['duration_s'] for i in unconfirmed)/duration,
        wrong_person_frames=len(wrong_frames), wrong_person_episodes=wrong_episodes,
        identity_assessable_count=assessable, identity_correct_count=correct,
        identity_ambiguous_count=ambiguous, identity_unknown_count=len(within)-assessable-ambiguous,
        identity_unassessable_count=len(within)-assessable,
        identity_annotation_mismatch_count=sum(not r['annotation_matched'] for r in within),
        identity_correct_fraction=correct/len(within) if within else None,
        command_valid_count=sum(r['command_valid'] for r in within),
        command_valid_fraction=sum(r['command_valid'] for r in within)/len(within) if within else None,
        latency_s=_percentiles([r['completed']-r['capture'] for r in within]),
        release_latency_s=_percentiles([r['command']['issued_at_s']-r['capture'] for r in within
                                      if _finite(r['command'].get('issued_at_s'))]),
        physical_yaw_span_rad=_yaw_span(sends), positive_send_ticks=len(positives),
        unsafe_positive_send_ticks=sum(not _positive_send(r, by_sequence) for r in positives),
        send_evidence_valid=not bad_sends and not malformed_rows and chronological,
        invalid_send_indices=bad_sends, malformed_row_count=malformed_rows,
        packet_integrity_errors=errors,
        flyvis_observation_calls=calls,
        flyvis_reported_total=max((v for v in reported_calls if type(v) is int and v >= 0), default=0),
        flyvis_guidance_commands=sum(r['command_valid'] and r['packet'].get('flyvis_used') is True
                                    and r['packet'].get('neural_valid') is True for r in within),
        flyvis_positive_send_ticks=sum(by_sequence.get(r.get('command_sequence'), {}).get('packet', {})
            .get('flyvis_used') is True and _positive_send(r, by_sequence) for r in positives),
        evaluation_scope='offline projected-mesh association; occlusion/overlap may be unknown; warmup calls excluded')


def _capture_yaw(record, rows):
    evaluation = record.get('evaluation') or {}
    if _finite(evaluation.get('pose_yaw')):
        return evaluation['pose_yaw']
    # Conservative fallback: never use a future or distant pose for a capture.
    preceding = [r for r in rows if _finite(r.get('sent_at_s')) and _finite(r.get('yaw'))
                 and _age(record['capture']-r['sent_at_s'], SENSOR_AGE_S)]
    return max(preceding, key=lambda r: r['sent_at_s'])['yaw'] if preceding else None


def recovery_proof(rows, observations, *, mission_start_s, mission_end_s):
    """Prove a real stationary depth-gap recovery; report every failed subcheck."""
    _window(mission_start_s, mission_end_s)
    records, selected, intended, errors = _records(observations, mission_end_s)
    by_sequence = {r['sequence']: r for r in records}
    gaps = [r for r in records if r['evaluation'] and r['evaluation'].get('depth_drop') is True
            and mission_start_s <= r['capture'] <= mission_end_s]
    checks = dict(injected_depth_gap=False, stationary_memory_used=False,
        pose_turn_at_least_002_rad=False, gap_command_invalid=False, gap_zero_forward=False,
        same_selected_id_recovered=False, current_depth_valid_recovery=False,
        positive_recovery_send=False, recovered_within_1_2s=False, evaluator_identity_preserved=False,
        restored_anchor_reprojected=False, original_anchor_fresh_at_recovery=False,
        no_explicit_reselection=False)
    details = dict(gap_sequence=None, recovery_sequence=None, pose_turn_rad=None,
                   recovery_delay_s=None, gap_send_ticks=0, association_recovery_sequence=None,
                   original_anchor_age_at_recovery_s=None)
    if len(gaps) != 1:
        return dict(passed=False, checks=checks, gap_count=len(gaps), **details)
    gap = gaps[0]
    details['gap_sequence'] = gap['sequence']
    checks['injected_depth_gap'] = (type(gap['packet'].get('input_finite_depth_pixels')) is int
                                  and gap['packet']['input_finite_depth_pixels'] == 0)
    checks['gap_command_invalid'] = gap['command'].get('valid') is False
    memory = next((m for m in gap['packet'].get('stationary_association_memory', [])
        if m.get('track_id') == selected and m.get('retained_for_association_only') is True
        and m.get('missing_depth_capture_time_s') == gap['capture']
        and _finite(m.get('anchor_capture_time_s'))
        and 0 < gap['capture']-m['anchor_capture_time_s'] <= RECOVERY_AGE_S), None)
    anchor = next((r for r in records if memory and r['capture'] == memory['anchor_capture_time_s']
                   and r['sequence'] < gap['sequence'] and r['tracking_available']), None)
    checks['stationary_memory_used'] = bool(memory and anchor
        and gap['evaluation'].get('trajectory') == 'stationary')
    restored = next((r for r in records if r['sequence'] > gap['sequence']
        and r['evaluation'] and r['evaluation'].get('depth_drop') is False
        and type(r['packet'].get('input_finite_depth_pixels')) is int
        and r['packet']['input_finite_depth_pixels'] > 0), None)
    if restored and anchor:
        details['association_recovery_sequence'] = restored['sequence']
        details['original_anchor_age_at_recovery_s'] = restored['capture']-anchor['capture']
        checks['original_anchor_fresh_at_recovery'] = (
            0 < details['original_anchor_age_at_recovery_s'] <= RECOVERY_AGE_S)
        checks['restored_anchor_reprojected'] = restored['tracking_available'] and any(
            r.get('valid') is True and r.get('track_id') == selected
            and r.get('anchor_capture_time_s') == anchor['capture']
            and r.get('previous_observation_time_s') == gap['capture']
            for r in restored['packet'].get('reprojections', []))
    if anchor:
        first, last = _capture_yaw(anchor, rows), _capture_yaw(gap, rows)
        if first is not None and last is not None:
            details['pose_turn_rad'] = abs(math.atan2(math.sin(last-first), math.cos(last-first)))
            checks['pose_turn_at_least_002_rad'] = details['pose_turn_rad'] >= .02
    recoveries = [r for r in records if r['sequence'] > gap['sequence']
        and r['command_valid'] and r['completed'] <= mission_end_s
        and r['evaluation'] and r['evaluation'].get('depth_drop') is False
        and type(r['packet'].get('input_finite_depth_pixels')) is int
        and r['packet']['input_finite_depth_pixels'] > 0
        and _finite(r['packet'].get('candidate', {}).get('surface_optical_z_m'))
        and r['packet']['candidate']['surface_optical_z_m'] > 0]
    recovery = next((r for r in recoveries if any(
        row.get('command_sequence') == r['sequence'] and _positive_send(row, by_sequence)
        and row['sent_at_s'] <= mission_end_s for row in rows)),
        recoveries[0] if recoveries else None)
    if recovery:
        details['recovery_sequence'] = recovery['sequence']
        details['recovery_delay_s'] = recovery['command']['issued_at_s']-gap['capture']
        checks['same_selected_id_recovered'] = (selected is not None
            and recovery['packet']['observation'].get('track_id') == selected
            and gap['packet'].get('selection', {}).get('track_id') == selected)
        checks['current_depth_valid_recovery'] = True
        checks['evaluator_identity_preserved'] = (intended is not None and anchor is not None
            and anchor['identity']['wrong_person'] is False
            and recovery['identity']['wrong_person'] is False)
        positive = [r for r in rows if r.get('command_sequence') == recovery['sequence']
                    and _positive_send(r, by_sequence) and r['sent_at_s'] <= mission_end_s]
        checks['positive_recovery_send'] = bool(positive)
        checks['recovered_within_1_2s'] = bool(positive and
            0 < min(r['sent_at_s'] for r in positive)-gap['capture'] <= RECOVERY_AGE_S)
        checks['no_explicit_reselection'] = not any(
            (r['packet'].get('selection_action') or {}).get('accepted') is True
            for r in records if gap['sequence'] <= r['sequence'] <= recovery['sequence'])
    issued = gap['command'].get('issued_at_s')
    until = recovery['command']['issued_at_s'] if recovery else mission_end_s
    gap_sends = [r for r in rows if _finite(issued) and _finite(r.get('sent_at_s'))
                 and issued <= r['sent_at_s'] < until]
    details['gap_send_ticks'] = len(gap_sends)
    checks['gap_zero_forward'] = bool(gap_sends) and all(
        r.get('sent_speed') == 0 and r.get('requested_speed') == 0 for r in gap_sends)
    return dict(passed=not errors and all(checks.values()), checks=checks, gap_count=1, **details)


def mission_score(rows, observations, events, *, mission_start_s, mission_end_s,
                  require_recovery=False, min_tracking_fraction=None):
    """Keep safe execution distinct from explicitly thresholded tracking success."""
    if min_tracking_fraction is not None and (
            not _finite(min_tracking_fraction) or not 0 < min_tracking_fraction <= 1):
        raise ValueError('Tracking fraction must be in (0, 1] or undeclared')
    metrics = mission_metrics(rows, observations,
        mission_start_s=mission_start_s, mission_end_s=mission_end_s)
    names = {e.get('event') for e in events}
    checks = dict(stable_takeoff='stable_takeoff' in names,
        landed_disarmed='landed_disarmed' in names,
        mission_window_complete='mission_window_complete' in names,
        no_runtime_failure='failure' not in names,
        mission_sends_observed=any(_finite(r.get('sent_at_s'))
            and mission_start_s <= r['sent_at_s'] <= mission_end_s for r in rows),
        send_evidence_valid=metrics['send_evidence_valid'],
        packet_evidence_valid=not metrics['packet_integrity_errors'],
        no_wrong_person=metrics['wrong_person_frames'] == 0)
    safety = all(checks.values())
    recovery = recovery_proof(rows, observations,
        mission_start_s=mission_start_s, mission_end_s=mission_end_s)
    tracking_checks = dict(explicit_identity_bound=metrics['intended_actor'] is not None,
        forward_follow_observed=metrics['positive_send_ticks'] > 0,
        fresh_guidance_observed=metrics['command_valid_count'] > 0,
        tracking_policy_declared=min_tracking_fraction is not None,
        correct_identity_time_coverage=(min_tracking_fraction is not None
            and metrics['correct_identity_duration_fraction'] >= min_tracking_fraction),
        required_recovery=(not require_recovery or recovery['passed']))
    tracking = safety and all(tracking_checks.values()) if min_tracking_fraction is not None else None
    return dict(passed=tracking is True, safety_passed=safety, tracking_success=tracking,
        checks=checks, tracking_checks=tracking_checks, metrics=metrics, recovery=recovery,
        tracking_policy=dict(min_correct_identity_duration_fraction=min_tracking_fraction,
            require_recovery=bool(require_recovery), unknown_identity_counts_as_correct=False))


def paired_summary(runs, methods=('neural', 'direct', 'filtered')):
    """Descriptive same-config/repeat comparison; retain failures and missing runs.

Each run supplies method, repeat, nonempty config (excluding method), summary,
and actual_px4=True for genuine native evidence. Equal scene configuration is
not identical observations: each controller produces its own closed-loop path.
Paired deltas require completed mission windows without runtime failures;
full-window tracking failures remain eligible rather than being discarded.
"""
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('Distinct comparison methods are required')
    snapshots = copy.deepcopy(list(runs))
    groups, invalid = {}, []
    for index, run in enumerate(snapshots):
        config, repeat = run.get('config'), run.get('repeat')
        if (not isinstance(config, dict) or not config or type(repeat) is not int or repeat < 0
                or run.get('method') not in methods or not isinstance(run.get('summary'), dict)):
            invalid.append(index)
            continue
        try:
            key = (json.dumps(config, sort_keys=True, allow_nan=False), repeat)
        except (TypeError, ValueError):
            invalid.append(index)
            continue
        groups.setdefault(key, []).append(index)
    pairs = []
    for (config, repeat), indices in groups.items():
        members = {method: [i for i in indices if snapshots[i]['method'] == method] for method in methods}
        complete = all(len(members[method]) == 1 for method in methods)
        actual = all(snapshots[i].get('actual_px4') is True for i in indices)
        actors = {method: snapshots[ids[0]]['summary'].get('metrics', {}).get('intended_actor')
                  if len(ids) == 1 else None for method, ids in members.items()}
        actors_known = all(isinstance(actor, str) and bool(actor) for actor in actors.values())
        same_actor = actors_known and len(set(actors.values())) == 1
        full_windows = all(isinstance(snapshots[i]['summary'].get('checks'), dict)
            and snapshots[i]['summary']['checks'].get('mission_window_complete') is True
            and snapshots[i]['summary']['checks'].get('no_runtime_failure') is True
            for i in indices)
        comparable = complete and actual and same_actor and full_windows
        target_issue = ('intended_actor_unknown' if not actors_known else
                        'intended_actor_mismatch' if not same_actor else None)
        metrics = ('tracking_available_fraction', 'correct_identity_duration_fraction',
                   'command_valid_fraction', 'tracking_loss_duration_s', 'wrong_person_frames')
        deltas = {}
        if comparable and 'neural' in members:
            reference = snapshots[members['neural'][0]]['summary'].get('metrics', {})
            for method in methods:
                if method == 'neural':
                    continue
                baseline = snapshots[members[method][0]]['summary'].get('metrics', {})
                deltas[method] = {name: reference[name]-baseline[name] for name in metrics
                                 if _finite(reference.get(name)) and _finite(baseline.get(name))}
        pairs.append(dict(config=json.loads(config), repeat=repeat, run_indices=indices,
            complete=complete, actual_px4_evidence=actual,
            comparable=comparable, intended_actors=actors, same_intended_actor=same_actor,
            target_comparison_issue=target_issue,
            missing_methods=[m for m, ids in members.items() if not ids],
            duplicate_methods=[m for m, ids in members.items() if len(ids) > 1],
            neural_minus_baseline= deltas))
    aggregates = {}
    for method in methods:
        chosen = [r for r in snapshots if r.get('method') == method]
        summaries = [r['summary'] if isinstance(r.get('summary'), dict) else {} for r in chosen]
        aggregates[method] = dict(run_count=len(chosen), passed=sum(
            s.get('passed') is True for s in summaries),
            failed_or_unqualified=sum(s.get('passed') is not True for s in summaries))
        for name in ('tracking_loss_duration_s', 'command_valid_fraction'):
            values = [s.get('metrics', {}).get(name) for s in summaries
                      if isinstance(s.get('metrics', {}), dict)]
            finite = [v for v in values if _finite(v)]
            aggregates[method][name+'_mean'] = mean(finite) if finite else None
            aggregates[method][name+'_observed_runs'] = len(finite)
    return dict(runs=snapshots, pairs=pairs, invalid_run_indices=invalid,
        methods=aggregates, superiority_established=False, superiority_claim=None,
        interpretation='Descriptive paired configurations and repeats; failed runs retained. '
            'Paired deltas require the same uniquely bound evaluator actor. Closed-loop images differ '
            'by controller. No automatic statistical or causal superiority claim.')
