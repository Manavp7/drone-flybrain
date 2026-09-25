"""Pure evidence scorer for independent post-event obstacle depth braking.

This module sees recorded observations and paired guardian receipts only. It
never renders a world, reads geometry, runs inference or provides motor authority.
The counterfactual receipt must come from a synchronized sensor capture with only
the obstacle hidden; that isolation is a separate runner integration contract.
"""
from __future__ import annotations

from collections import Counter
import math

from experiments.flight_contracts import (PHYSICS_DT, DEPTH_MAX_AGE_S,
    COMMAND_MAX_CAPTURE_AGE_S, MAX_OBSERVATION_AGE_S, MAX_SPEED_M_S)

CAUSAL_DEPTH_LIMITS = dict(minimum_consecutive_override_ticks=10,
    minimum_sustained_override_s=.05, minimum_postevent_person_captures=3,
    minimum_postevent_target_fraction=.8, maximum_depth_age_s=DEPTH_MAX_AGE_S,
    timestamp_tolerance_s=1e-8)
EPS = CAUSAL_DEPTH_LIMITS['timestamp_tolerance_s']


def _number(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)


def _integer(value):
    return isinstance(value,int) and not isinstance(value,bool) and value >= 0


def _same(a,b):
    return _number(a) and _number(b) and abs(a-b) <= EPS


def _object(value):
    return value if isinstance(value,dict) else {}


def _observed_person(row, original_id=None, duration=None):
    observation = _object(row.get('observation'))
    seq, capture = row.get('sequence'), row.get('capture_time_s')
    if not (_integer(seq) and _number(capture) and capture >= 0
            and observation.get('valid') is True and observation.get('reason') == 'observed'
            and _integer(observation.get('track_id'))
            and observation.get('sequence') == seq
            and _same(observation.get('capture_time_s'),capture)):
        return False
    if original_id is not None and observation['track_id'] != original_id:
        return False
    if row.get('discarded_after_episode') is not False:
        return False
    completed = row.get('completed_time_s')
    if not (_number(completed) and completed >= capture-EPS
            and completed-capture <= MAX_OBSERVATION_AGE_S+EPS
            and (duration is None or completed <= duration+EPS)):
        return False
    detections = _object(row.get('detections')).get('detections',[])
    return isinstance(detections,list) and any(isinstance(d,dict)
        and d.get('track_id') == observation['track_id'] and d.get('class_id') == 0
        and not isinstance(d.get('class_id'),bool) and d.get('label') == 'person'
        for d in detections)


def _candidate_reason(tick,row,event,original_id,duration):
    t, sequence = tick['time_s'], tick.get('applied_command_sequence')
    request = _object(tick.get('request'))
    if not _integer(sequence) or request.get('sequence') != sequence or row is None:
        return 'missing_or_mismatched_applied_sequence'
    if row['capture_time_s'] < event-EPS:
        return 'command_captured_before_obstacle'
    if not _observed_person(row,original_id,duration):
        return 'no_fresh_observed_original_person'
    candidate = _object(row.get('candidate'))
    estimate = _object(candidate.get('estimate'))
    capture, completed = row['capture_time_s'],row.get('completed_time_s')
    issued, expires = candidate.get('issued_at_s'),candidate.get('valid_until_s')
    if not (candidate.get('valid') is True and candidate.get('sequence') == sequence
            and _same(candidate.get('capture_time_s'),capture)
            and estimate.get('valid') is True and estimate.get('track_id') == original_id
            and _same(estimate.get('capture_time_s'),capture)
            and _same(issued,completed) and _number(expires)
            and _same(expires,capture+COMMAND_MAX_CAPTURE_AGE_S)
            and issued <= t+EPS and t < expires-EPS
            and _same(candidate.get('forward_speed'),request.get('forward_speed'))):
        return 'invalid_unreleased_or_expired_candidate'
    return None


def _pair_reason(tick):
    t = tick['time_s']
    request = _object(tick.get('request')).get('forward_speed')
    actual, pair = _object(tick.get('guardian')),_object(tick.get('counterfactual_guardian'))
    actual_time, pair_time = tick.get('actual_depth_capture_time_s'),tick.get('counterfactual_depth_capture_time_s')
    if not (_number(actual_time) and _number(pair_time) and actual_time >= 0 and pair_time >= 0
            and _same(actual_time,pair_time) and actual_time <= t+EPS
            and t-actual_time <= DEPTH_MAX_AGE_S+EPS):
        return 'missing_future_stale_or_misaligned_depth_pair'
    if not (pair.get('reason') == 'clear' and pair.get('valid_clearance') is True
            and _same(pair.get('forward_speed'),request)
            and _same(actual.get('stopping_distance_m'),pair.get('stopping_distance_m'))
            and actual.get('valid_clearance') is True):
        return 'unobstructed_counterfactual_did_not_allow_same_request'
    actual_meta,pair_meta = _object(actual.get('metadata')),_object(pair.get('metadata'))
    for meta in (actual_meta,pair_meta):
        if not (_same(meta.get('requested_speed_clamped_m_s'),request)
                and _same(meta.get('depth_age_s'),t-actual_time)):
            return 'counterfactual_state_request_or_age_mismatch'
    # These state-derived values must agree for a genuinely paired decision.
    for name in ('considered_speed_m_s','total_reaction_s','state_age_s'):
        if not _same(actual_meta.get(name),pair_meta.get(name)):
            return 'counterfactual_state_request_or_age_mismatch'
    a,b = actual_meta.get('level_velocity_m_s'),pair_meta.get('level_velocity_m_s')
    if not (isinstance(a,list) and isinstance(b,list) and len(a)==len(b)==3
            and all(_same(x,y) for x,y in zip(a,b))):
        return 'counterfactual_state_request_or_age_mismatch'
    return None


def depth_causality(spec,ticks,observations):
    """Return additional gates/metrics for explicitly declared causal tests.

    An old pre-event command alone never counts. Qualification requires actual
    post-event person observations, exact command binding, a blocked positive
    request, and a paired obstacle-free depth receipt that permits that request.
    Ten consecutive 200 Hz control intervals establish the fixed .05 s minimum.
    Existing operational and structural gates remain the caller's responsibility.
    """
    if _object(spec).get('causal_depth_test') is not True:
        return dict(gates={},metrics={})
    event,duration = spec.get('event_s'),spec.get('duration_s')
    valid_spec = (spec.get('kind') == 'stop' and _number(event) and event >= 0
                  and _number(duration) and duration > event)
    gates = dict(causal_depth_trace_well_formed=bool(valid_spec),
                 causal_depth_initial_person_identity=False,
                 causal_depth_postevent_person_captures=False,
                 causal_depth_postevent_target_fraction=False,
                 causal_depth_sustained_fresh_paired_braking=False)
    metrics = dict(causal_depth_limits=dict(CAUSAL_DEPTH_LIMITS))
    if not valid_spec or not isinstance(ticks,list) or not isinstance(observations,list):
        gates['causal_depth_trace_well_formed']=False
        return dict(gates=gates,metrics=dict(metrics,causal_depth_reason='invalid_spec_or_trace'))
    mapping={};post=[];original_id=None;previous_capture=-1.
    for row in observations:
        if not isinstance(row,dict):
            gates['causal_depth_trace_well_formed']=False
            continue
        sequence,capture = row.get('sequence'),row.get('capture_time_s')
        if (not _integer(sequence) or not _number(capture) or capture < 0
                or capture > duration+EPS or capture <= previous_capture
                or sequence in mapping):
            gates['causal_depth_trace_well_formed']=False
            continue
        previous_capture=capture;mapping[sequence]=row
        if original_id is None and capture < event-EPS and _observed_person(row,duration=duration):
            original_id=row['observation']['track_id']
        if capture >= event-EPS:
            post.append(row)
    valid_post=[row for row in post if original_id is not None and _observed_person(row,original_id,duration)]
    fraction=len(valid_post)/len(post) if post else 0.
    gates.update(causal_depth_initial_person_identity=original_id is not None,
        causal_depth_postevent_person_captures=len(valid_post)>=CAUSAL_DEPTH_LIMITS['minimum_postevent_person_captures'],
        causal_depth_postevent_target_fraction=bool(post) and fraction>=CAUSAL_DEPTH_LIMITS['minimum_postevent_target_fraction'])
    rejected=Counter();matched=[];longest=run=0;previous_time=None;previous_index=None
    for tick in ticks:
        if not isinstance(tick,dict):
            gates['causal_depth_trace_well_formed']=False;run=0;continue
        t,index=tick.get('time_s'),tick.get('index')
        if (not _number(t) or t < 0 or t >= duration+EPS or not _integer(index)
                or previous_time is not None and not _same(t-previous_time,PHYSICS_DT)
                or previous_index is not None and index != previous_index+1):
            gates['causal_depth_trace_well_formed']=False;run=0
        if not _number(t) or not _integer(index):
            continue
        previous_time,previous_index=t,index
        if t < event-EPS:
            run=0;continue
        request=_object(tick.get('request')).get('forward_speed')
        actual=_object(tick.get('guardian'))
        if not (_number(request) and 0 < request <= MAX_SPEED_M_S+EPS
                and _same(actual.get('forward_speed'),0.)
                and actual.get('reason') == 'blocked_stopping_distance'):
            run=0;continue
        sequence=tick.get('applied_command_sequence')
        row=mapping.get(sequence) if _integer(sequence) else None
        reason=_candidate_reason(tick,row,event,original_id,duration)
        if (reason is None and _number(tick.get('actual_depth_capture_time_s'))
                and tick['actual_depth_capture_time_s'] < event-EPS):
            reason='paired_depth_captured_before_obstacle'
        if reason is None:
            reason=_pair_reason(tick)
        if reason:
            rejected[reason]+=1;run=0;continue
        run+=1;longest=max(longest,run);matched.append((index,t,sequence))
    sustained=longest*PHYSICS_DT
    gates['causal_depth_sustained_fresh_paired_braking']=(
        longest>=CAUSAL_DEPTH_LIMITS['minimum_consecutive_override_ticks']
        and sustained+EPS>=CAUSAL_DEPTH_LIMITS['minimum_sustained_override_s'])
    metrics.update(causal_depth_original_target_id=original_id,
        causal_depth_postevent_captures=len(post),causal_depth_valid_postevent_person_captures=len(valid_post),
        causal_depth_postevent_target_fraction=float(fraction),causal_depth_qualifying_ticks=len(matched),
        causal_depth_longest_consecutive_ticks=longest,causal_depth_longest_sustained_s=float(sustained),
        causal_depth_qualifying_observation_sequences=sorted({r[2]for r in matched}),
        causal_depth_first_qualifying_time_s=matched[0][1]if matched else None,
        causal_depth_rejected_positive_block_ticks=dict(rejected))
    return dict(gates=gates,metrics=metrics)
