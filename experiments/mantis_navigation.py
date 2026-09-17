"""Short forward-only detours certified from current registered camera depth.

The local planner has no actor IDs, obstacle poses, map or side/reverse motion
authority. It brakes before turning, probes a few visible headings, then
rechecks the corridor at every step. DepthGuardian remains final speed authority.
"""
from dataclasses import replace
from numbers import Integral, Real
import math

import numpy as np

from experiments.flight_contracts import rotation_from_euler, wrap_angle
from experiments.flight_safety import DepthGuardian


def _finite(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


class DetourNavigator:
    """Bounded conventional local planner; fresh selection comes from caller.

    request['sequence'] must be None unless the caller has validated the selected
    person's observation and capture-anchored command expiry. range_speed must
    derive from that same registered target depth, retaining standoff control
    while a short detour yaw differs from target bearing. A failed-attempt hold
    is latched; disable detours or reset to attempt again. This is not a map.
    """
    def __init__(self):
        self.probe = DepthGuardian(reaction_allowance_s=4.)
        self.guardian = DepthGuardian()
        self.completed = 0
        self.attempts = 0
        self.reset()

    def reset(self):
        self.phase = 'following'
        self.heading = None
        self.origin = None
        self.started_s = None
        self.last_candidates = []
        self._hold_reason = None
        self._last_time = None
        self._selected_id = None
        self._last_rejoin_distance = 0.

    def _probe_heading(self, depth, state, yaw, speed):
        hypothetical = replace(state, rotation=rotation_from_euler(yaw=yaw))
        return self.probe.check(depth, hypothetical, speed, state.time_s)

    def _result(self, depth, state, speed, yaw, reason, **extras):
        """Final check uses the actual current heading, never a planned pose."""
        safety = self.guardian.check(depth, state, speed, state.time_s)
        if not safety['valid_clearance']:
            speed, yaw = 0., state.yaw
            reason = 'depth_' + safety['reason']
        else:
            speed = safety['forward_speed']
            if speed == 0 and safety['reason'] == 'blocked_stopping_distance':
                yaw = state.yaw
                reason = 'depth_blocked_stopping_distance'
        return dict(forward_speed=float(speed), yaw_target=float(yaw), phase=self.phase,
                    reason=reason, candidates=list(self.last_candidates),
                    detours_completed=self.completed, detour_attempts=self.attempts,
                    scene_truth_used=False, guardian=safety, **extras)

    def _hold(self, depth, state, reason):
        self.phase, self._hold_reason = 'holding', reason
        return self._result(depth, state, 0., state.yaw, reason)

    def update(self, depth, state, request, enabled=True):
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        health = self.guardian.check(depth, state, 0., getattr(state, 'time_s', float('nan')))
        if health['reason'] == 'invalid_state_or_request':
            self.reset()
            return dict(forward_speed=0., yaw_target=0., phase='holding',
                        reason='invalid_state_or_request', candidates=[], guardian=health,
                        detours_completed=self.completed, detour_attempts=self.attempts,
                        scene_truth_used=False)
        if self._last_time is not None and state.time_s < self._last_time - 1e-9:
            self.reset()
            return self._hold(depth, state, 'reordered_state')
        self._last_time = float(state.time_s)
        if not enabled:
            self.reset()
        if self.started_s is not None and state.time_s-self.started_s > 14.:
            return self._hold(depth, state, 'detour_time_limit')
        if not isinstance(request, dict):
            self.reset()
            return self._hold(depth, state, 'invalid_selected_request')
        sequence = request.get('sequence')
        if (sequence is None or not isinstance(sequence, Integral)
                or isinstance(sequence, (bool, np.bool_)) or sequence < 0):
            # A verified detector timeout may later recover the same target.
            # Waiting cannot renew the attempt's budget or clear a failed plan.
            return self._result(depth, state, 0., state.yaw,
                                self._hold_reason or 'no_fresh_selected_target')
        forward, yaw = request.get('forward_speed'), request.get('yaw_target')
        range_speed = request.get('range_speed', forward)
        if (not all(_finite(v) for v in (forward, yaw, range_speed))
                or forward < 0 or range_speed < 0 or request.get('valid') is False):
            self.reset()
            return self._hold(depth, state, 'invalid_selected_request')
        # Optional source timestamps are rechecked; the original active_request
        # caller already enforces these before supplying a non-None sequence.
        for key in ('valid_until_s', 'capture_time_s'):
            if key in request and not _finite(request[key]):
                self.reset()
                return self._hold(depth, state, 'invalid_selected_request')
        if (('valid_until_s' in request and state.time_s >= request['valid_until_s']-1e-9)
                or ('capture_time_s' in request and (request['capture_time_s'] > state.time_s+1e-9
                    or state.time_s-request['capture_time_s'] >= .90-1e-9))):
            return self._result(depth, state, 0., state.yaw,
                                self._hold_reason or 'no_fresh_selected_target')
        selected_id = request.get('track_id')
        if self._selected_id is not None and selected_id != self._selected_id:
            self.reset()
        self._selected_id = selected_id
        if self.started_s is not None and state.time_s-self.started_s > 14.:
            return self._hold(depth, state, 'detour_time_limit')
        if not health['valid_clearance']:
            if health['reason'] in ('unknown_reverse_motion', 'unknown_side_or_vertical_motion'):
                # The autopilot can briefly overshoot zero velocity while
                # braking. Keep requesting zero at the current heading; resume
                # the phase only once the unchanged guardian accepts the real
                # velocity again. This grants no reverse/side authority.
                return self._result(depth, state, 0., state.yaw, 'stabilizing_before_detour')
            return self._hold(depth, state, 'depth_' + health['reason'])
        if not enabled:
            self.reset()
            return self._result(depth, state, min(.45, float(forward)), yaw, 'detours_disabled')
        speed = min(.30, float(range_speed))
        if speed <= .01:
            self.reset()
            return self._result(depth, state, 0., state.yaw, 'follow_distance_reached')
        if self.phase == 'holding':
            return self._result(depth, state, 0., state.yaw, self._hold_reason or 'no_observed_detour')
        target_heading = float(yaw)
        direct = self._probe_heading(depth, state, target_heading, speed)
        if self.phase == 'following':
            if direct['reason'] == 'clear':
                return self._result(depth, state, min(speed, float(forward)), target_heading, 'following')
            if direct['reason'] != 'blocked_stopping_distance':
                return self._result(depth, state, 0., state.yaw, 'direct_' + direct['reason'])
            self.phase, self.started_s = 'braking', float(state.time_s)
            self.attempts += 1
        if state.time_s-self.started_s > 14.:
            return self._hold(depth, state, 'detour_time_limit')
        if self.phase == 'braking':
            if np.linalg.norm(state.velocity) > .025:
                return self._result(depth, state, 0., state.yaw, 'stop_before_inspecting')
            self.last_candidates = []
            for offset in (-.18, .18, -.25, .25):
                candidate_yaw = wrap_angle(target_heading+offset)
                inspected = self._probe_heading(depth, state, candidate_yaw, speed)
                self.last_candidates.append(dict(heading_rad=candidate_yaw, offset_rad=offset,
                    observed_clear=inspected['reason'] == 'clear', reason=inspected['reason'],
                    clearance_m=inspected['minimum_clearance_m'], coverage=inspected['coverage']))
            clear = [candidate for candidate in self.last_candidates if candidate['observed_clear']]
            if not clear:
                return self._hold(depth, state, 'no_observed_detour')
            chosen = min(clear, key=lambda candidate: abs(candidate['offset_rad']))
            self.heading = chosen['heading_rad']
            if self.origin is None:
                self.origin = np.asarray(state.position, dtype=float).copy()
            self.phase = 'turning'
        if self.phase in ('turning', 'passing'):
            if abs(wrap_angle(self.heading-target_heading)) > .26:
                return self._hold(depth, state, 'selected_target_left_detour_view')
            route = self._probe_heading(depth, state, self.heading, speed)
            if (route['reason'] != 'clear' and not (self.phase == 'turning'
                    and route['reason'] == 'unknown_outside_field_of_view')):
                return self._hold(depth, state, 'detour_' + route['reason'])
        if self.phase == 'turning':
            if np.linalg.norm(state.velocity) > .025:
                return self._result(depth, state, 0., state.yaw, 'stop_before_turning')
            if abs(wrap_angle(self.heading-state.yaw)) > .045:
                return self._result(depth, state, 0., self.heading, 'turn_in_place')
            if route['reason'] != 'clear':
                return self._hold(depth, state, 'detour_' + route['reason'])
            self.phase = 'passing'
        if self.phase == 'passing':
            distance = float(np.linalg.norm(state.position[:2]-self.origin[:2]))
            if distance >= 2.:
                return self._hold(depth, state, 'detour_distance_limit')
            # Reinspect before a gradually changing target bearing reaches the
            # unchanged hard view limit. A long fixed segment can otherwise
            # lose a still-observed target before its distance trigger fires.
            approaching_view_limit = abs(wrap_angle(self.heading-target_heading)) >= .23
            if distance >= self._last_rejoin_distance + .55 or approaching_view_limit:
                # A stopped hypothetical state only plans a rejoin. Actual
                # turning waits for measured near-zero velocity below.
                stopped = replace(state, velocity=np.zeros(3))
                rejoin = self._probe_heading(depth, stopped, target_heading, speed)
                if approaching_view_limit or rejoin['reason'] in ('clear', 'unknown_outside_field_of_view'):
                    self.phase = 'rejoining'
                    return self._result(depth, state, 0., state.yaw, 'stop_before_rejoining',
                                        detour_distance_m=distance)
            return self._result(depth, state,
                speed if abs(wrap_angle(self.heading-state.yaw)) <= .045 else 0., self.heading,
                'observed_detour', detour_distance_m=distance)
        if self.phase == 'rejoining':
            if np.linalg.norm(state.velocity) > .025:
                return self._result(depth, state, 0., state.yaw, 'stop_before_rejoining')
            if abs(wrap_angle(target_heading-state.yaw)) > .045:
                # Inspect in place. The current forward footprint is measured,
                # but a candidate farther ahead may not fit the old camera FOV.
                # No translation occurs until a new view certifies that route.
                return self._result(depth, state, 0., target_heading, 'turn_to_selected_target')
            direct = self._probe_heading(depth, state, target_heading, speed)
            if direct['reason'] in ('blocked_stopping_distance', 'unknown_occluded_corridor'):
                self._last_rejoin_distance = float(np.linalg.norm(state.position[:2]-self.origin[:2]))
                # Re-probe small headings from this newly observed position.
                # Retain the original time/distance budget across refinements.
                self.phase = 'braking'
                reason = 'target_route_still_blocked' if direct['reason'] == 'blocked_stopping_distance' else 'target_route_unobserved'
                return self._result(depth, state, 0., state.yaw, reason)
            if direct['reason'] != 'clear':
                return self._hold(depth, state, 'rejoin_' + direct['reason'])
            self.completed += 1
            self.reset()
            return self._result(depth, state, 0., target_heading, 'detour_complete')
        return self._result(depth, state, 0., state.yaw, 'holding')
