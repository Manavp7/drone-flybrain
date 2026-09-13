"""Reporting fixtures expose false success from missing, lost or future data."""
from copy import deepcopy
import unittest

import numpy as np

from experiments.flight_report import latest_completed, score_records, suite_coverage


def fixture(duration=10., kind='normal'):
    event = 3.5
    spec = dict(name='test', kind=kind, duration_s=duration, target=[5., 0., .95])
    if kind not in ('normal', 'ablation'):
        spec['event_s'] = event

    def state(i):
        t = i*.005
        stop = event+.5 if kind not in ('normal', 'ablation') else duration+1.
        velocity = .16 if .4 <= t < stop and kind != 'ablation' else 0.
        x = max(0., min(t, stop)-.4)*.16 if kind != 'ablation' else 0.
        return dict(time_s=t, position=[x, 0., 1.1], velocity=[velocity, 0., 0.],
                    rotation=np.eye(3).tolist(), angular_velocity=[0., 0., 0.], motor_forces=[1.962]*4)

    observations = []
    for sequence, capture in enumerate(np.arange(0., duration-1e-9, .4)):
        capture = round(float(capture), 8)
        complete = round(capture+.4, 8)
        hidden = kind == 'target_loss' and capture >= event
        valid = not hidden
        command_valid = valid and kind != 'ablation'
        reason = 'target_lost' if hidden else 'neural_evidence_unavailable' if kind == 'ablation' else 'neural_bearing_and_registered_depth'
        observation = dict(valid=valid, bbox_xyxy=[100., 100., 200., 300.] if valid else None,
                           center_normalized=[0., 0.] if valid else None, track_id=1)
        candidate = dict(sequence=sequence, capture_time_s=capture, issued_at_s=complete,
                         valid_until_s=capture+.9, forward_speed=.16 if command_valid else 0.,
                         yaw_target=0., valid=command_valid, reason=reason)
        observations.append(dict(sequence=sequence, capture_time_s=capture, completed_time_s=complete,
            inference_wall_s=.4, injected_delay_s=0., observation=observation,
            candidate=candidate, target_truth_bbox=[100., 100., 200., 300.] if valid else None,
            target_surface_optical_z_m=3.5 if valid else None,
            neural_valid=True, neural_stimulus_time_s=sequence*.1, neural_response_time_s=(sequence+1)*.1))
    ticks = []
    completions = np.array([r['completed_time_s'] for r in observations])
    for i in range(round(duration/.005)):
        t = i*.005
        k = int(np.searchsorted(completions, t+1e-9, side='right')-1)
        candidate = observations[k]['candidate'] if k >= 0 else None
        active = candidate is not None and candidate['valid'] and t < candidate['valid_until_s']-1e-9
        speed = candidate['forward_speed'] if active else 0.
        guardian_speed = 0. if kind not in ('normal', 'ablation') and t >= event else speed
        guardian_reason = 'clear'
        if kind == 'depth_loss' and t >= event:
            guardian_reason = 'missing_depth'
        if kind == 'stale_depth' and t >= event:
            guardian_reason = 'stale_depth'
        if kind == 'obstacle_stop' and t >= event:
            guardian_reason = 'blocked_stopping_distance'
        truth = dict(time_s=(i+1)*.005, contacts=[], contact_count=0,
                     obstacle_enabled=kind == 'obstacle_stop' and t >= event,
                     obstacle_position=[2.5, 0., .7])
        ticks.append(dict(index=i, time_s=t, state_before=state(i), state_after=state(i+1),
            motor_targets=[1.962]*4, truth_after=truth,
            request=dict(forward_speed=speed, yaw_target=0., sequence=k if active else None,
                         reason='neural_bearing_and_registered_depth' if active else 'no_fresh_guidance'),
            guardian=dict(forward_speed=guardian_speed, reason=guardian_reason),
            applied_command_sequence=k if active else None))
    episode = dict(status='completed', physics_ticks=len(ticks), observations=len(observations),
                   final_state=deepcopy(ticks[-1]['state_after']), whole_wall_s=12.)
    return spec, ticks, observations, episode


class FlightReportingTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.ticks, self.rows, self.episode = fixture()

    def score(self):
        return score_records(self.spec, self.ticks, self.rows, self.episode)

    def test_complete_moving_fixture_passes(self):
        report = self.score()
        self.assertTrue(report['passed'], report['failures'])
        self.assertEqual(report['metrics']['active_overlap_fraction'], 1.)
        self.assertAlmostEqual(report['metrics']['horizontal_translation_m'], 1.536)

    def test_omitted_last_tick_and_receipt_cannot_pass(self):
        self.ticks.pop()
        report = self.score()
        self.assertFalse(report['gates']['complete_tick_count'])
        self.assertFalse(report['gates']['episode_receipt_matches_trace'])
        self.assertFalse(report['passed'])

    def test_duplicated_or_misclocked_tick_fails(self):
        self.ticks[20]['time_s'] = self.ticks[19]['time_s']
        self.assertFalse(self.score()['gates']['complete_tick_clock'])

    def test_state_jump_cannot_hide_behind_small_bounded_position(self):
        self.ticks[10]['state_after']['position'][1] += .01
        self.assertFalse(self.score()['gates']['state_continuity'])

    def test_hover_cannot_count_as_normal_flight_success(self):
        for tick in self.ticks:
            for name in ('state_before', 'state_after'):
                tick[name]['position'] = [0., 0., 1.1]
                tick[name]['velocity'] = [0., 0., 0.]
        self.episode['final_state'] = deepcopy(self.ticks[-1]['state_after'])
        report = self.score()
        self.assertFalse(report['gates']['normal_genuine_translation'])
        self.assertFalse(report['passed'])

    def test_lost_box_remains_in_overlap_denominator(self):
        for row in self.rows[-7:]:
            row['observation']['valid'] = False
        report = self.score()
        self.assertEqual(report['metrics']['active_overlap_hits'], 18)
        self.assertEqual(report['metrics']['observation_count'], 25)
        self.assertFalse(report['gates']['normal_approximate_photo_overlap'])
        self.assertFalse(report['gates']['normal_tail_centered'])

    def test_changed_identity_fails_even_with_perfect_overlap(self):
        self.rows[-1]['observation']['track_id'] = 2
        report = self.score()
        self.assertFalse(report['gates']['fixed_target_identity'])
        self.assertEqual(report['metrics']['active_overlap_hits'], 0)

    def test_missing_tail_never_earns_centering(self):
        self.rows = self.rows[:-5]
        report = self.score()
        self.assertFalse(report['gates']['normal_complete_tail_observations'])
        self.assertFalse(report['passed'])

    def test_missing_surface_truth_not_replaced_by_estimate(self):
        self.rows[-1]['target_surface_optical_z_m'] = None
        self.rows[-1]['estimate'] = {'surface_optical_z_m': 3.5}
        self.assertFalse(self.score()['gates']['normal_tail_surface_standoff'])

    def test_future_command_cannot_be_applied(self):
        tick = self.ticks[40]
        tick['applied_command_sequence'] = 0
        tick['request'].update(sequence=0, forward_speed=.16)
        tick['guardian']['forward_speed'] = .16
        self.assertFalse(self.score()['gates']['no_future_or_expired_command_application'])

    def test_expired_command_cannot_be_applied(self):
        tick = self.ticks[180]
        tick['applied_command_sequence'] = 0
        tick['request'].update(sequence=0, forward_speed=.16)
        self.assertFalse(self.score()['gates']['no_future_or_expired_command_application'])

    def test_capture_expiry_cannot_be_extended_to_completion(self):
        self.rows[0]['candidate']['valid_until_s'] = 1.3
        self.assertFalse(self.score()['gates']['capture_anchored_candidates'])

    def test_discarded_final_result_is_tracking_only(self):
        row = self.rows[-1]
        row.update(inference_wall_s=1.2, completed_time_s=10.8, candidate=None,
                   discarded_after_episode=True)
        report = self.score()
        self.assertTrue(report['passed'], report['failures'])
        self.assertEqual(report['metrics']['discarded_after_episode_count'], 1)
        self.assertNotEqual(latest_completed(self.rows, 11.)['sequence'], row['sequence'])

    def test_late_result_cannot_smuggle_candidate(self):
        self.rows[-1].update(inference_wall_s=1.2, completed_time_s=10.8, discarded_after_episode=True)
        self.assertFalse(self.score()['gates']['capture_anchored_candidates'])

    def test_completion_must_include_measured_delay(self):
        self.rows[4]['inference_wall_s'] = .5
        self.assertFalse(self.score()['gates']['causal_observation_clock'])

    def test_force_contact_and_altitude_checks_fail_independently(self):
        self.ticks[12]['motor_targets'][1] = 5.01
        self.ticks[12]['truth_after'].update(contact_count=1, contacts=[{'geom1': 'fuselage'}])
        self.ticks[12]['state_after']['position'][2] = 1.23
        report = self.score()
        self.assertFalse(report['gates']['bounded_motor_forces'])
        self.assertFalse(report['gates']['no_aircraft_contacts'])
        self.assertFalse(report['gates']['bounded_altitude'])

    def test_guardian_cannot_create_forward_authority(self):
        self.ticks[0]['guardian']['forward_speed'] = .1
        self.assertFalse(self.score()['gates']['depth_gate_never_increases_forward_request'])

    def test_missing_contact_truth_is_not_no_contact(self):
        del self.ticks[12]['truth_after']['contacts']
        self.assertFalse(self.score()['gates']['no_aircraft_contacts'])

    def test_ablation_expected_hover_is_labelled_separately(self):
        report = score_records(*fixture(8., 'ablation'))
        self.assertTrue(report['passed'], report['failures'])
        self.assertIn('absence of pursuit', report['outcome_scope'])

    def test_actual_motion_before_fault_then_stop_passes(self):
        for kind in ('target_loss', 'depth_loss', 'stale_depth', 'obstacle_stop'):
            with self.subTest(kind=kind):
                report = score_records(*fixture(8., kind))
                self.assertTrue(report['passed'], report['failures'])

    def test_always_hovering_event_case_fails(self):
        spec, ticks, rows, episode = fixture(8., 'depth_loss')
        for tick in ticks:
            for key in ('state_before', 'state_after'):
                tick[key]['position'] = [0., 0., 1.1]
                tick[key]['velocity'] = [0., 0., 0.]
        episode['final_state'] = deepcopy(ticks[-1]['state_after'])
        report = score_records(spec, ticks, rows, episode)
        self.assertFalse(report['gates']['event_prefault_genuine_motion'])

    def test_missing_fault_evidence_cannot_pass_by_stopping(self):
        spec, ticks, rows, episode = fixture(8., 'depth_loss')
        for tick in ticks:
            tick['guardian']['reason'] = 'clear'
        self.assertFalse(score_records(spec, ticks, rows, episode)['gates']['declared_fault_observed'])

    def test_late_tail_motion_is_not_hidden_by_first_settled_tick(self):
        spec, ticks, rows, episode = fixture(8., 'depth_loss')
        ticks[-10]['state_after']['velocity'][0] = .081
        self.assertFalse(score_records(spec, ticks, rows, episode)['gates']['event_settled_speed'])

    def test_preview_never_shows_uncompleted_result(self):
        self.assertIsNone(latest_completed(self.rows, .399))
        self.assertEqual(latest_completed(self.rows, .4)['sequence'], 0)
        self.assertEqual(latest_completed(self.rows, .799)['sequence'], 0)
        self.assertEqual(latest_completed(self.rows, .8)['sequence'], 1)

    def test_empty_trace_cannot_score_as_success(self):
        report = score_records(self.spec, [], [], {})
        self.assertFalse(report['passed'])
        self.assertFalse(report['gates']['complete_tick_count'])
        self.assertFalse(report['gates']['normal_complete_tail_observations'])

    def test_failed_execution_receipt_cannot_count_as_complete(self):
        self.episode['status'] = 'failed'
        self.assertFalse(self.score()['gates']['episode_receipt_matches_trace'])


class SuiteCoverageTests(unittest.TestCase):
    def setUp(self):
        self.specs = [dict(name='left', kind='normal', duration_s=10.),
                      dict(name='loss', kind='target_loss', duration_s=8., event_s=3.5)]
        self.definition = dict(specs=deepcopy(self.specs), development=False)
        self.execution = dict(cases=['left', 'loss'], complete=True, development=False)

    def test_exact_registered_suite_complete(self):
        self.assertTrue(suite_coverage(self.definition, self.execution, self.specs)['passed'])

    def test_missing_case_cannot_make_suite_pass(self):
        report = suite_coverage(self.definition, self.execution, self.specs[:1])
        self.assertFalse(report['passed'])
        self.assertEqual(report['missing_cases'], ['loss'])

    def test_changed_duration_cannot_replace_frozen_case(self):
        self.specs[0]['duration_s'] = 3.
        self.assertFalse(suite_coverage(self.definition, self.execution, self.specs)['passed'])

    def test_duplicated_case_cannot_replace_missing_case(self):
        self.assertFalse(suite_coverage(self.definition, self.execution, [self.specs[0], self.specs[0]])['passed'])

    def test_missing_or_incomplete_execution_receipt_fails(self):
        for receipt in ({}, dict(self.execution, complete=False)):
            self.assertFalse(suite_coverage(self.definition, receipt, self.specs)['passed'])

    def test_unregistered_cases_cannot_pass(self):
        self.assertFalse(suite_coverage({}, self.execution, self.specs)['passed'])

    def test_development_subset_is_labelled_and_matches_its_own_plan(self):
        definition = dict(specs=self.specs[:1], development=True)
        execution = dict(cases=['left'], complete=True, development=True)
        result = suite_coverage(definition, execution, self.specs[:1])
        self.assertTrue(result['passed'])
        self.assertTrue(result['development'])


if __name__ == '__main__':
    unittest.main()
