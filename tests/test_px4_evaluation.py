"""Evaluator receipts only: no PX4 process, model, renderer, or network."""
import copy
import json
import math
import unittest

from experiments.px4_evaluation import mission_metrics, mission_score, paired_summary, recovery_proof


BOX = [0., 0., 10., 10.]
OTHER = [20., 0., 30., 10.]


def observation(sequence=0, capture=100., *, actor='blue', valid=True,
                selected=True, gap=False, yaw=0., method='neural'):
    box = BOX if actor == 'blue' else OTHER
    packet = dict(session='test-session', sequence=sequence, capture_time_s=capture, completed_s=capture+.1,
        observation=dict(valid=valid, track_id=1, bbox_xyxy=box, sequence=sequence, capture_time_s=capture),
        selection=dict(track_id=1, held=False),
        candidate=dict(valid=valid and not gap, capture_time_s=capture,
                       track_id=1, surface_optical_z_m=None if gap else 4.),
        selection_action=dict(accepted=True, request=dict(track_id=1, sequence=sequence)) if selected else None,
        input_finite_depth_pixels=0 if gap else 100, stationary_association_memory=[],
        flyvis_used=method == 'neural', flyvis_observations=sequence+1 if method == 'neural' else 0,
        neural_valid=method == 'neural')
    command = dict(sequence=sequence, capture_time_s=capture, issued_at_s=capture+.1,
        valid_until_s=capture+.9, valid=valid and not gap, forward_speed=.2 if valid and not gap else 0.)
    evaluation = dict(sequence=sequence, capture_time_s=capture, source_time_s=capture-90,
        scenario_time_s=capture-100, trajectory='stationary', depth_drop=gap, pose_yaw=yaw,
        truth_position=[0., 0., 1.1], projections=[
            dict(actor_id='blue', visible=True, bbox_xyxy=BOX),
            dict(actor_id='orange', visible=True, bbox_xyxy=OTHER)])
    return dict(packet=packet, command=command, evaluation=evaluation)


def send(item, at=None, speed=.2, yaw=0.):
    command = item['command']
    at = command['capture_time_s']+.2 if at is None else at
    return dict(sent_at_s=at, elapsed_s=at-100., source_time_s=at-90.,
        position=[0., 0., 1.1], yaw=yaw, sent_speed=speed, requested_speed=speed,
        command_sequence=command['sequence'] if speed else None,
        command_expiry_s=command['valid_until_s'], depth_reason='clear' if speed else 'hold_clear',
        depth_age_s=.03, state_age_s=.02, tick_s=.025)


def recovery_fixture():
    first = observation(yaw=0.)
    gap = observation(1, 100.3, selected=False, gap=True, yaw=.03)
    gap['packet']['stationary_association_memory'] = [dict(track_id=1,
        retained_for_association_only=True, anchor_capture_time_s=100.,
        missing_depth_capture_time_s=100.3)]
    recovered = observation(2, 100.6, selected=False, yaw=.04)
    recovered['packet']['reprojections'] = [dict(track_id=1, valid=True,
        anchor_capture_time_s=100., previous_observation_time_s=100.3)]
    rows = [send(first), send(gap, 100.45, speed=0., yaw=.03),
            send(gap, 100.6, speed=0., yaw=.035), send(recovered, 100.8, yaw=.04)]
    return rows, [first, gap, recovered]


class MissionMetricsTests(unittest.TestCase):
    def metrics(self, rows=(), observations=(), end=101.):
        return mission_metrics(rows, observations, mission_start_s=100., mission_end_s=end)

    def test_completion_intervals_include_startup_and_result_expiry(self):
        item = observation()
        item['packet']['completed_s'] = item['command']['issued_at_s'] = 100.2
        result = self.metrics(observations=[item])
        self.assertEqual(result['tracking_loss_episodes'], 2)
        self.assertAlmostEqual(result['tracking_loss_duration_s'], .55)
        self.assertAlmostEqual(result['tracking_available_fraction'], .45)
        self.assertAlmostEqual(result['correct_identity_duration_fraction'], .45)
        self.assertAlmostEqual(result['latency_s']['p95'], .2)

    def test_invalid_result_starts_loss_at_completion_not_capture(self):
        items = [observation(), observation(1, 100.3, valid=False, selected=False),
                 observation(2, 100.5, selected=False)]
        result = self.metrics(observations=items, end=101.2)
        self.assertEqual(result['tracking_loss_episodes'], 3)
        self.assertAlmostEqual(result['tracking_loss_duration_s'], .35)
        self.assertAlmostEqual(result['command_valid_fraction'], 2/3)

    def test_mission_window_clips_intervals_and_ignores_future_binding(self):
        result = self.metrics(observations=[observation(capture=101.2)])
        self.assertEqual(result['observation_count'], 0)
        self.assertEqual(result['tracking_loss_duration_s'], 1.)
        self.assertIsNone(result['intended_actor'])
        with self.assertRaises(ValueError):
            mission_metrics([], [], mission_start_s=1., mission_end_s=1.)

    def test_explicit_selection_required_and_actor_binds_only_once(self):
        first = observation(selected=False)
        second = observation(1, 100.2)
        third = observation(2, 100.4, actor='orange', selected=False)
        result = self.metrics(observations=[first, second, third])
        self.assertEqual(result['intended_actor'], 'blue')
        self.assertEqual(result['identity_unknown_count'], 1)
        self.assertEqual(result['wrong_person_frames'], 1)
        self.assertEqual(result['wrong_person_episodes'], 1)
        self.assertAlmostEqual(result['identity_correct_fraction'], 1/3)

    def test_ambiguous_match_is_never_correct_or_initial_identity(self):
        first = observation()
        first['evaluation']['projections'][1]['bbox_xyxy'] = BOX
        result = self.metrics(observations=[first])
        self.assertIsNone(result['intended_actor'])
        self.assertEqual(result['identity_ambiguous_count'], 1)
        self.assertEqual(result['identity_unassessable_count'], 1)
        self.assertEqual(result['identity_correct_count'], 0)
        self.assertEqual(result['correct_identity_duration_fraction'], 0.)

    def test_selection_binds_original_detection_before_first_valid_guidance(self):
        first = observation(valid=False)
        first['packet']['detections'] = dict(detections=[dict(track_id=1, class_id=0, bbox_xyxy=BOX)])
        replaced = observation(1, 100.3, actor='orange', selected=False)
        result = self.metrics(observations=[first, replaced])
        self.assertEqual(result['intended_actor'], 'blue')
        self.assertEqual(result['wrong_person_frames'], 1)
        self.assertEqual(result['identity_correct_count'], 0)

    def test_unknown_initial_actor_never_binds_to_a_later_occupant(self):
        first = observation()
        first['evaluation']['projections'][1]['bbox_xyxy'] = BOX
        later = observation(1, 100.3, selected=False)
        result = self.metrics(observations=[first, later])
        self.assertIsNone(result['intended_actor'])
        self.assertEqual(result['identity_correct_count'], 0)

    def test_nested_observation_capture_and_sequence_must_match(self):
        for field, value in [('capture_time_s', 50.), ('sequence', 42)]:
            item = observation()
            item['packet']['observation'][field] = value
            with self.subTest(field=field):
                result = self.metrics([send(item)], [item])
                self.assertEqual(len(result['packet_integrity_errors']), 1)
                self.assertEqual(result['unsafe_positive_send_ticks'], 1)

    def test_annotations_require_exact_sequence_and_capture(self):
        for key, value in [('sequence', 7), ('capture_time_s', 100.00001)]:
            with self.subTest(key=key):
                item = observation()
                item['evaluation'][key] = value
                result = self.metrics(observations=[item])
                self.assertEqual(result['identity_annotation_mismatch_count'], 1)
                self.assertEqual(result['identity_correct_count'], 0)

    def test_wrong_person_episodes_are_separated_by_unknown(self):
        items = [observation(), observation(1, 100.15, actor='orange', selected=False),
            observation(2, 100.3, actor='orange', selected=False),
            observation(3, 100.45, valid=False, selected=False),
            observation(4, 100.6, actor='orange', selected=False)]
        result = self.metrics(observations=items)
        self.assertEqual(result['wrong_person_frames'], 3)
        self.assertEqual(result['wrong_person_episodes'], 2)

    def test_every_positive_send_requires_fresh_authority_depth_and_state(self):
        item = observation()
        baseline = send(item)
        self.assertEqual(self.metrics([baseline], [item])['unsafe_positive_send_ticks'], 0)
        changes = [('sent_at_s', 100.9), ('state_age_s', .101), ('depth_age_s', None),
            ('tick_s', .101), ('depth_reason', 'unknown_depth'), ('command_sequence', 4),
            ('command_expiry_s', 102.), ('sent_speed', .3)]
        for key, value in changes:
            with self.subTest(key=key):
                row = dict(baseline, **{key: value})
                self.assertEqual(self.metrics([row], [item])['unsafe_positive_send_ticks'], 1)

    def test_fresh_send_cannot_use_stale_or_mismatched_release(self):
        for alteration in ('stale', 'sequence', 'capture', 'candidate'):
            item = observation()
            if alteration == 'stale':
                item['packet']['completed_s'] = item['command']['issued_at_s'] = 100.7
            elif alteration == 'sequence':
                item['command']['sequence'] = 10
            elif alteration == 'capture':
                item['command']['capture_time_s'] = 99.9
            else:
                item['packet']['candidate']['track_id'] = 2
            row = send(item, 100.8)
            with self.subTest(alteration=alteration):
                self.assertEqual(self.metrics([row], [item])['unsafe_positive_send_ticks'], 1)

    def test_duplicate_packet_and_nonfinite_send_are_not_silent_passes(self):
        item = observation()
        result = self.metrics([dict(send(item), sent_speed=float('nan'))], [item, item])
        self.assertFalse(result['send_evidence_valid'])
        self.assertEqual(len(result['packet_integrity_errors']), 1)

    def test_session_change_and_capture_regression_invalidate_packet_evidence(self):
        first = observation()
        for key, value in [('session', 'another-session'), ('capture_time_s', 99.9)]:
            later = observation(1, 100.3, selected=False)
            later['packet'][key] = value
            with self.subTest(key=key):
                result = self.metrics(observations=[first, later])
                self.assertEqual(len(result['packet_integrity_errors']), 1)

    def test_explicit_maximum_speed_applies_even_if_command_claims_more(self):
        item = observation()
        item['command']['forward_speed'] = .7
        row = send(item, speed=.46)
        self.assertEqual(self.metrics([row], [item])['unsafe_positive_send_ticks'], 1)

    def test_yaw_unwrap_and_actual_flyvis_receipt_counts(self):
        rows, items = recovery_fixture()
        rows[0]['yaw'], rows[1]['yaw'] = 3.13, -3.13
        result = self.metrics(rows[:2], items)
        self.assertAlmostEqual(result['physical_yaw_span_rad'], 2*math.pi-6.26)
        self.assertEqual(result['flyvis_observation_calls'], 3)
        self.assertEqual(result['flyvis_guidance_commands'], 2)
        conventional = observation(method='direct')
        result = self.metrics([send(conventional)], [conventional])
        self.assertEqual(result['flyvis_observation_calls'], 0)
        self.assertEqual(result['flyvis_positive_send_ticks'], 0)

    def test_no_input_mutation_and_json_serializable_output(self):
        rows, items = recovery_fixture()
        before = copy.deepcopy((rows, items))
        json.dumps(self.metrics(rows, items), allow_nan=False)
        self.assertEqual((rows, items), before)


class RecoveryTests(unittest.TestCase):
    def proof(self, rows, observations):
        return recovery_proof(rows, observations, mission_start_s=100., mission_end_s=102.)

    def test_actual_gap_turn_hold_and_same_person_forward_recovery(self):
        rows, items = recovery_fixture()
        result = self.proof(rows, items)
        self.assertTrue(result['passed'], result)
        self.assertAlmostEqual(result['pose_turn_rad'], .03)
        self.assertEqual(result['gap_send_ticks'], 2)
        self.assertEqual(result['recovery_sequence'], 2)

    def test_memory_alone_cannot_prove_recovery(self):
        rows, items = recovery_fixture()
        for row in rows:
            row['sent_speed'] = row['requested_speed'] = 0.
        result = self.proof(rows, items)
        self.assertTrue(result['checks']['stationary_memory_used'])
        self.assertFalse(result['checks']['positive_recovery_send'])
        self.assertFalse(result['passed'])

    def test_each_required_piece_can_fail_independently(self):
        changes = {
            'injected_depth_gap': lambda rows, items: items[1]['packet'].update(input_finite_depth_pixels=1),
            'stationary_memory_used': lambda rows, items: items[1]['packet'].update(stationary_association_memory=[]),
            'pose_turn_at_least_002_rad': lambda rows, items: items[1]['evaluation'].update(pose_yaw=.01),
            'gap_command_invalid': lambda rows, items: items[1]['command'].update(valid=True),
            'gap_zero_forward': lambda rows, items: rows[1].update(sent_speed=.1),
            'evaluator_identity_preserved': lambda rows, items: items[2]['packet']['observation'].update(bbox_xyxy=OTHER),
            'positive_recovery_send': lambda rows, items: rows[-1].update(depth_age_s=.2),
            'restored_anchor_reprojected': lambda rows, items: items[-1]['packet'].update(reprojections=[]),
            'no_explicit_reselection': lambda rows, items: items[-1]['packet'].update(
                selection_action=dict(accepted=True, request=dict(track_id=1, sequence=2))),
        }
        for check, change in changes.items():
            rows, items = recovery_fixture()
            change(rows, items)
            with self.subTest(check=check):
                result = self.proof(rows, items)
                self.assertFalse(result['checks'][check], result)
                self.assertFalse(result['passed'])

    def test_late_recovery_fails_unextended_capture_deadline(self):
        rows, items = recovery_fixture()
        late = observation(2, 101.6, selected=False, yaw=.04)
        items[-1], rows[-1] = late, send(late, 101.8)
        result = self.proof(rows, items)
        self.assertTrue(result['checks']['positive_recovery_send'])
        self.assertFalse(result['checks']['recovered_within_1_2s'])

    def test_gap_deadline_cannot_extend_original_anchor_lifetime(self):
        rows, items = recovery_fixture()
        later = observation(2, 101.3, selected=False, yaw=.04)
        later['packet']['reprojections'] = items[-1]['packet']['reprojections']
        items[-1], rows[-1] = later, send(later, 101.45)
        result = self.proof(rows, items)
        self.assertTrue(result['checks']['recovered_within_1_2s'])
        self.assertFalse(result['checks']['original_anchor_fresh_at_recovery'])
        self.assertFalse(result['passed'])

    def test_no_gap_and_multiple_gaps_are_unproven(self):
        rows, items = recovery_fixture()
        items[1]['evaluation']['depth_drop'] = False
        self.assertFalse(self.proof(rows, items)['passed'])
        items[1]['evaluation']['depth_drop'] = True
        items[2]['evaluation']['depth_drop'] = True
        self.assertEqual(self.proof(rows, items)['gap_count'], 2)

    def test_later_positive_valid_result_can_complete_recovery(self):
        rows, items = recovery_fixture()
        rows[-1]['sent_speed'] = rows[-1]['requested_speed'] = 0.
        later = observation(3, 100.9, selected=False, yaw=.04)
        items.append(later)
        rows.append(send(later, 101.1))
        self.assertTrue(self.proof(rows, items)['passed'])


class ScoreAndComparisonTests(unittest.TestCase):
    events = [dict(event=name) for name in ('stable_takeoff', 'mission_window_complete', 'landed_disarmed')]

    def test_safe_hold_is_not_successful_tracking(self):
        row = send(observation(), speed=0.)
        result = mission_score([row], [], self.events,
            mission_start_s=100., mission_end_s=101., min_tracking_fraction=.8)
        self.assertTrue(result['safety_passed'])
        self.assertFalse(result['tracking_success'])
        self.assertFalse(result['passed'])

    def test_explicit_policy_and_observed_correct_identity_time_required(self):
        items = [observation(), observation(1, 100.5, selected=False)]
        rows = [send(i) for i in items]
        result = mission_score(rows, items, self.events, mission_start_s=100., mission_end_s=101.)
        self.assertIsNone(result['tracking_success'])
        self.assertFalse(result['passed'])
        result = mission_score(rows, items, self.events, mission_start_s=100., mission_end_s=101.,
                               min_tracking_fraction=.8)
        self.assertTrue(result['passed'], result)
        items[1]['evaluation']['projections'][1]['bbox_xyxy'] = BOX
        result = mission_score(rows, items, self.events, mission_start_s=100., mission_end_s=101.,
                               min_tracking_fraction=.8)
        self.assertFalse(result['passed'])

    def test_failed_or_incomplete_mission_is_preserved(self):
        item = observation()
        result = mission_score([send(item)], [item], self.events+[dict(event='failure')],
            mission_start_s=100., mission_end_s=100.5, min_tracking_fraction=.8)
        self.assertFalse(result['safety_passed'])
        result = mission_score([send(item)], [item], self.events[:1],
            mission_start_s=100., mission_end_s=100.5, min_tracking_fraction=.8)
        self.assertFalse(result['checks']['mission_window_complete'])

    def run_record(self, method='neural', repeat=0, config=None, passed=True):
        return dict(method=method, repeat=repeat, config=config or dict(trajectory='walk', duration_s=30),
            actual_px4=True, summary=dict(passed=passed,
                checks=dict(mission_window_complete=True, no_runtime_failure=True),
                metrics=dict(command_valid_fraction=.8, tracking_loss_duration_s=2., intended_actor='blue')))

    def test_paired_comparison_retains_failure_and_makes_no_superiority_claim(self):
        runs = [self.run_record(), self.run_record('direct', passed=False), self.run_record('filtered')]
        result = paired_summary(runs)
        self.assertTrue(result['pairs'][0]['comparable'])
        self.assertFalse(result['runs'][1]['summary']['passed'])
        self.assertEqual(result['methods']['direct']['failed_or_unqualified'], 1)
        self.assertFalse(result['superiority_established'])
        self.assertIsNone(result['superiority_claim'])
        result['runs'][0]['summary']['passed'] = False
        self.assertTrue(runs[0]['summary']['passed'])

    def test_incomplete_or_failed_runtime_exposure_cannot_supply_paired_deltas(self):
        for checks in (None, {}, dict(mission_window_complete=False, no_runtime_failure=True),
                       dict(mission_window_complete=True, no_runtime_failure=False),
                       dict(mission_window_complete=1, no_runtime_failure=True)):
            runs = [self.run_record(m) for m in ('neural', 'direct', 'filtered')]
            if checks is None:
                runs[1]['summary'].pop('checks')
            else:
                runs[1]['summary']['checks'] = checks
            with self.subTest(checks=checks):
                result = paired_summary(runs)
                pair = result['pairs'][0]
                self.assertTrue(pair['complete'])
                self.assertTrue(pair['same_intended_actor'])
                self.assertFalse(pair['comparable'])
                self.assertEqual(pair['neural_minus_baseline'], {})
                self.assertEqual(len(result['runs']), 3)

    def test_incomplete_pair_shape_is_unchanged_by_window_qualification(self):
        runs = [self.run_record(), self.run_record('direct')]
        original = paired_summary(runs)
        for run in runs:
            run['summary']['checks'].update(mission_window_complete=False, no_runtime_failure=False)
        changed = paired_summary(runs)
        self.assertEqual(changed['pairs'], original['pairs'])
        self.assertEqual(changed['methods'], original['methods'])
        self.assertEqual(changed['interpretation'], original['interpretation'])

    def test_mismatched_config_repeat_duplicates_and_synthetic_runs_not_paired(self):
        sets = [
            [self.run_record(), self.run_record('direct', repeat=1), self.run_record('filtered')],
            [self.run_record(), self.run_record('direct', config={'trajectory': 'crossing'}), self.run_record('filtered')],
            [self.run_record(), self.run_record(), self.run_record('direct'), self.run_record('filtered')],
        ]
        synthetic = [self.run_record(m) for m in ('neural', 'direct', 'filtered')]
        synthetic[1]['actual_px4'] = False
        sets.append(synthetic)
        for runs in sets:
            with self.subTest(runs=runs):
                result = paired_summary(runs)
                self.assertFalse(any(p['comparable'] for p in result['pairs']))
                self.assertEqual(len(result['runs']), len(runs))

    def test_invalid_receipts_remain_visible(self):
        result = paired_summary([dict(method='neural', summary=None)])
        self.assertEqual(result['invalid_run_indices'], [0])
        self.assertEqual(result['methods']['neural']['failed_or_unqualified'], 1)

    def test_different_or_unknown_physical_targets_cannot_supply_paired_deltas(self):
        for actor, issue in [('orange', 'intended_actor_mismatch'), (None, 'intended_actor_unknown'),
                             ('', 'intended_actor_unknown')]:
            runs = [self.run_record(m) for m in ('neural', 'direct', 'filtered')]
            runs[1]['summary']['metrics']['intended_actor'] = actor
            with self.subTest(actor=actor):
                pair = paired_summary(runs)['pairs'][0]
                self.assertTrue(pair['complete'])
                self.assertTrue(pair['actual_px4_evidence'])
                self.assertFalse(pair['comparable'])
                self.assertEqual(pair['neural_minus_baseline'], {})
                self.assertEqual(pair['target_comparison_issue'], issue)


if __name__ == '__main__':
    unittest.main()
