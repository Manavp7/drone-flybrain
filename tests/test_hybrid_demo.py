"""Acceptance scoring tests from complete, artificial telemetry fixtures.

These verify evaluator failure semantics; they do not claim neural inference,
detector accuracy, physical dynamics, or flight success.
"""
from copy import deepcopy
import unittest

from experiments.hybrid_demo import CONTROL_CONFIG, OBS_DT, SPECS, score_episode


def person_box(center_x=0., center_y=0., height=None):
    height = CONTROL_CONFIG['desired_height_fraction'] if height is None else height
    cx, cy = (center_x+1)*391/2, (center_y+1)*391/2
    h, w = height*391, height*391*.4
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def complete_trace(spec):
    """A complete fixture that approaches, centers, then holds or loses sight."""
    rows = []
    for i in range(round(spec['duration_s']/OBS_DT)):
        t = i*OBS_DT
        lost = spec['kind'] == 'loss' and t >= spec['hidden_after_s']-1e-9
        error = .25*max(0., 1-t/4)
        velocity = [0., 0., 0.] if (i+1)*OBS_DT >= 6-1e-9 else [.3, 0., 0.]
        rows.append(dict(sequence=i, capture_time_s=t,
            truth={'relative_depth_m': 3.5},
            person_truth_bbox=None if lost else person_box(error, -.4*error),
            target_observation={'valid': not lost, 'track_id': None if lost else 7},
            state_after={'time': (i+1)*OBS_DT, 'position': [min(t*.2, 1.2), 0., 1.],
                         'velocity': velocity},
            new_command={'velocity': [0., 0., 0.] if lost or t >= 6 else [.3, 0., 0.],
                         'issued_at_s': (i+1)*OBS_DT}))
    return rows


class HybridAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.normal = deepcopy(next(s for s in SPECS if s['name'] == 'approach_left'))
        self.loss = deepcopy(next(s for s in SPECS if s['name'] == 'target_loss'))

    def test_approaches_centers_and_holds_good_size_passes(self):
        result = score_episode(complete_trace(self.normal), self.normal)
        self.assertTrue(result['passed'])
        self.assertGreater(result['initial_center_norm'], .2)
        self.assertLess(result['tail_mean_center_norm'], 1e-12)
        self.assertLess(result['tail_mean_height_error'], 1e-12)

    def test_missed_target_braking_does_not_pass_normal_episode(self):
        rows = complete_trace(self.normal)
        for row in rows:
            row['target_observation'] = {'valid': False, 'track_id': None}
            row['state_after']['velocity'] = [0., 0., 0.]
            row['new_command']['velocity'] = [0., 0., 0.]
        # Even favorable simulator truth cannot replace actual observation.
        self.assertFalse(score_episode(rows, self.normal)['passed'])

    def test_wrong_final_size_fails_even_when_centered(self):
        rows = complete_trace(self.normal)
        for row in rows:
            if row['capture_time_s'] >= self.normal['duration_s']-2:
                row['person_truth_bbox'] = person_box(height=.15)
        self.assertFalse(score_episode(rows, self.normal)['passed'])

    def test_insufficient_separation_fails(self):
        rows = complete_trace(self.normal)
        rows[20]['truth']['relative_depth_m'] = 1.99
        self.assertFalse(score_episode(rows, self.normal)['passed'])

    def test_loss_after_observed_motion_and_settled_braking_passes(self):
        result = score_episode(complete_trace(self.loss), self.loss)
        self.assertTrue(result['passed'])
        self.assertAlmostEqual(result['first_loss_response_s'], 4.1)
        self.assertGreater(result['pre_loss_max_speed_m_s'], .1)

    def test_permanently_braked_loss_episode_fails(self):
        rows = complete_trace(self.loss)
        for row in rows:
            row['state_after']['velocity'] = [0., 0., 0.]
            row['new_command']['velocity'] = [0., 0., 0.]
        result = score_episode(rows, self.loss)
        self.assertTrue(result['all_loss_commands_brake'])
        self.assertFalse(result['passed'])

    def test_lost_target_commanding_motion_fails(self):
        rows = complete_trace(self.loss)
        rows[45]['new_command']['velocity'] = [.01, 0., 0.]
        self.assertFalse(score_episode(rows, self.loss)['passed'])

    def test_unsettled_velocity_after_loss_fails(self):
        rows = complete_trace(self.loss)
        rows[-1]['state_after']['velocity'] = [.11, 0., 0.]
        self.assertFalse(score_episode(rows, self.loss)['passed'])

    def test_missing_pre_loss_detections_fail(self):
        rows = complete_trace(self.loss)
        for row in rows[:20]:
            row['target_observation'] = {'valid': False, 'track_id': None}
        self.assertFalse(score_episode(rows, self.loss)['passed'])

    def test_initialization_only_is_rejected(self):
        rows = complete_trace(self.normal)[:1]
        with self.assertRaises(ValueError):
            score_episode(rows, self.normal)

    def test_initialization_and_one_good_tail_sample_is_rejected(self):
        rows = complete_trace(self.normal)
        with self.assertRaises(ValueError):
            score_episode([rows[0], rows[-1]], self.normal)

    def test_missing_observation_in_middle_is_rejected(self):
        rows = complete_trace(self.normal)
        del rows[50]
        with self.assertRaises(ValueError):
            score_episode(rows, self.normal)

    def test_duplicate_capture_with_full_count_is_rejected(self):
        rows = complete_trace(self.normal)
        rows[50]['capture_time_s'] = rows[49]['capture_time_s']
        with self.assertRaises(ValueError):
            score_episode(rows, self.normal)

    def test_missing_tail_truth_cannot_hide_behind_one_good_frame(self):
        rows = complete_trace(self.normal)
        for row in rows:
            if self.normal['duration_s']-2 <= row['capture_time_s'] < self.normal['duration_s']-.1-1e-9:
                row['person_truth_bbox'] = None
        self.assertFalse(score_episode(rows, self.normal)['passed'])

    def test_one_missing_tail_truth_also_fails_complete_tail_gate(self):
        rows = complete_trace(self.normal)
        rows[-1]['person_truth_bbox'] = None
        self.assertFalse(score_episode(rows, self.normal)['passed'])


if __name__ == '__main__':
    unittest.main()
