"""Observation-only click selection, ambiguity holds and explicit recovery."""
import copy
import unittest

import numpy as np

from experiments.flight_contracts import CameraFrame
from experiments.mantis_selection import SelectionGuard


def person(track_id, box=(10, 10, 40, 90)):
    return dict(track_id=track_id, bbox_xyxy=list(box), class_id=0,
                label='person', confidence=.9)


def sample(sequence, people, timestamp=None):
    return dict(sequence=sequence, capture_time_s=sequence*.1 if timestamp is None else timestamp,
                stream_id='camera', clock_domain='sim', frame_id='front', status='ok',
                inference_executed=True, detections=people, age_at_finish_s=0.)


def deadline_sample(sequence, capture, finish=1., reason='inference_deadline_missed'):
    return dict(sample(sequence, [], capture), status='rejected', reason=reason,
                control_authority=False, age_at_start_s=0., age_at_finish_s=finish,
                processing_ms=finish*1000., received_monotonic_s=100.+capture,
                tracking_memory=dict(preserved_after_deadline=True,
                    observation_timestamps_renewed=False, control_authority=False,
                    restored_track_count=2, reason=reason,
                    age_checked_at_capture_time_s=capture+finish, max_observed_memory_age_s=3.))


def frame(timestamp, people, colors=None):
    rgb = np.full((100, 140, 3), 100, dtype=np.uint8)
    for p in people:
        x0, y0, x1, y1 = map(int, p['bbox_xyxy'])
        rgb[y0:y1, x0:x1] = (colors or {}).get(p['track_id'], (20, 65, 180))
    return CameraFrame(rgb, np.full((100, 140), 5., np.float32), (100., 100., 70., 50.),
                       np.eye(3), np.zeros(3), timestamp)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.guard = SelectionGuard()
        self.people = [person(1), person(2, (90, 10, 120, 90))]
        self.colors = {1: (20, 65, 180), 2: (210, 70, 20)}

    def update(self, sequence, people=None, colors=None, **changes):
        people = self.people if people is None else people
        result = sample(sequence, people)
        result.update(changes)
        self.guard.observe_frame(frame(result['capture_time_s'], people, self.colors if colors is None else colors))
        return self.guard.update(result, result['capture_time_s'], (100, 140))

    def select(self):
        self.update(0)
        self.guard.select(1, 0)

    def test_never_automatically_selects_a_visible_person(self):
        self.assertFalse(self.update(0)['valid'])
        self.assertIsNone(self.guard.track_id)
        self.assertEqual(self.guard.status()['reason'], 'selection_required')
        self.assertEqual(len(self.guard.status()['people']), 2)

    def test_click_binds_observed_track_and_next_fresh_frame_controls(self):
        self.select()
        selected = self.update(1)
        self.assertTrue(selected['valid'])
        self.assertEqual(selected['track_id'], 1)
        self.assertAlmostEqual(selected['appearance_similarity'], 1.)
        self.assertFalse(selected['control_authority'])

    def test_unknown_stale_and_nonfinite_clicks_rejected(self):
        self.update(0)
        for tid, sequence, now in [(7, 0, 0.), (1, 3, 0.), (1, 0, .5),
                                   (1, 0, -.1), (1, 0, float('nan')), (True, 0, 0.)]:
            with self.subTest(tid=tid, sequence=sequence, now=now):
                with self.assertRaises(ValueError):
                    self.guard.select(tid, sequence, now)
        self.assertIsNone(self.guard.track_id)

    def test_old_click_cannot_select_a_new_occupant(self):
        self.update(0)
        self.update(1)
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.guard.select(1, 0)

    def test_overlap_stops_and_does_not_resume_automatically(self):
        self.select()
        crossed = [person(1, (40, 10, 75, 90)), person(2, (55, 10, 90, 90))]
        out = self.update(1, crossed)
        self.assertFalse(out['valid'])
        self.assertEqual(out['reason'], 'ambiguous_people_overlap')
        self.assertFalse(self.update(2)['valid'])
        self.assertEqual(self.guard.track_id, 1)
        self.guard.select(1, 2)
        self.assertTrue(self.update(3)['valid'])

    def test_clicking_overlap_is_rejected(self):
        crossed = [person(1, (40, 10, 75, 90)), person(2, (55, 10, 90, 90))]
        self.update(0, crossed)
        self.assertFalse(any(p['selectable'] for p in self.guard.status()['people']))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            self.guard.select(1, 0)

    def test_selected_track_loss_does_not_switch_to_remaining_person(self):
        self.select()
        lost = self.update(1, [self.people[1]])
        self.assertEqual(lost['reason'], 'selected_person_lost')
        self.assertEqual(lost['track_id'], 1)
        self.assertFalse(self.update(2)['valid'])

    def test_explicit_reselection_can_choose_another_person(self):
        self.select()
        self.update(1, [self.people[1]])
        self.guard.select(2, 1)
        observed = self.update(2, [self.people[1]])
        self.assertTrue(observed['valid'])
        self.assertEqual(observed['track_id'], 2)

    def test_same_track_id_cannot_silently_change_clothing(self):
        self.select()
        swapped = {1: self.colors[2], 2: self.colors[1]}
        out = self.update(1, colors=swapped)
        self.assertFalse(out['valid'])
        self.assertEqual(out['reason'], 'selected_appearance_changed')
        self.assertFalse(self.update(2)['valid'])

    def test_head_box_jitter_and_trouser_fraction_do_not_replace_clothing(self):
        from perception.pipeline import clothing_histogram
        rgb = np.full((100, 140, 3), 110, dtype=np.uint8)
        rgb[10:25, 10:40] = (155, 98, 62)
        rgb[25:55, 10:40] = (30, 65, 145)
        rgb[55:90, 10:40] = (20, 20, 20)
        clipped = person(1, (10, 29, 40, 90))
        full = person(1, (10, 10, 40, 90))
        # Reproduces the old failure mechanism: a head/box correction changes
        # clothing/background proportions enough to fail histogram intersection.
        old_anchor = clothing_histogram(rgb, clipped['bbox_xyxy'])
        old_current = clothing_histogram(rgb, full['bbox_xyxy'])
        self.assertLess(float(np.minimum(old_anchor, old_current).sum()), .65)
        for sequence, detection in [(0, clipped), (1, full), (2, clipped), (3, full)]:
            camera = frame(sequence*.1, [])
            camera.rgb[:] = rgb
            self.guard.observe_frame(camera)
            observed = self.guard.update(sample(sequence, [detection]), sequence*.1, (100, 140))
            if sequence == 0:
                self.guard.select(1, 0)
            else:
                self.assertTrue(observed['valid'], observed)
                self.assertGreater(observed['appearance_similarity'], .9)

    def test_lighting_change_preserves_fixed_colour_anchor(self):
        self.select()
        for sequence, blue in [(1, (12, 39, 108)), (2, (28, 91, 252)), (3, (20, 65, 180))]:
            observed = self.update(sequence, colors={1: blue, 2: self.colors[2]})
            self.assertTrue(observed['valid'], observed)
            self.assertGreater(observed['appearance_similarity'], .99)
        self.assertFalse(self.update(4, colors={1: self.colors[2], 2: self.colors[1]})['valid'])

    def test_shared_neutral_background_cannot_hide_changed_coloured_clothing(self):
        for sequence, colour in [(0, (20, 65, 180)), (1, (210, 70, 20))]:
            camera = frame(sequence*.1, [])
            # Most of the detector box is the identical neutral background;
            # only a narrow observed torso supplies distinctive clothing.
            camera.rgb[20:60, 22:28] = colour
            self.guard.observe_frame(camera)
            observed = self.guard.update(sample(sequence, [self.people[0]]), sequence*.1, (100, 140))
            if sequence == 0:
                self.guard.select(1, 0)
            else:
                self.assertFalse(observed['valid'])
                self.assertEqual(observed['reason'], 'selected_appearance_changed')
                self.assertLess(self.guard.status()['appearance_similarity'], .1)

    def test_neutral_clothing_has_own_descriptor_and_cannot_become_chromatic(self):
        self.update(0, colors={1: (100, 100, 100), 2: self.colors[2]})
        self.guard.select(1, 0)
        observed = self.update(1, colors={1: (110, 110, 110), 2: self.colors[2]})
        self.assertTrue(observed['valid'])
        self.assertFalse(self.update(2)['valid'])

    def test_merged_detection_during_crossing_is_ambiguous(self):
        self.select()
        merged = [person(1, (10, 10, 120, 90))]
        out = self.update(1, merged)
        self.assertEqual(out['reason'], 'ambiguous_person_occlusion')
        self.assertFalse(out['valid'])

    def test_no_current_image_cannot_borrow_old_appearance(self):
        self.select()
        out = self.guard.update(sample(1, self.people), .1, (100, 140))
        self.assertFalse(out['valid'])
        self.assertEqual(out['reason'], 'current_appearance_unavailable')

    def test_stale_and_duplicate_frames_hold_selection(self):
        for bad in ('stale', 'duplicate'):
            with self.subTest(bad=bad):
                self.setUp()
                self.select()
                result = sample(1 if bad == 'stale' else 0, self.people)
                self.guard.observe_frame(frame(result['capture_time_s'], self.people, self.colors))
                out = self.guard.update(result, 1. if bad == 'stale' else 0., (100, 140))
                self.assertFalse(out['valid'])
                self.assertTrue(self.guard.status()['held'])
                self.assertEqual(self.guard.status()['people'], [])

    def test_changed_stream_cannot_be_overridden_with_click(self):
        self.select()
        self.assertFalse(self.update(1, stream_id='new_camera')['valid'])
        self.update(2)
        with self.assertRaises(ValueError):
            self.guard.select(1, 2)

    def test_clear_never_reselects_automatically(self):
        self.select()
        self.guard.clear()
        self.assertFalse(self.update(1)['valid'])
        self.assertIsNone(self.guard.track_id)

    def test_malformed_detection_cannot_remain_clickable(self):
        self.update(0)
        bad = sample(1, [person(1, (10, 10, float('nan'), 90))])
        self.assertFalse(self.guard.update(bad, .1, (100, 140))['valid'])
        self.assertEqual(self.guard.status()['people'], [])
        with self.assertRaises(ValueError):
            self.guard.select(1, 1)

    def test_thresholds_require_finite_numbers(self):
        for value in (True, float('nan'), float('inf'), 0, -1, 1.1, 'bad'):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    SelectionGuard(overlap_threshold=value)

    def timeout(self, result=None):
        result = deadline_sample(1, .1) if result is None else result
        return self.guard.update(result, result['capture_time_s'], (100, 140), .65)

    def test_verified_deadline_has_no_observation_then_same_current_person_recovers(self):
        for reason in ('inference_deadline_missed', 'processing_deadline_missed'):
            with self.subTest(reason=reason):
                self.setUp()
                self.select()
                anchor = self.guard._anchor['histogram'].copy()
                rejected = self.timeout(deadline_sample(1, .1, reason=reason))
                self.assertFalse(rejected['valid'])
                self.assertFalse(rejected['control_authority'])
                self.assertIsNone(rejected['bbox_xyxy'])
                status = self.guard.status()
                self.assertTrue(status['held'])
                self.assertTrue(status['recovery_pending'])
                self.assertFalse(status['selection_required'])
                self.assertEqual(status['people'], [])
                self.assertIsNone(status['capture_time_s'])
                self.assertIsNone(status['sequence'])
                with self.assertRaises(ValueError):
                    self.guard.select(1, 0, now_s=1.1)
                recovered = self.update(2, capture_time_s=1.2)
                self.assertTrue(recovered['valid'], recovered)
                self.assertEqual(recovered['track_id'], 1)
                self.assertFalse(self.guard.status()['recovery_pending'])
                np.testing.assert_array_equal(self.guard._anchor['histogram'], anchor)

    def test_repeated_deadlines_cannot_renew_the_last_observed_time(self):
        self.select()
        for receipt in (deadline_sample(1, .1), deadline_sample(2, 1.2, .8), deadline_sample(3, 2.1, .8)):
            self.assertFalse(self.timeout(receipt)['valid'])
            self.assertTrue(self.guard.status()['recovery_pending'])
            self.assertEqual(self.guard._last_accepted_capture, 0.)
        expired = self.update(4, capture_time_s=3.01)
        self.assertFalse(expired['valid'])
        self.assertEqual(expired['reason'], 'deadline_recovery_window_expired')
        self.assertFalse(self.update(5, capture_time_s=3.1)['valid'])

    def test_deadline_completion_itself_cannot_outlive_private_memory(self):
        self.select()
        self.assertFalse(self.timeout(deadline_sample(1, .1, 3.))['valid'])
        self.assertFalse(self.guard.status()['recovery_pending'])
        self.assertTrue(self.guard.status()['selection_required'])
        self.assertFalse(self.update(2, capture_time_s=3.2)['valid'])

    def test_untrusted_deadline_fields_cannot_enable_recovery(self):
        changes = [({'reason': 'stale_camera_frame'}, {}), ({'status': 'timing_unverified'}, {}),
                   ({'inference_executed': False}, {}),
                   ({'detections': [person(1)]}, {}), ({'control_authority': True}, {}),
                   ({'age_at_finish_s': float('nan')}, {}), ({'age_at_start_s': .7}, {}),
                   ({'processing_ms': -1}, {}), ({'processing_ms': 500.}, {}),
                   ({'received_monotonic_s': None}, {}),
                   ({}, {'preserved_after_deadline': False}),
                   ({}, {'observation_timestamps_renewed': True}),
                   ({}, {'control_authority': True}), ({}, {'restored_track_count': 0}),
                   ({}, {'restored_track_count': True}), ({}, {'max_observed_memory_age_s': 4.}),
                   ({}, {'age_checked_at_capture_time_s': 2.}),
                   ({}, {'reason': 'different_reason'}), ({'tracking_memory': None}, {})]
        for update, memory in changes:
            with self.subTest(update=update, memory=memory):
                self.setUp()
                self.select()
                receipt = deadline_sample(1, .1)
                receipt['tracking_memory'].update(memory)
                receipt.update(update)
                self.assertFalse(self.timeout(receipt)['valid'])
                self.assertFalse(self.guard.status()['recovery_pending'])
                self.assertFalse(self.update(2, capture_time_s=1.2)['valid'])

    def test_existing_overlap_hold_cannot_be_unlocked_by_verified_timeout(self):
        self.select()
        overlap = [person(1, (40, 10, 75, 90)), person(2, (55, 10, 90, 90))]
        self.assertEqual(self.update(1, overlap)['reason'], 'ambiguous_people_overlap')
        self.timeout(deadline_sample(2, .2))
        self.assertFalse(self.guard.status()['recovery_pending'])
        self.assertEqual(self.guard.status()['reason'], 'ambiguous_people_overlap')
        self.assertFalse(self.update(3, capture_time_s=1.3)['valid'])

    def test_same_number_with_changed_clothing_and_new_track_ids_latch_after_gap(self):
        cases = [([self.people[1]], self.colors),
                 ([person(3), self.people[1]], self.colors),
                 (self.people, {1: self.colors[2], 2: self.colors[1]})]
        for people, colors in cases:
            with self.subTest(people=people, colors=colors):
                self.setUp()
                self.select()
                self.timeout()
                self.assertFalse(self.update(2, people, colors, capture_time_s=1.2)['valid'])
                self.assertFalse(self.update(3, capture_time_s=1.3)['valid'])
                self.assertTrue(self.guard.status()['selection_required'])

    def test_pre_gap_competitor_location_remains_ambiguous_even_if_it_did_not_vanish(self):
        self.select()
        self.timeout()
        swapped_locations = [person(1, self.people[1]['bbox_xyxy']), person(2, self.people[0]['bbox_xyxy'])]
        out = self.update(2, swapped_locations, capture_time_s=1.2)
        self.assertFalse(out['valid'])
        self.assertEqual(out['reason'], 'ambiguous_identity_after_deadline')

    def test_vanished_pre_gap_competitor_near_selected_person_stops_recovery(self):
        self.select()
        self.timeout()
        out = self.update(2, [person(1, self.people[1]['bbox_xyxy'])], capture_time_s=1.2)
        self.assertFalse(out['valid'])
        self.assertEqual(out['reason'], 'ambiguous_identity_after_deadline')

    def test_multiple_current_colour_matches_after_gap_are_ambiguous(self):
        self.select()
        self.timeout()
        out = self.update(2, colors={1: self.colors[1], 2: self.colors[1]}, capture_time_s=1.2)
        self.assertFalse(out['valid'])
        self.assertEqual(out['reason'], 'ambiguous_clothing_after_deadline')

    def test_current_overlap_during_gap_latches_before_appearance_recovery(self):
        self.select()
        self.timeout()
        people = [person(1), person(2, (25, 10, 55, 90))]
        out = self.update(2, people, capture_time_s=1.2)
        self.assertEqual(out['reason'], 'ambiguous_people_overlap')
        self.assertFalse(self.update(3, capture_time_s=1.3)['valid'])

    def test_deadline_recovery_needs_new_current_image_and_fresh_age(self):
        for kind in ('missing_image', 'stale_elapsed'):
            with self.subTest(kind=kind):
                self.setUp()
                self.select()
                self.timeout()
                result = sample(2, self.people, 1.2)
                if kind == 'stale_elapsed':
                    self.guard.observe_frame(frame(1.2, self.people, self.colors))
                    result['age_at_finish_s'] = .7
                self.assertFalse(self.guard.update(result, 1.2, (100, 140), .65)['valid'])

    def test_fresh_completion_age_counts_towards_recovery_bound(self):
        self.select()
        self.timeout()
        self.guard.observe_frame(frame(2.9, self.people, self.colors))
        result = dict(sample(2, self.people, 2.9), age_at_finish_s=.2)
        out = self.guard.update(result, 2.9, (100, 140), .65)
        self.assertEqual(out['reason'], 'deadline_recovery_window_expired')

    def test_missing_completion_age_cannot_recover_after_verified_deadline(self):
        self.select()
        self.timeout()
        self.guard.observe_frame(frame(1.2, self.people, self.colors))
        result = sample(2, self.people, 1.2)
        result.pop('age_at_finish_s')
        out = self.guard.update(result, 1.2, (100, 140), .65)
        self.assertFalse(out['valid'])
        self.assertEqual(out['reason'], 'recovery_observation_not_fresh')
        self.assertFalse(self.guard.status()['recovery_pending'])
        self.assertTrue(self.guard.status()['selection_required'])

    def test_recovery_window_tracks_last_accepted_capture_not_old_selection(self):
        self.select()
        self.assertTrue(self.update(1, capture_time_s=2.)['valid'])
        self.timeout(deadline_sample(2, 2.1))
        self.assertTrue(self.guard.status()['recovery_pending'])
        self.assertTrue(self.update(3, capture_time_s=3.3)['valid'])

    def test_clear_during_deadline_wait_requires_another_explicit_selection(self):
        self.select()
        self.timeout()
        self.guard.clear()
        self.assertFalse(self.guard.status()['recovery_pending'])
        self.assertFalse(self.update(2, capture_time_s=1.2)['valid'])
        self.assertIsNone(self.guard.track_id)

    def test_verified_deadline_order_and_stream_watermarks_are_not_rolled_back(self):
        for change in ({'sequence': 0}, {'capture_time_s': 0.}, {'stream_id': 'other'}):
            with self.subTest(change=change):
                self.setUp()
                self.select()
                receipt = deadline_sample(1, .1)
                receipt.update(change)
                self.assertFalse(self.timeout(receipt)['valid'])
                self.assertFalse(self.guard.status()['recovery_pending'])
        self.setUp()
        self.select()
        receipt = deadline_sample(1, .1)
        self.timeout(receipt)
        self.assertFalse(self.timeout(copy.deepcopy(receipt))['valid'])
        self.assertFalse(self.guard.status()['recovery_pending'])
        self.assertFalse(self.update(2, capture_time_s=1.2)['valid'])

    def test_actual_pipeline_receipt_recovers_without_extending_observation_freshness(self):
        from tests.test_mantis_pipeline import Fixture
        fixture = Fixture()
        guard = SelectionGuard()
        def accept(result):
            prepared = fixture.pipeline.tracker.prepared_frame
            guard.observe_frame(prepared)
            return guard.update(result, result['capture_time_s'], (64, 64), .65)
        initial, _ = fixture.run(0., 0)
        self.assertFalse(accept(initial)['valid'])
        guard.select(1, 0)
        rejected, _ = fixture.run(.2, 1, readings=[100.2, 101.2])
        self.assertFalse(accept(rejected)['valid'])
        self.assertTrue(guard.status()['recovery_pending'])
        fresh, _ = fixture.run(1.3, 2)
        self.assertTrue(accept(fresh)['valid'])
        self.assertEqual(guard.track_id, 1)


if __name__ == '__main__':
    unittest.main()
