"""Independent Mantis integration rejection tests; no learned inference runs."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.flight_contracts import CameraFrame, R_BODY_CAMERA
from experiments.mantis_comparison import AlphaBeta, METHODS
from experiments.mantis_flight import (bearing_input, fixture_pose, run_benchmarks,
    run as run_flight, score_episode, selected_candidate, update_fixture)
from experiments.mantis_world import MantisWorld
from test_flight_report import fixture as motor_trace_fixture


def frame():
    return CameraFrame(np.zeros((391, 391, 3), np.uint8), np.full((391, 391), 4., np.float32),
                       (280., 280., 195., 195.), R_BODY_CAMERA.copy(), np.array([.25, 0., 1.1]), 0.)


def bundle():
    return dict(sequence=0, observation=dict(valid=True, bbox_xyxy=[160., 130., 230., 260.],
                    track_id=1, capture_time_s=0., sequence=0),
                surface=dict(surface_optical_z_m=4., valid_depth_fraction=1., depth_spread_p90_p10_m=.01),
                candidate=dict(valid=True, reason='neural_bearing_and_registered_depth',
                    heading_world_rad=.02, surface_optical_z_m=4., decoded=[0., 0., .33],
                    capture_time_s=0., track_id=1, cue_input=[0., 0., .33]))


def operational_fixture(kind='normal'):
    old_kind = 'obstacle_stop' if kind == 'stop' else 'normal'
    spec, ticks, rows, receipt = motor_trace_fixture(8. if kind == 'stop' else 10., old_kind)
    spec.update(kind=kind, name='fixture', method='mantis_neural', trajectory='forward' if kind == 'stop' else 'diagonal')
    for row in rows:
        row.update(truth_heading_world_rad=0., image_hw=[391, 391],
                   camera_intrinsics=[280., 280., 195., 195.], camera_rotation=R_BODY_CAMERA.tolist(),
                   camera_position=[.25, 0., 1.1],
                   detections=dict(inference_executed=True, status='ok', processing_ms=200.),
                   discarded_after_episode=False)
        row['target_truth_bbox'] = [160., 130., 230., 260.]
        row['observation']['bbox_xyxy'] = [160., 130., 230., 260.]
    receipt.update(actual_yolo_calls=len(rows), actual_flyvis_observations=len(rows),
                   neural_steps=5*len(rows), method='mantis_neural')
    return spec, ticks, rows, receipt


class CandidateBoundaryTests(unittest.TestCase):
    def unavailable(self, candidate_bundle, camera):
        for method in METHODS:
            with self.subTest(method=method):
                try:
                    result = selected_candidate(method, candidate_bundle, camera, AlphaBeta())
                except ValueError:
                    continue
                self.assertFalse(result['valid'], result)

    def test_baselines_can_operate_when_only_neural_evidence_failed(self):
        data = bundle()
        data['candidate'].update(valid=False, reason='neural_evidence_unavailable', heading_world_rad=None)
        for method in ('direct_yolo', 'alpha_beta'):
            result = selected_candidate(method, data, frame(), AlphaBeta())
            self.assertTrue(result['valid'])
            self.assertAlmostEqual(result['heading_world_rad'], 0.)
            self.assertEqual(result['surface_optical_z_m'], 4.)
        self.assertFalse(selected_candidate('mantis_neural', data, frame(), AlphaBeta())['valid'])

    def test_every_method_requires_current_registered_supported_depth(self):
        self.unavailable(bundle(), replace(frame(), registration_verified=False))
        for surface in (None, {}, dict(surface_optical_z_m=.1, valid_depth_fraction=1., depth_spread_p90_p10_m=0.),
                        dict(surface_optical_z_m=4., valid_depth_fraction=.3, depth_spread_p90_p10_m=.01),
                        dict(surface_optical_z_m=4., valid_depth_fraction=1., depth_spread_p90_p10_m=1.),
                        dict(surface_optical_z_m=True, valid_depth_fraction=1., depth_spread_p90_p10_m=0.)):
            data = bundle(); data['surface'] = surface
            self.unavailable(data, frame())

    def test_shared_current_missing_and_envelope_rejection_includes_neural_branch(self):
        missing = bundle(); missing['observation'].update(valid=False, bbox_xyxy=None)
        self.unavailable(missing, frame())
        outside = bundle(); outside['observation']['bbox_xyxy'] = [310., 130., 380., 260.]
        self.unavailable(outside, frame())

    def test_neural_candidate_cannot_relabel_capture_or_selected_identity(self):
        for change in (dict(capture_time_s=.05), dict(track_id=7)):
            data = bundle(); data['candidate'].update(change)
            try:
                result = selected_candidate('mantis_neural', data, frame(), AlphaBeta())
            except ValueError:
                continue
            self.assertFalse(result['valid'], result)

    def test_truth_fields_do_not_change_sensor_guidance_and_inputs_are_preserved(self):
        data = bundle(); original = deepcopy(data)
        expected = {m: selected_candidate(m, data, frame(), AlphaBeta()) for m in METHODS}
        data.update(truth_heading_world_rad=-2., actor_truth_position=[-100., 100., 100.])
        actual = {m: selected_candidate(m, data, frame(), AlphaBeta()) for m in METHODS}
        self.assertEqual(actual, expected)
        for key, value in original.items(): self.assertEqual(data[key], value)
        sensor = bearing_input(frame(), data)
        self.assertNotIn('truth_heading_world_rad', vars(sensor))


class OperationalScoringTests(unittest.TestCase):
    def setUp(self):
        self.spec, self.ticks, self.rows, self.receipt = operational_fixture()

    def score(self):
        return score_episode(self.spec, self.ticks, self.rows, self.receipt)

    def assert_rejected(self):
        try:
            result = self.score()
        except ValueError:
            return
        self.assertFalse(result['passed'], result)

    def test_complete_causal_moving_fixture_passes(self):
        result = self.score()
        self.assertTrue(result['passed'], result)
        self.assertEqual(json.loads(json.dumps(result, allow_nan=False)), result)

    def test_duplicate_tick_does_not_pass_by_preserving_row_count(self):
        self.ticks[21] = deepcopy(self.ticks[20])
        self.assert_rejected()

    def test_state_jump_and_false_final_receipt_are_rejected(self):
        self.ticks[21]['state_after']['position'][1] += .01
        self.assert_rejected()
        self.spec, self.ticks, self.rows, self.receipt = operational_fixture()
        self.receipt['final_state']['position'][0] += .1
        self.assert_rejected()

    def test_no_translation_cannot_pass_due_to_nonzero_starting_position(self):
        for tick in self.ticks:
            for key in ('state_before', 'state_after'):
                tick[key]['position'] = [5., 0., 1.1]
                tick[key]['velocity'] = [0., 0., 0.]
        self.receipt['final_state'] = deepcopy(self.ticks[-1]['state_after'])
        self.assert_rejected()

    def test_neural_and_yolo_call_counts_must_match_each_capture(self):
        self.receipt['actual_flyvis_observations'] -= 1
        self.assert_rejected()

    def test_tail_loss_cannot_be_hidden_by_one_centered_tail_sample(self):
        for row in self.rows:
            if row['capture_time_s'] > self.spec['duration_s']-2.:
                row['observation'].update(valid=False, bbox_xyxy=None, center_normalized=None)
        self.assert_rejected()

    def test_single_ground_truth_observation_cannot_certify_normal_overlap(self):
        for row in self.rows[1:]:
            row['truth_heading_world_rad'] = None
            row['target_truth_bbox'] = None
        self.assert_rejected()

    def test_same_person_claim_rejects_changed_track_identity(self):
        self.rows[len(self.rows)//2]['observation']['track_id'] = 99
        self.assert_rejected()

    def test_early_command_and_extended_capture_expiry_are_rejected(self):
        self.rows[0]['candidate']['issued_at_s'] = .1
        self.assert_rejected()
        self.spec, self.ticks, self.rows, self.receipt = operational_fixture()
        self.rows[0]['candidate']['valid_until_s'] += 100.
        self.assert_rejected()

    def test_future_or_unreleased_command_cannot_be_applied(self):
        self.ticks[0]['request'].update(sequence=0, forward_speed=.16)
        self.ticks[0]['applied_command_sequence'] = 0
        self.ticks[0]['guardian']['forward_speed'] = .16
        self.assert_rejected()

    def test_depth_cannot_create_forward_authority(self):
        self.ticks[0]['guardian']['forward_speed'] = .3
        self.assert_rejected()

    def test_out_of_bounds_motor_target_and_contact_receipt_are_rejected(self):
        self.ticks[0]['motor_targets'][0] = 6.
        self.assert_rejected()
        self.spec, self.ticks, self.rows, self.receipt = operational_fixture()
        self.ticks[0]['truth_after']['contacts'] = [{'geom1': 'quad', 'geom2': 'wall', 'distance_m': -.01}]
        self.assert_rejected()

    def test_actual_depth_reason_counts_and_stop_requires_no_tail_authority(self):
        self.spec, self.ticks, self.rows, self.receipt = operational_fixture('stop')
        result = self.score()
        self.assertTrue(result['passed'], result)
        self.assertGreater(result['metrics']['depth_override_ticks'], 0)
        for tick in self.ticks:
            if tick['time_s'] >= self.spec['event_s']+2.:
                tick['guardian']['forward_speed'] = tick['request']['forward_speed']
        self.assert_rejected()

    def test_obstacle_case_requires_actual_obstacle_truth(self):
        self.spec, self.ticks, self.rows, self.receipt = operational_fixture('stop')
        for tick in self.ticks: tick['truth_after']['obstacle_enabled'] = False
        self.assert_rejected()


class BenchmarkReceiptTests(unittest.TestCase):
    def test_missing_requested_neural_source_is_an_error_not_empty_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(ValueError, 'missing'):
                run_benchmarks(root, SimpleNamespace(brain=object(), readout=object()),
                               [dict(name='walk', kind='normal')], [('clean', 27101)])
            self.assertEqual(list(root.iterdir()), [])

    def test_benchmark_without_neural_controller_rejected_before_output_or_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'not_started'
            with self.assertRaisesRegex(ValueError, 'benchmark needs'):
                run_flight(output, [dict(name='walk', kind='normal')], methods=('direct_yolo',), benchmark=True)
            self.assertFalse(output.exists())

    def test_empty_scenario_or_controller_plan_cannot_pass_vacuously(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'not_started'
            for specs, methods in (([], METHODS), ([dict(name='walk', kind='normal')], ())):
                with self.subTest(specs=specs, methods=methods), self.assertRaises(ValueError):
                    run_flight(output, specs, methods=methods, benchmark=False)
                self.assertFalse(output.exists())

    def test_recorded_pipeline_milliseconds_are_not_silently_zeroed(self):
        spec, _, rows, _ = operational_fixture()
        rows = rows[:2]
        for row in rows: row['detections']['processing_ms'] = 123.4
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root/(spec['name']+'__mantis_neural'); folder.mkdir()
            (folder/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            seen = []
            def compare(inputs, *_args, **_kwargs):
                seen.extend(inputs)
                return {'rows': []}
            with patch('experiments.mantis_flight.run_comparison', side_effect=compare), \
                    patch('experiments.mantis_flight.score_comparison', return_value={'matched_count': 2}), \
                    redirect_stdout(io.StringIO()):
                run_benchmarks(root, SimpleNamespace(brain=object(), readout=object()), [spec], [('clean', 27101)])
            self.assertEqual(len(seen), 2)
            self.assertTrue(all(abs(row.detector_wall_s-.1234) < 1e-12 for row in seen))
            self.assertEqual([row.capture_time_s for row in seen], [r['capture_time_s'] for r in rows])


class MantisWorldInvariantTests(unittest.TestCase):
    def setUp(self):
        self.world = MantisWorld()
        self.addCleanup(self.world.close)

    def test_actor_trajectory_changes_only_mocap_not_aircraft_state_or_time(self):
        self.world.data.qpos[:3] = [1., 2., 1.1]
        self.world.data.qvel[:] = .03
        before = {k: getattr(self.world.data, k).copy() for k in ('qpos', 'qvel', 'ctrl')}
        timestamp = float(self.world.data.time)
        self.world.set_actor([5., .3, 0.], .4)
        for key, value in before.items(): np.testing.assert_array_equal(getattr(self.world.data, key), value)
        self.assertEqual(self.world.data.time, timestamp)
        self.assertEqual((self.world.model.nq, self.world.model.nv, self.world.model.nu), (7, 6, 4))
        self.assertGreater(self.world.model.nskin, 0)

    def test_evaluator_rejects_old_camera_with_future_actor_pose(self):
        camera = frame()
        self.world.step(self.world.motor_forces.copy())
        self.world.set_actor([5., .3, 0.], .4)
        with self.assertRaises(ValueError): self.world.evaluation_projection(camera)

    def test_current_actor_projection_is_evaluator_only_and_time_aligned(self):
        before = self.world.state()
        annotation = self.world.evaluation_projection(frame())
        after = self.world.state()
        self.assertTrue(annotation['visible'])
        self.assertAlmostEqual(annotation['actor_pose_time_s'], before.time_s)
        self.assertTrue(np.isfinite(annotation['heading_world_rad']))
        np.testing.assert_array_equal(after.position, before.position)
        np.testing.assert_array_equal(after.velocity, before.velocity)

    def test_development_and_heldout_fixture_paths_are_distinct(self):
        spec = dict(name='walk', kind='normal', trajectory='diagonal')
        heldout, _, _ = fixture_pose(spec, 1.)
        development, _, _ = fixture_pose(dict(spec, development_fixture=True), 1.)
        self.assertFalse(np.allclose(heldout, development))
        update_fixture(self.world, dict(spec, development_fixture=True))
        np.testing.assert_allclose(self.world.truth()['target_ground_anchor'], fixture_pose(dict(spec, development_fixture=True), 0.)[0])


if __name__ == '__main__':
    unittest.main()
