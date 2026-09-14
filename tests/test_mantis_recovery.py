"""Independent pause, scoring and counterfactual-isolation regressions.

Learned inference and rendering are replaced only in the runner integration
fixtures. The native aircraft dynamics, autopilot, tracking, command release
and tick scheduling execute unchanged in those tests.
"""
from contextlib import ExitStack
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from experiments.flight_contracts import CameraFrame, R_BODY_CAMERA
from experiments.mantis_flight import (capture_depth_pair, pause_delay, perceive_with_pause,
                                      run_episode, score_episode)
from experiments.mantis_world import MantisWorld
from experiments.mantis_vision import MantisVision, MANTIS_READOUT_FOLDER, load_mantis_readout
from perception.detector import Detection
from test_mantis_flight import operational_fixture
from test_mantis_tracking import Brain, Detector, Readout, BOX, RED


def pause_fixture(capture=3.2):
    spec, ticks, observations, episode = operational_fixture()
    spec.update(name='pause_fixture', kind='latency_recovery', event_s=3., injected_delay_s=1.6)
    complete = round(capture+2., 8)  # .4 seconds processing plus injected 1.6.
    observations = [r for r in observations if not capture < r['capture_time_s'] < complete]
    for i, row in enumerate(observations):
        row['sequence'] = i
        row['neural_stimulus_time_s'], row['neural_response_time_s'] = i*.1, (i+1)*.1
        command = row['candidate']; command['sequence'] = i
        if abs(row['capture_time_s']-capture) < 1e-8:
            row.update(injected_delay_s=1.6, completed_time_s=complete)
            command.update(issued_at_s=complete, valid=False, reason='stale_perception_result', forward_speed=0.)
    bind_requests(ticks, observations)
    positions = [0.]
    for tick in ticks:
        positions.append(positions[-1]+tick['request']['forward_speed']*.005)
    states = []
    for i in range(len(ticks)+1):
        speed = ticks[min(i, len(ticks)-1)]['request']['forward_speed']
        states.append(dict(time_s=i*.005, position=[positions[i], 0., 1.1], velocity=[speed, 0., 0.],
                           rotation=np.eye(3).tolist(), angular_velocity=[0., 0., 0.], motor_forces=[1.962]*4))
    for i, tick in enumerate(ticks):
        tick['state_before'], tick['state_after'] = deepcopy(states[i]), deepcopy(states[i+1])
    episode.update(observations=len(observations), actual_yolo_calls=len(observations),
                   actual_flyvis_observations=len(observations), neural_steps=5*len(observations),
                   final_state=deepcopy(states[-1]))
    return spec, ticks, observations, episode


def bind_requests(ticks, observations):
    for tick in ticks:
        t = tick['time_s']
        ready = [r for r in observations if r['completed_time_s'] <= t+1e-9]
        command = ready[-1]['candidate'] if ready else None
        active = command and command['valid'] and t < command['valid_until_s']-1e-9
        speed = command['forward_speed'] if active else 0.
        sequence = command['sequence'] if active else None
        tick['request'] = dict(forward_speed=speed, yaw_target=0., sequence=sequence,
                               reason=command['reason'] if active else 'no_fresh_guidance')
        tick['guardian'] = dict(forward_speed=speed, reason='clear' if speed else 'hold_clear')
        tick['applied_command_sequence'] = sequence


def detector_pause_fixture():
    spec, ticks, observations, episode = pause_fixture()
    spec['pause_stage'] = 'detector'
    row = next(r for r in observations if r['injected_delay_s'])
    row.update(injected_delay_s=0., injected_detector_delay_s=1.6, inference_wall_s=2.)
    row['detections'].update(status='rejected', reason='inference_deadline_missed',
        detections=[], control_authority=False, processing_ms=1800., age_at_finish_s=1.8,
        tracking_memory=dict(preserved_after_deadline=True, restored_track_count=1,
                             observation_timestamps_renewed=False, control_authority=False))
    row['observation'].update(valid=False, bbox_xyxy=None, center_normalized=None)
    return spec, ticks, observations, episode


class DetectorPauseHelperTests(unittest.TestCase):
    def test_invalid_pause_values_fail_before_process_or_sleep(self):
        for delay in (True, False, None, '1.6', -1., float('nan'), float('inf')):
            with self.subTest(delay=delay), \
                    patch('experiments.mantis_flight.time.sleep') as sleep:
                # An object without a pipeline/process proves validation happens
                # before a wrapper can alter or invoke either component.
                with self.assertRaises(ValueError):
                    perceive_with_pause(object(), object(), delay)
                sleep.assert_not_called()

    def test_delegates_once_and_restores_detector_on_return_or_exception(self):
        image = object()
        for failure_at in (None, 'detector', 'after_detection'):
            with self.subTest(failure_at=failure_at):
                original = Detector()
                output = [Detection(BOX, .9, 0, 'person')]
                calls = []
                def detect(received):
                    self.assertIs(received, image)
                    calls.append(received)
                    if failure_at == 'detector':
                        raise RuntimeError('injected detector exception')
                    return output
                original.detect = detect
                pipeline = SimpleNamespace(detector=original)
                def process(received):
                    result = pipeline.detector.detect(received)
                    if failure_at == 'after_detection':
                        raise RuntimeError('injected downstream exception')
                    return result
                vision = SimpleNamespace(pipeline=pipeline, process=process)
                with patch('experiments.mantis_flight.time.sleep') as sleep:
                    if failure_at:
                        with self.assertRaisesRegex(RuntimeError, 'injected'):
                            perceive_with_pause(vision, image, 1.6)
                    else:
                        self.assertIs(perceive_with_pause(vision, image, 1.6), output)
                    sleep.assert_called_once_with(1.6)
                self.assertEqual(len(calls), 1)
                self.assertIs(pipeline.detector, original)

    def test_zero_pause_does_not_install_a_wrapper_or_sleep(self):
        original = Detector()
        pipeline = SimpleNamespace(detector=original)
        def process(frame):
            self.assertIs(pipeline.detector, original)
            return frame
        frame = object()
        with patch('experiments.mantis_flight.time.sleep') as sleep:
            self.assertIs(perceive_with_pause(SimpleNamespace(pipeline=pipeline, process=process), frame), frame)
            sleep.assert_not_called()


class PauseScoringTests(unittest.TestCase):
    def test_detector_pause_requires_real_deadline_receipt_and_recovery(self):
        result = score_episode(*detector_pause_fixture())
        self.assertTrue(result['passed'], result)
        self.assertTrue(result['gates']['detector_deadline_memory_preserved'])

    def test_physical_only_pause_cannot_claim_detector_deadline_coverage(self):
        spec, ticks, observations, episode = pause_fixture()
        spec['pause_stage'] = 'detector'
        result = score_episode(spec, ticks, observations, episode)
        self.assertFalse(result['passed'], result)
        self.assertFalse(result['gates']['detector_deadline_memory_preserved'])

    def test_detector_delay_cannot_be_added_twice_to_physics_clock(self):
        spec, ticks, observations, episode = detector_pause_fixture()
        row = next(r for r in observations if r.get('injected_detector_delay_s'))
        # The fixture already includes sleep inside its two-second wall time.
        row['injected_delay_s'] = 1.6
        result = score_episode(spec, ticks, observations, episode)
        self.assertFalse(result['passed'], result)
        self.assertFalse(result['gates']['causal_observation_clock'])

    def test_inconsistent_detector_rejection_receipt_cannot_pass(self):
        mutations = [
            lambda r: r['detections'].update(status='ok'),
            lambda r: r['detections'].update(detections=[dict(track_id=1)]),
            lambda r: r['detections'].update(processing_ms=100.),
            lambda r: r['detections'].update(control_authority=True),
            lambda r: r['detections']['tracking_memory'].update(restored_track_count=0),
            lambda r: r['detections']['tracking_memory'].update(observation_timestamps_renewed=True),
            lambda r: r['detections']['tracking_memory'].update(control_authority=True),
            lambda r: r['observation'].update(valid=True, bbox_xyxy=BOX, center_normalized=[0., 0.]),
        ]
        for i, mutate in enumerate(mutations):
            with self.subTest(mutation=i):
                spec, ticks, observations, episode = detector_pause_fixture()
                row = next(r for r in observations if r.get('injected_detector_delay_s'))
                mutate(row)
                result = score_episode(spec, ticks, observations, episode)
                self.assertFalse(result['passed'], result)

    def test_once_at_first_due_capture_and_invalid_delay_rejected(self):
        spec = dict(kind='latency_recovery', event_s=6., injected_delay_s=1.6)
        applied = False; injected = []
        for capture in (0., 5.8, 6.05, 7.8, 9.):
            delay = pause_delay(spec, capture, applied)
            if delay: injected.append((capture, delay)); applied = True
        self.assertEqual(injected, [(6.05, 1.6)])
        for value in (True, 0., .9, float('nan'), float('inf')):
            with self.subTest(delay=value), self.assertRaises(ValueError):
                pause_delay(dict(spec, injected_delay_s=value), 6., False)
        self.assertEqual(pause_delay(dict(spec, kind='normal'), 6., False), 0.)

    def test_complete_pause_stop_recovery_fixture_passes(self):
        result = score_episode(*pause_fixture())
        self.assertTrue(result['passed'], result)
        self.assertTrue(result['gates']['paused_vehicle_stopped'])
        self.assertGreater(result['metrics']['pause_hold_ticks'], 0)

    def test_pause_cannot_be_injected_early_or_skip_first_due_capture(self):
        for capture in (.8, 4.4):
            with self.subTest(capture=capture):
                result = score_episode(*pause_fixture(capture))
                self.assertFalse(result['passed'], result)
                self.assertFalse(result['gates']['pause_injected_once'])

    def test_fresh_capture_whose_release_misses_recovery_deadline_fails(self):
        spec, ticks, observations, episode = pause_fixture()
        # Pause completes at 5.2; first recovery capture is 7.2 but its fresh
        # result is not released until 7.6, beyond the two-second deadline.
        for row in observations:
            if 5.2 <= row['capture_time_s'] < 7.2:
                row['candidate'].update(valid=False, forward_speed=0., reason='target_depth_unavailable')
        bind_requests(ticks, observations)
        result = score_episode(spec, ticks, observations, episode)
        self.assertFalse(result['passed'], result)
        self.assertFalse(result['gates']['fresh_same_id_after_pause'])

    def test_zero_commands_do_not_prove_vehicle_stopped(self):
        spec, ticks, observations, episode = pause_fixture()
        for tick in ticks:
            for key in ('state_before', 'state_after'):
                if 4.1 <= tick[key]['time_s'] <= 5.2:
                    tick[key]['velocity'] = [.15, 0., 0.]
        result = score_episode(spec, ticks, observations, episode)
        self.assertFalse(result['passed'], result)
        self.assertFalse(result['gates']['paused_vehicle_stopped'])

    def test_stale_pause_result_and_fresh_changed_id_are_rejected(self):
        spec, ticks, observations, episode = pause_fixture()
        paused = next(r for r in observations if r['injected_delay_s'])
        paused['candidate'].update(valid=True, reason='neural_bearing_and_registered_depth')
        self.assertFalse(score_episode(spec, ticks, observations, episode)['passed'])
        spec, ticks, observations, episode = pause_fixture()
        for row in observations:
            if row['capture_time_s'] >= 5.2:
                row['observation']['track_id'] = 2
        self.assertFalse(score_episode(spec, ticks, observations, episode)['passed'])

    def test_missing_injection_or_missing_fresh_recovery_cannot_pass(self):
        spec, ticks, observations, episode = pause_fixture()
        for row in observations:
            if row['injected_delay_s']:
                row['injected_delay_s'] = 0.
        self.assertFalse(score_episode(spec, ticks, observations, episode)['passed'])
        spec, ticks, observations, episode = pause_fixture()
        for row in observations:
            if 5.2 <= row['capture_time_s'] <= 7.2:
                row['candidate'].update(valid=False, forward_speed=0., reason='target_depth_unavailable')
        bind_requests(ticks, observations)
        self.assertFalse(score_episode(spec, ticks, observations, episode)['passed'])


class DepthPairRestorationTests(unittest.TestCase):
    def test_original_barrier_and_aircraft_restored_after_success_or_render_error(self):
        world = MantisWorld()
        self.addCleanup(world.close)
        spec = dict(causal_depth_test=True, event_s=0., obstacle_position=[9., 9., .7])
        for enabled in (True, False):
            for throws in (False, True):
                with self.subTest(enabled=enabled, rendering_raises=throws):
                    world.set_obstacle([2., .3, .7], enabled=enabled)
                    before = {k: getattr(world.data, k).copy() for k in ('qpos', 'qvel', 'ctrl', 'mocap_pos', 'mocap_quat')}
                    original_time = float(world.data.time)
                    calls = []
                    def capture(*, safety):
                        self.assertTrue(safety)
                        calls.append(world._obstacle_enabled)
                        if len(calls) == 2 and throws:
                            raise RuntimeError('synthetic renderer failure')
                        return SimpleNamespace(capture_time_s=float(world.data.time))
                    with patch.object(world, 'capture', side_effect=capture):
                        if throws:
                            with self.assertRaisesRegex(RuntimeError, 'synthetic renderer'):
                                capture_depth_pair(world, spec)
                        else:
                            actual, other = capture_depth_pair(world, spec)
                            self.assertEqual(actual.capture_time_s, other.capture_time_s)
                    self.assertEqual(calls, [enabled, False])
                    self.assertEqual(world._obstacle_enabled, enabled)
                    np.testing.assert_array_equal(world._obstacle_position, [2., .3, .7])
                    for key, value in before.items():
                        np.testing.assert_array_equal(getattr(world.data, key), value)
                    self.assertEqual(float(world.data.time), original_time)


class ReadoutIntegrityTests(unittest.TestCase):
    def test_accepted_readout_loads_without_initializing_learned_models(self):
        with patch('experiments.flight_vision.FlightVision.__init__') as initialize:
            readout, receipt = load_mantis_readout()
        initialize.assert_not_called()
        self.assertIs(receipt['validation']['passed'], True)
        self.assertTrue(np.isfinite(readout.predict(np.zeros(8))).all())

    def test_changed_readout_or_selection_bytes_are_rejected(self):
        for filename in ('readout.json', 'selection_frozen.json'):
            with self.subTest(file=filename), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp)
                for name in ('readout.json', 'selection_frozen.json'):
                    shutil.copyfile(MANTIS_READOUT_FOLDER/name, folder/name)
                value = json.loads((folder/filename).read_text())
                if filename == 'readout.json':
                    value['intercept'][0] += .001
                else:
                    value['validation']['passed'] = False
                (folder/filename).write_text(json.dumps(value))
                with patch('experiments.mantis_vision.MANTIS_READOUT_FOLDER', folder):
                    with self.assertRaisesRegex(ValueError, 'integrity|selection receipt'):
                        load_mantis_readout()

    def test_model_manifest_mismatch_rejected_before_any_model_initialization(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest = Path(temp)/'changed_manifest.json'
            manifest.write_text('{}')
            with patch('experiments.mantis_vision.MANIFEST', manifest), \
                    patch('experiments.flight_vision.FlightVision.__init__') as initialize:
                with self.assertRaisesRegex(ValueError, 'manifest mismatch'):
                    MantisVision()
            initialize.assert_not_called()


class StubBrain(Brain):
    def __init__(self):
        self.baseline_activity = np.zeros(45669, np.float32)
        self.cell_indices = np.arange(721)
        self.centers_rc = np.zeros((721, 2))
        self.reset()

    def reset(self):
        self.steps = 0

    def step(self, mask, hold_s):
        result = super().step(mask, hold_s)
        result.update(activity=self.baseline_activity.copy(), retina=np.zeros(721), elapsed_wall_s=0.,
                      stimulus_time_s=self.steps*.1, response_time_s=(self.steps+1)*.1)
        self.steps += 1
        return result


class StubVision(MantisVision):
    def __init__(self):
        self.brain, self.detector, self.readout = StubBrain(), Detector(), Readout()
        self.detector.output = [Detection(BOX, .9, 0, 'person')]
        self.reset()


class NativeSensorFixture(MantisWorld):
    """Native motors and 3D world, deterministic synthetic ideal RGB-D inputs."""
    def capture(self, safety=False):
        state = self.state()
        rgb = np.full((391, 391, 3), 110, np.uint8)
        x0, y0, x1, y1 = map(int, BOX); rgb[y0:y1, x0:x1] = RED
        rgb[0, 0, 0] = 255 if self._obstacle_enabled else 0
        return CameraFrame(rgb, np.full((391, 391), 4.5, np.float32),
                           (280., 280., 195., 195.), state.rotation@R_BODY_CAMERA,
                           state.position+state.rotation@np.array([.25, 0., 0.]), state.time_s)

    def overview(self):
        return np.zeros((4, 4, 3), np.uint8)


class VirtualInferenceClock:
    """Only deterministic test sleep advances this wall clock; no real delay."""
    def __init__(self):
        self.value, self.sleeps = 100., []

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.value += seconds


class RunnerIsolationTests(unittest.TestCase):
    def run_fixture(self, spec, counterfactual_allows=True, virtual_clock=None):
        class Guardian:
            def check(self, frame, state, requested_speed, now_s):
                blocked = frame.rgb[0, 0, 0] == 255
                counterfactual = spec['kind'] == 'stop' and now_s >= spec['event_s']-1e-9 and not blocked
                speed = 0. if blocked or counterfactual and not counterfactual_allows else requested_speed
                return dict(forward_speed=speed, reason='blocked_stopping_distance' if blocked else 'clear')
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)/'episode'
            with patch('experiments.mantis_flight.MantisWorld', NativeSensorFixture), \
                    patch('experiments.mantis_flight.DepthGuardian', Guardian), \
                    patch('experiments.mantis_flight.np.savez_compressed'), \
                    patch('experiments.mantis_flight.score_episode', return_value={'passed': False}), \
                    ExitStack() as stack:
                vision = StubVision()
                if virtual_clock is not None:
                    stack.enter_context(patch('experiments.mantis_flight.time.perf_counter', new=virtual_clock))
                    stack.enter_context(patch('experiments.flight_vision.time.monotonic', new=virtual_clock))
                    stack.enter_context(patch('experiments.mantis_flight.time.sleep', side_effect=virtual_clock.sleep))
                    reset = vision.reset
                    def clocked_reset():
                        reset()
                        vision.pipeline.clock = virtual_clock
                    stack.enter_context(patch.object(vision, 'reset', side_effect=clocked_reset))
                run_episode(spec, folder, vision)
            import json
            ticks = [json.loads(s) for s in (folder/'ticks.jsonl').read_text().splitlines()]
            observations = [json.loads(s) for s in (folder/'observations.jsonl').read_text().splitlines()]
            return ticks, observations

    def test_detector_pause_exercises_real_deadline_once_without_doubled_physics_delay(self):
        spec = dict(name='detector_pause', kind='latency_recovery', duration_s=4., trajectory='diagonal',
                    method='mantis_neural', event_s=1., injected_delay_s=1.6, pause_stage='detector')
        clock = VirtualInferenceClock()
        ticks, observations = self.run_fixture(spec, virtual_clock=clock)
        self.assertEqual(clock.sleeps, [1.6])
        paused = [r for r in observations if r['injected_detector_delay_s']]
        self.assertEqual(len(paused), 1)
        row = paused[0]
        self.assertEqual(row['injected_delay_s'], 0.)
        self.assertAlmostEqual(row['inference_wall_s'], 1.6)
        self.assertAlmostEqual(row['completed_time_s']-row['capture_time_s'], 1.6)
        self.assertAlmostEqual(row['detections']['processing_ms'], 1600.)
        self.assertEqual(row['detections']['status'], 'rejected')
        self.assertEqual(row['detections']['reason'], 'inference_deadline_missed')
        self.assertEqual(row['detections']['detections'], [])
        self.assertTrue(row['detections']['tracking_memory']['preserved_after_deadline'])
        self.assertFalse(row['observation']['valid'])
        self.assertFalse(row['candidate']['valid'])
        self.assertEqual(row['candidate']['reason'], 'stale_perception_result')
        self.assertEqual(len(ticks), 800)
        self.assertEqual(row['delay_physics_tick_end']-row['delay_physics_tick_start'], 320)
        held = [t for t in ticks if row['capture_time_s']+.9 <= t['time_s'] < row['completed_time_s']]
        self.assertTrue(held)
        self.assertTrue(all(t['request']['forward_speed'] == t['guardian']['forward_speed'] == 0. for t in held))
        self.assertLess(np.linalg.norm(held[-1]['state_after']['velocity']), .08)
        fresh = [r for r in observations if r['capture_time_s'] >= row['completed_time_s'] and
                 r['candidate'] and r['candidate']['valid']]
        self.assertTrue(fresh)
        self.assertTrue(all(r['observation']['track_id'] == 1 for r in fresh))

    def test_injected_pause_advances_native_physics_and_expires_commands_before_recovery(self):
        spec = dict(name='pause', kind='latency_recovery', duration_s=4., trajectory='diagonal',
                    method='mantis_neural', event_s=1., injected_delay_s=1.6)
        ticks, observations = self.run_fixture(spec)
        paused = [r for r in observations if r['injected_delay_s']]
        self.assertEqual(len(paused), 1)
        row = paused[0]
        self.assertGreaterEqual(row['capture_time_s'], 1.-1e-9)
        self.assertGreaterEqual(row['completed_time_s']-row['capture_time_s'], 1.6)
        self.assertFalse(row['candidate']['valid'])
        self.assertEqual(row['candidate']['reason'], 'stale_perception_result')
        self.assertEqual(len(ticks), 800)
        self.assertGreater(row['delay_physics_tick_end']-row['delay_physics_tick_start'], 300)
        held = [t for t in ticks if row['capture_time_s']+.9 <= t['time_s'] < row['completed_time_s']]
        self.assertTrue(held)
        self.assertTrue(all(t['request']['forward_speed'] == t['guardian']['forward_speed'] == 0. for t in held))
        self.assertLess(np.linalg.norm(held[-1]['state_after']['velocity']), .08)
        self.assertGreater(ticks[-1]['state_after']['position'][0], .2)
        fresh = [r for r in observations if r['capture_time_s'] >= row['completed_time_s'] and r['candidate'] and r['candidate']['valid']]
        self.assertTrue(fresh)
        self.assertTrue(all(r['observation']['track_id'] == 1 for r in fresh))

    def test_counterfactual_guardian_cannot_change_actual_rotor_targets_or_motion(self):
        spec = dict(name='barrier', kind='stop', duration_s=1.5, trajectory='forward', method='mantis_neural',
                    event_s=.5, obstacle_position=[1.75, .4, .7], causal_depth_test=True)
        allowing, _ = self.run_fixture(spec, counterfactual_allows=True)
        rejecting, _ = self.run_fixture(spec, counterfactual_allows=False)
        self.assertEqual(len(allowing), len(rejecting))
        changed = 0
        for a, b in zip(allowing, rejecting):
            np.testing.assert_array_equal(a['motor_targets'], b['motor_targets'])
            self.assertEqual(a['state_after'], b['state_after'])
            self.assertEqual(a['guardian'], b['guardian'])
            if a['counterfactual_guardian'] and a['counterfactual_guardian'] != b['counterfactual_guardian']:
                changed += 1
        self.assertGreater(changed, 0)
        self.assertTrue(any(t['request']['forward_speed'] > 0 and t['guardian']['forward_speed'] == 0 for t in allowing))


if __name__ == '__main__':
    unittest.main()
