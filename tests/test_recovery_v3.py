"""Recovery tests include the recorded V2 collision checkpoint, not just a new route."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch

from flybrain_sim.autonomy import Autonomy, ObstacleMemory, ObservationTracker
from flybrain_sim.contracts import Box, BrainOutput, Observation, Scenario, VehicleState
from flybrain_sim.geometry import distance, point_box_distance, segment_intersects_box
from flybrain_sim.physics import step
from flybrain_sim.runner import motion_envelopes
from stress.course import make_trial, obstacles_at_trial
from stress.runner import run_trial
from stress.sensors import StressSensorModel


def observation(t=0., boxes=(), valid=True):
    return Observation(t, t, (5., 5., 3.), (0., 0., 0.), tuple(boxes), 22., valid)


class MemoryTests(unittest.TestCase):
    def test_rejected_geometry_and_pose_never_enter_state(self):
        scenario = Scenario(3, 'nominal', (30., 30., 10.), (5., 5., 3.), ((20., 20., 3.),), ())
        controller = Autonomy(scenario)
        accepted = observation()
        controller.update(accepted, 0., BrainOutput())
        injected = Box('untrusted', (4., 4., 1.), (6., 6., 5.), (1., 0., 0.))
        rejected = replace(observation(.2, (injected,), False), position=(20., 20., 3.))
        controller.update(rejected, .2, BrainOutput())
        self.assertIs(controller.tracker.last_observation, accepted)
        self.assertEqual(controller.tracker.last_valid, 0.)
        self.assertEqual(controller.obstacle_memory.entries, {})
        self.assertEqual(controller.completed, 0)
        predicted = controller.tracker.update(None, .4)
        self.assertFalse(predicted.valid)
        self.assertEqual(predicted.fault, 'dead_reckoning')

    def test_corrupted_timestamp_never_refreshes_memory(self):
        controller = Autonomy(Scenario(3, 'nominal', (30., 30., 10.), (5., 5., 3.), (), ()))
        box = Box('object', (10., 10., 1.), (11., 11., 5.), (0., 1., 0.))
        controller.update(observation(0., (box,)), 0., BrainOutput())
        controller.update(replace(observation(.2, (box,)), capture_time=-12.), .2, BrainOutput())
        self.assertEqual(controller.obstacle_memory.entries['object']['capture'], 0.)
        self.assertEqual(controller.tracker.last_valid, 0.)

    def test_disappearance_preserves_extent_and_reversal(self):
        memory = ObstacleMemory(set())
        first = Box('object', (10., 10., 1.), (11., 11., 5.), (0., 1., 0.))
        second = Box('object', (10., 14., 1.), (11., 15., 5.), (0., -.5, 0.))
        memory.accept(observation(0., (first,)), 0.)
        memory.accept(observation(4., (second,)), 4.)
        memory.accept(observation(5.), 5.)
        envelope, = memory.recovery_envelopes(10.)
        self.assertLessEqual(envelope.low[1], first.low[1])
        self.assertGreaterEqual(envelope.high[1], second.high[1])
        self.assertEqual(memory.snapshot(10.)['ages_s']['object'], 6.)
        self.assertEqual(memory.fresh_boxes(10., .8), ())

    def test_memory_is_bounded_and_expires_on_age_or_clock_reset(self):
        memory = ObstacleMemory(set())
        for i in range(100):
            box = Box(str(i), (10., 10., 1.), (11., 11., 5.), (0., 1., 0.))
            memory.accept(observation(i * .1, (box,)), i * .1)
        self.assertEqual(len(memory.entries), 64)
        memory.prune(71.)
        self.assertEqual(memory.entries, {})
        memory.accept(observation(100., (box,)), 100.)
        memory.prune(99.)
        self.assertEqual(memory.entries, {})


def replay_collision_checkpoint(remember=True):
    """Saved trial860 rounded state at31.6s; independent current sensor dynamics.

    Four accepted obstacle frames retain exactly the extent, peak velocity, and
    last observation relevant to memory. No rejected obstacle frame is used.
    Baseline replay with empty memory demonstrates the original failure remains
    reachable, so a changed global route cannot make this regression vacuous.
    """
    scenario = make_trial(1200859, 'compound')
    controller = Autonomy(scenario)
    if remember:
        for t, y, velocity in ((4.4, 11.428214, .698677),
                               (8., 15.00894, 1.148628),
                               (14.2, 19.499575, -.01579),
                               (17.8, 17.680681, -.922582)):
            box = Box('moving-inspection-fixture', (17., y, .4),
                      (19., y + 2., 6.6), (0., velocity, 0.))
            controller.obstacle_memory.accept(observation(t, (box,)), t)
    tracker = controller.tracker
    tracker.position = (17.465679, 22.565549, 4.939579)
    tracker.velocity = (-.451877, -1.902881, -.269493)
    tracker.disturbance = (.485473, .112973, .15583)
    tracker.command = (-.590764, -1.936326, -.251295)
    tracker.now = tracker.last_valid = 31.4
    tracker.last_observation = Observation(31.4, 31.4,
        (17.117832, 22.824359, 5.040078), (-.485043, -1.887314, -.397374),
        controller.planner.static, 20.25793)
    controller.gate.previous_now = controller.gate.previous_capture = 31.4
    controller._returning, controller.completed = True, 1
    state = VehicleState(31.6, (17.520314, 22.112856, 4.586419),
                         (-.334324, -1.869288, -.22714), 20.2467)
    sensor = StressSensorModel(scenario, 'compound', obstacles_at_trial)
    minimum, contacts, reasons = math.inf, [], set()
    while state.time < 45.:
        obs = sensor.observe(state)
        guidance = controller.update(obs, state.time, BrainOutput())
        reasons.add(guidance.reason)
        following = step(state, guidance.velocity, scenario)
        current_boxes, next_boxes = obstacles_at_trial(scenario, state.time), obstacles_at_trial(scenario, following.time)
        for box in motion_envelopes(current_boxes, next_boxes):
            if segment_intersects_box(state.position, following.position, box, .45):
                contacts.append((following.time, box.id))
        minimum = min(minimum, *(point_box_distance(following.position, box) - .45 for box in next_boxes))
        state = following
        if contacts:
            break
    return minimum, contacts, reasons


class RecoveryRegressionTests(unittest.TestCase):
    def test_original_collision_checkpoint_needs_memory_retreat(self):
        baseline_clearance, baseline_contacts, _ = replay_collision_checkpoint(False)
        self.assertTrue(baseline_contacts)
        clearance, contacts, reasons = replay_collision_checkpoint(True)
        self.assertEqual(contacts, [])
        self.assertGreater(clearance, .4)
        self.assertIn('remembered_obstacle_retreat', reasons)

    def test_original_collision_seed_completes_with_integrated_planner(self):
        result = run_trial(make_trial(1200859, 'compound'), 'compound', record=False)['result']
        self.assertTrue(result['mission_complete'])
        self.assertFalse(result['collision'])
        self.assertEqual(result['waypoints_completed'], 3)


class ReserveTests(unittest.TestCase):
    def test_return_reserve_uses_map_route_and_unknown_is_not_zero(self):
        scenario = make_trial(1200000, 'nominal')
        controller = Autonomy(scenario)
        position = scenario.waypoints[-1]
        actual = controller._return_distance(position, 0.)
        self.assertGreater(actual, distance(position, scenario.home) * 1.5)
        with patch.object(controller.planner, 'plan', return_value=None):
            self.assertEqual(controller._return_distance(position, 6.), actual + 30.)
            self.assertEqual(controller._return_distance(position, 12.), actual + 60.)
        self.assertEqual(controller._reserve_success_time, 0.)
        unknown = Autonomy(scenario)
        with patch.object(unknown.planner, 'plan', return_value=None):
            self.assertTrue(math.isinf(unknown._return_distance(position, 0.)))

    def test_low_energy_at_far_target_starts_return(self):
        scenario = make_trial(1200000, 'nominal')
        controller = Autonomy(scenario)
        obs = Observation(0., 0., scenario.waypoints[-1], (0., 0., 0.),
                          controller.planner.static, 2.)
        controller.update(obs, 0., BrainOutput())
        self.assertTrue(controller._returning)
        self.assertEqual(controller.abort_reason, 'low_energy_reserve')


if __name__ == '__main__':
    unittest.main()
