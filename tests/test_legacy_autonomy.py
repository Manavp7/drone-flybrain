import math
import unittest

from flybrain_sim.autonomy import Autonomy
from flybrain_sim.contracts import Box, BrainOutput, Observation, Scenario
from flybrain_sim.geometry import norm, segment_intersects_box, within_bounds
from flybrain_sim.planning import VisibilityPlanner
from flybrain_sim.safety import SafetyGate


def scenario(obstacles=(), waypoints=((12.0, 4.0, 3.0),)):
    return Scenario(1, "unit", (20.0, 20.0, 12.0), (3.0, 3.0, 3.0),
                    waypoints, obstacles)


def observation(now=0.0, position=(3.0, 3.0, 3.0), **kwargs):
    values = dict(capture_time=now, receive_time=now, position=position,
                  velocity=(0.0, 0.0, 0.0), obstacles=(), battery_wh=22.0,
                  valid=True)
    values.update(kwargs)
    return Observation(**values)


class PlanningTests(unittest.TestCase):
    def test_route_goes_around_box_in_three_dimensions(self):
        box = Box("fixture", (6.0, 1.0, 0.0), (9.0, 8.0, 7.0))
        s = scenario((box,))
        planner = VisibilityPlanner(s)
        route = planner.plan(s.home, s.waypoints[0])
        self.assertIsNotNone(route)
        prior = s.home
        for point in route:
            self.assertTrue(within_bounds(point, s.bounds))
            self.assertFalse(segment_intersects_box(prior, point, box, 0.94))
            prior = point
        self.assertEqual(route[-1], s.waypoints[0])

    def test_goal_inside_planning_margin_uses_guarded_final_approach(self):
        from dataclasses import replace
        box = Box("corner", (6.0, 6.0, 0.0), (9.0, 9.0, 7.0))
        s = replace(scenario((box,), ((4.7, 4.7, 3.0),)), sensor_noise=0.32)
        planner = VisibilityPlanner(s, margin=0.5 + math.sqrt(3.0) * s.sensor_noise)
        self.assertFalse(planner.segment_clear(s.waypoints[0], s.waypoints[0]))
        route = planner.plan(s.home, s.waypoints[0])
        self.assertIsNotNone(route)
        self.assertEqual(route[-1], s.waypoints[0])
        self.assertFalse(segment_intersects_box(s.home, route[-1], box,
                                                planner.recovery_clearance))

    def test_enclosed_goal_is_not_reported_as_reachable(self):
        box = Box("solid", (10.0, 2.0, 1.0), (14.0, 6.0, 5.0))
        s = scenario((box,))
        self.assertIsNone(VisibilityPlanner(s).plan(s.home, s.waypoints[0]))

    def test_moving_obstacle_is_used_when_observed(self):
        moving = Box("moving", (6.0, 1.0, 0.0), (9.0, 8.0, 7.0), (0.5, 0.0, 0.0))
        s = scenario((moving,))
        planner = VisibilityPlanner(s)
        self.assertEqual(planner.plan(s.home, s.waypoints[0]), [s.waypoints[0]])
        route = planner.plan(s.home, s.waypoints[0], (moving,))
        self.assertIsNotNone(route)
        self.assertGreater(len(route), 1)

    def test_recovery_from_discretionary_clearance_is_possible(self):
        box = Box("fixture", (6.0, 1.0, 0.0), (9.0, 8.0, 7.0))
        s = scenario((box,), waypoints=((3.0, 10.0, 3.0),))
        planner = VisibilityPlanner(s)
        start = (5.2, 4.0, 3.0)
        route = planner.plan(start, s.waypoints[0])
        self.assertIsNotNone(route)
        self.assertFalse(segment_intersects_box(start, route[0], box,
                                                planner.recovery_clearance))


class SafetyTests(unittest.TestCase):
    def test_faults_request_zero_simulated_brake(self):
        cases = [dict(valid=False), dict(capture_time=-10.0),
                 dict(capture_time=10.0), dict(position=(math.nan, 3.0, 3.0)),
                 dict(receive_time=math.inf), dict(battery_wh=math.nan)]
        for case in cases:
            with self.subTest(case=case):
                autonomy = Autonomy(scenario())
                command = autonomy.update(observation(**case), 0.0, BrainOutput())
                self.assertEqual(command.velocity, (0.0, 0.0, 0.0))
                self.assertEqual(command.mode, "RECOVERY")
                self.assertEqual(autonomy.completed, 0)

    def test_clock_regression_rejects_then_recovers(self):
        autonomy = Autonomy(scenario())
        autonomy.update(observation(now=10.0), 10.0, BrainOutput())
        command = autonomy.update(observation(now=2.0), 2.0, BrainOutput())
        self.assertEqual(command.reason, "clock_reset")
        command = autonomy.update(observation(now=2.2), 2.2, BrainOutput())
        self.assertEqual(command.mode, "INSPECT")

    def test_rejected_sample_cannot_poison_dead_reckoning(self):
        autonomy = Autonomy(scenario())
        autonomy.update(observation(), 0.0, BrainOutput())
        poisoned = observation(0.2, (math.nan, math.nan, math.nan), valid=False,
                               velocity=(math.inf, 0.0, 0.0))
        command = autonomy.update(poisoned, 0.2, BrainOutput())
        self.assertTrue(all(math.isfinite(value) for value in command.velocity))
        self.assertTrue(all(math.isfinite(value) for value in autonomy.tracker.position))
        self.assertEqual(autonomy.completed, 0)
        self.assertIn(command.mode, ("RECOVERY", "BLOCKED"))

    def test_long_gap_reinitializes_observer_from_accepted_sample(self):
        from flybrain_sim.autonomy import ObservationTracker
        tracker = ObservationTracker(0.32)
        tracker.update(observation(), 0.0)
        for tick in range(1, 11):
            tracker.update(None, tick * 0.2)
        recovered = observation(2.2, (4.0, 4.0, 3.0), velocity=(0.1, 0.0, 0.0))
        estimate = tracker.update(recovered, 2.2)
        self.assertEqual(estimate.position, recovered.position)
        self.assertEqual(estimate.velocity, recovered.velocity)

    def test_long_outage_requires_accepted_settling_before_inspection(self):
        s = scenario()
        autonomy = Autonomy(s)
        autonomy.update(observation(0.0, s.waypoints[0]), 0.0, BrainOutput())
        for tick in range(1, 11):
            autonomy.update(observation(tick * 0.2, s.waypoints[0], valid=False),
                            tick * 0.2, BrainOutput())
        command = autonomy.update(observation(2.2, s.waypoints[0]), 2.2, BrainOutput())
        self.assertEqual(command.reason, "localization_reacquisition")
        self.assertEqual(autonomy.completed, 0)
        self.assertTrue(autonomy._reacquiring)
        autonomy.update(observation(2.4, s.waypoints[0], valid=False), 2.4, BrainOutput())
        command = autonomy.update(observation(2.6, s.waypoints[0]), 2.6, BrainOutput())
        self.assertEqual(command.reason, "localization_reacquisition")
        self.assertEqual(autonomy.completed, 0)
        self.assertTrue(autonomy._reacquiring)

    def test_static_escape_leaves_guard_without_crossing_body(self):
        gate = SafetyGate((20.0, 20.0, 12.0), position_uncertainty=0.55)
        box = Box("fixture", (6.0, 1.0, 0.0), (9.0, 8.0, 7.0))
        obs = observation(position=(5.2, 4.0, 3.0))
        command = gate.escape_static(obs, 0.0, (box,))
        self.assertIsNotNone(command)
        self.assertEqual(command.reason, "static_margin_escape")
        self.assertLess(command.velocity[0], 0.0)
        self.assertFalse(segment_intersects_box(obs.position, obs.position, box, gate.radius))

    def test_static_escape_cannot_waive_body_or_other_wall(self):
        gate = SafetyGate((20.0, 20.0, 12.0), position_uncertainty=0.55)
        box = Box("fixture", (6.0, 1.0, 0.0), (9.0, 8.0, 7.0))
        self.assertIsNone(gate.escape_static(observation(position=(5.7, 4.0, 3.0)), 0.0, (box,)))
        opposite = Box("opposite", (1.0, 1.0, 0.0), (4.4, 8.0, 7.0))
        self.assertIsNone(gate.escape_static(observation(position=(5.2, 4.0, 3.0)), 0.0,
                                            (box, opposite)))

    def test_observer_preserves_obstacle_age_when_predicting_to_now(self):
        from flybrain_sim.autonomy import ObservationTracker
        tracker = ObservationTracker(0.02)
        moving = Box("moving", (6.0, 4.0, 0.0), (7.0, 5.0, 3.0), (0.5, 0.0, 0.0))
        aged = observation(0.0, obstacles=(moving,), receive_time=0.6)
        predicted = tracker.update(aged, 0.6)
        self.assertAlmostEqual(predicted.obstacles[0].low[0], 6.3)
        self.assertEqual(predicted.capture_time, 0.6)
        # Later predictions always start from the original capture, not from an
        # already projected box, preventing accidental double extrapolation.
        later = tracker.update(None, 0.8)
        self.assertAlmostEqual(later.obstacles[0].low[0], 6.4)
        self.assertEqual(tracker.last_observation.obstacles[0].low[0], 6.0)

    def test_observer_learns_disturbance_from_accepted_velocity(self):
        from flybrain_sim.autonomy import ObservationTracker
        tracker = ObservationTracker(0.02)
        velocity = (0.0, 0.0, 0.0)
        position = (3.0, 3.0, 3.0)
        tracker.update(observation(), 0.0)
        for step in range(1, 101):
            dt = 0.2
            next_velocity = (velocity[0] + dt * (-velocity[0] / 0.55 + 0.3), 0.0, 0.0)
            position = (position[0] + dt * (velocity[0] + next_velocity[0]) / 2.0, 3.0, 3.0)
            velocity = next_velocity
            tracker.update(observation(step * dt, position, velocity=velocity), step * dt)
        self.assertAlmostEqual(tracker.disturbance[0], 0.3, delta=0.02)
        self.assertAlmostEqual(tracker.compensation[0], 0.165, delta=0.02)

    def test_bad_brain_output_falls_back_without_nan_command(self):
        autonomy = Autonomy(scenario())
        command = autonomy.update(observation(), 0.0,
                                  BrainOutput(speed_scale=math.nan))
        self.assertEqual(autonomy.brain_rejections, 1)
        self.assertTrue(all(math.isfinite(v) for v in command.velocity))
        self.assertGreater(norm(command.velocity), 0.0)
        self.assertLessEqual(norm(command.velocity), 2.5)

    def test_brain_cannot_increase_speed_limit(self):
        gate = SafetyGate((20.0, 20.0, 12.0))
        self.assertLessEqual(gate.brain_scale(BrainOutput(speed_scale=1e9)), 1.0)

    def test_malformed_brain_diagnostics_fail_to_baseline(self):
        gate = SafetyGate((20.0, 20.0, 12.0))
        self.assertEqual(gate.brain_scale(BrainOutput(diagnostics={"bad": "text"})), 1.0)
        self.assertEqual(gate.brain_rejections, 1)

    def test_approaching_moving_fixture_requests_bounded_escape(self):
        gate = SafetyGate((20.0, 20.0, 12.0))
        obs = observation(position=(3.0, 3.0, 3.0))
        approaching = Box("moving", (4.5, 2.0, 0.0), (6.0, 4.0, 6.0),
                          (-0.5, 0.0, 0.0))
        command = gate.escape_moving(obs, 0.0, (approaching,))
        self.assertIsNotNone(command)
        self.assertEqual(command.reason, "moving_obstacle_escape")
        self.assertGreater(norm(command.velocity), 0.0)
        self.assertLessEqual(norm(command.velocity), gate.max_speed + 1e-9)

    def test_receding_fixture_does_not_trigger_escape(self):
        gate = SafetyGate((20.0, 20.0, 12.0))
        receding = Box("moving", (4.5, 2.0, 0.0), (6.0, 4.0, 6.0),
                       (0.5, 0.0, 0.0))
        self.assertIsNone(gate.escape_moving(observation(), 0.0, (receding,)))

    def test_escape_does_not_pass_through_static_enclosure(self):
        gate = SafetyGate((20.0, 20.0, 12.0))
        moving = Box("moving", (4.5, 2.0, 0.0), (6.0, 4.0, 6.0),
                     (-0.5, 0.0, 0.0))
        enclosure = Box("static-solid", (2.0, 2.0, 2.0), (4.0, 4.0, 4.0))
        self.assertIsNone(gate.escape_moving(observation(), 0.0, (moving, enclosure)))

    def test_guard_rejects_collision_segment(self):
        gate = SafetyGate((20.0, 20.0, 12.0))
        obs = observation(position=(4.0, 3.0, 3.0))
        box = Box("wall", (5.0, 1.0, 0.0), (6.0, 6.0, 8.0))
        command = gate.authorize((2.5, 0.0, 0.0), obs, 0.0, "INSPECT", "test", (box,))
        self.assertEqual(command.reason, "predicted_obstacle")
        self.assertEqual(command.velocity, (0.0, 0.0, 0.0))


class MissionTests(unittest.TestCase):
    def test_inspection_requires_dwell_and_return(self):
        s = scenario()
        autonomy = Autonomy(s)
        for now in (0.0, 0.4, 0.81):
            autonomy.update(observation(now, s.waypoints[0]), now, BrainOutput())
        self.assertEqual(autonomy.completed, 1)
        self.assertFalse(autonomy.mission_complete)
        self.assertFalse(autonomy.returned_home)
        for now in (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8):
            autonomy.update(observation(now, s.home), now, BrainOutput())
        self.assertTrue(autonomy.mission_complete)
        self.assertTrue(autonomy.returned_home)

    def test_localization_loss_interrupts_inspection_dwell(self):
        s = scenario()
        autonomy = Autonomy(s)
        autonomy.update(observation(0.0, s.waypoints[0]), 0.0, BrainOutput())
        autonomy.update(observation(0.9, s.waypoints[0], valid=False), 0.9, BrainOutput())
        autonomy.update(observation(1.0, s.waypoints[0]), 1.0, BrainOutput())
        self.assertEqual(autonomy.completed, 0)

    def test_abort_return_is_not_success(self):
        s = scenario()
        autonomy = Autonomy(s)
        for now in (0.0, 0.5, 0.81):
            autonomy.update(observation(now, s.home, battery_wh=0.8), now, BrainOutput())
        self.assertTrue(autonomy.returned_home)
        self.assertFalse(autonomy.mission_complete)
        self.assertEqual(autonomy.completed, 0)
        self.assertEqual(autonomy.mode, "ABORTED_HOME")


class RecordedRegressionTests(unittest.TestCase):
    def test_recorded_wind_and_terminal_failures_complete(self):
        from flybrain_sim.runner import run_episode
        from flybrain_sim.scenarios import generate_scenario
        for seed in (200071, 200059, 200112, 501866, 700489):
            with self.subTest(seed=seed):
                result = run_episode(generate_scenario(seed), "baseline")
                self.assertEqual(result.outcome, "mission_complete")
                self.assertFalse(result.collision)
                self.assertTrue(result.returned_home)


if __name__ == "__main__":
    unittest.main()
