import math
import unittest
from dataclasses import replace

from flybrain_sim.contracts import Box, Scenario, VehicleState
from flybrain_sim.geometry import (
    VEHICLE_RADIUS, clearance, distance, norm, obstacles_at,
    point_box_distance, segment_intersects_box, swept_collision, within_bounds,
)
from flybrain_sim.physics import MAX_ACCELERATION, MAX_SPEED, step
from flybrain_sim.scenarios import CATEGORIES, generate_scenario
from flybrain_sim.sensors import SensorModel


class WorldTests(unittest.TestCase):
    def test_seed_determinism_and_categories(self):
        for seed in range(30):
            self.assertEqual(generate_scenario(seed), generate_scenario(seed))
        self.assertNotEqual(generate_scenario(1).obstacles, generate_scenario(2).obstacles)
        self.assertEqual(len(CATEGORIES), 10)
        with self.assertRaises(ValueError):
            generate_scenario(0, "unsupported")

    def test_low_battery_charge_varies_with_seed_within_declared_range(self):
        charges = [generate_scenario(seed, "low_battery").initial_battery_wh
                   for seed in range(30)]
        self.assertTrue(all(3.2 <= charge <= 4.8 for charge in charges))
        self.assertGreater(len(set(charges)), 1)
        self.assertEqual(charges, [generate_scenario(seed, "low_battery").initial_battery_wh
                                   for seed in range(30)])
        self.assertEqual(generate_scenario(0, "nominal").initial_battery_wh, 22.0)

    def test_start_waypoints_and_overhead_connections_are_free(self):
        for seed in range(100):
            scenario = generate_scenario(seed)
            obstacles = obstacles_at(scenario, 0.0)
            for point in (scenario.home,) + scenario.waypoints:
                self.assertTrue(within_bounds(point, scenario.bounds))
                self.assertGreater(clearance(point, obstacles), VEHICLE_RADIUS)
                overhead = (point[0], point[1], 11.5)
                self.assertFalse(swept_collision(point, overhead, obstacles))
            self.assertTrue(within_bounds((24.0, 18.0, 11.5), scenario.bounds))

    def test_segment_collision_catches_tunneling_and_tangency(self):
        box = Box("wall", (5.0, 0.0, 0.0), (5.1, 5.0, 5.0))
        self.assertTrue(segment_intersects_box((0, 2, 2), (10, 2, 2), box))
        self.assertFalse(segment_intersects_box((0, 7, 2), (10, 7, 2), box))
        self.assertTrue(segment_intersects_box((0, 5.45, 2), (10, 5.45, 2), box))
        self.assertTrue(segment_intersects_box((5.05, 2, 2), (5.05, 2, 2), box))
        self.assertAlmostEqual(point_box_distance((8.1, 2, 2), box), 3.0)
        self.assertFalse(within_bounds((0.1, 1.0, 1.0), (10, 10, 10)))

    def test_moving_obstacles_bounded_and_repeatable(self):
        scenario = generate_scenario(2, "moving_obstacle")
        base = scenario.obstacles[-1]
        for t in (0.0, 2.0, 15.0, 30.0, 100.0):
            box = obstacles_at(scenario, t)[-1]
            self.assertEqual(box, obstacles_at(scenario, t)[-1])
            self.assertLessEqual(abs(box.low[0] - base.low[0]), 3.0 + 1e-10)
        self.assertNotEqual(obstacles_at(scenario, 0), obstacles_at(scenario, 2))

    def test_dynamics_acceleration_speed_energy_and_purity(self):
        scenario = generate_scenario(3, "nominal")
        state = VehicleState(0.0, scenario.home, (0, 0, 0), 22)
        initial_position = state.position
        first = step(state, (100, 0, 0), scenario)
        self.assertEqual(state.position, initial_position)
        self.assertAlmostEqual(norm(first.velocity), MAX_ACCELERATION * scenario.dt)
        self.assertAlmostEqual(first.position[0] - state.position[0], 0.06)
        previous = state
        for _ in range(100):
            result = step(previous, (100, 0, 0), scenario)
            self.assertLessEqual(norm(result.velocity), MAX_SPEED + 1e-9)
            self.assertLessEqual(distance(result.velocity, previous.velocity),
                                 MAX_ACCELERATION * scenario.dt + 1e-9)
            self.assertGreater(result.energy_used_wh, previous.energy_used_wh)
            self.assertAlmostEqual(result.battery_wh + result.energy_used_wh, 22)
            previous = result
        self.assertTrue(all(math.isfinite(v) for v in step(state, (math.nan, 0, 0), scenario).position))

    def test_fault_schedule_is_paired_and_time_indexed(self):
        scenario = generate_scenario(10, "sensor_dropout")
        a, b = SensorModel(scenario), SensorModel(scenario)
        invalid = 0
        for tick in range(250):
            state = VehicleState(tick * scenario.dt, scenario.home, (0, 0, 0), 22)
            one, two = a.observe(state), b.observe(state)
            self.assertEqual(one, two)
            if not one.valid:
                invalid += 1
                self.assertGreaterEqual(state.time, scenario.fault_start)
                self.assertLess(state.time, scenario.fault_start + scenario.fault_duration)
        self.assertGreater(invalid, 0)
        state = VehicleState(scenario.fault_start + 1, scenario.home, (0, 0, 0), 22)
        self.assertEqual(SensorModel(scenario).observe(state), SensorModel(scenario).observe(state))

    def test_latency_and_one_clock_reset_event(self):
        scenario = generate_scenario(0, "latency")
        sensor = SensorModel(scenario)
        for tick in range(110):
            state = VehicleState(tick * scenario.dt, scenario.home, (0, 0, 0), 22)
            observed = sensor.observe(state)
            if scenario.fault_start + 1 <= state.time < scenario.fault_start + scenario.fault_duration:
                self.assertAlmostEqual(observed.receive_time - observed.capture_time,
                                       scenario.latency_steps * scenario.dt)
        scenario = generate_scenario(0, "clock_reset")
        sensor = SensorModel(scenario)
        resets = 0
        previous_timestamp = -1.0
        backwards = 0
        for tick in range(160):
            state = VehicleState(tick * scenario.dt, scenario.home, (0, 0, 0), 22)
            observed = sensor.observe(state)
            resets += observed.fault == "clock_reset"
            backwards += observed.capture_time < previous_timestamp
            previous_timestamp = observed.capture_time
        self.assertEqual(resets, 1)
        self.assertEqual(backwards, 1)

    def test_compute_stall_and_dynamic_sensor_range(self):
        scenario = generate_scenario(4, "compute_stall")
        sensor = SensorModel(scenario)
        state = VehicleState(scenario.fault_start + 1, scenario.home, (0, 0, 0), 22)
        self.assertFalse(sensor.observe(state).valid)
        after = replace(state, time=scenario.fault_start + scenario.fault_duration + 1)
        self.assertTrue(sensor.observe(after).valid)
        scenario = replace(generate_scenario(4, "moving_obstacle"), sensor_range=1.0)
        sensor = SensorModel(scenario)
        state = VehicleState(0, scenario.home, (0, 0, 0), 22)
        observed = sensor.observe(state)
        self.assertNotIn("moving-inspection-fixture", {box.id for box in observed.obstacles})
        self.assertEqual(len(observed.obstacles), len(scenario.obstacles) - 1)
        box = obstacles_at(scenario, 0)[-1]
        nearby = replace(state, position=(box.low[0] - 0.2, box.low[1], 3.0))
        self.assertIn("moving-inspection-fixture", {b.id for b in sensor.observe(nearby).obstacles})


if __name__ == "__main__":
    unittest.main()
