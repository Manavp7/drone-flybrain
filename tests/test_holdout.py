"""Geometry/protocol validation only: no held-out controller executions."""
from dataclasses import asdict
import math
import unittest

from flybrain_sim.contracts import Box
from stress.course import make_trial as make_fault_trial
from validation.holdout import (
    PROFILES, SEED_START, SEED_END, apertures_for_seed, course_certificate,
    course_manifest, geometry_sha256, make_blocked_trial, make_trial,
    obstacles_at_trial, validate_suite, validation_manifest,
    _expanded_box_intersects_segment,
)


class HoldoutTests(unittest.TestCase):
    def test_every_registered_geometry_has_a_clear_witness(self):
        report = validate_suite()
        self.assertTrue(report["valid"])
        self.assertEqual(report["courses"], 500)
        self.assertEqual(report["unique_static_geometries"], 500)
        self.assertTrue(report["all_courses_body_clear"])
        self.assertTrue(report["all_courses_planning_clear"])
        self.assertGreater(report["minimum_body_clearance_m"], 1.6)
        self.assertLess(report["witness_length_range_m"][1] / 2.5, 90.0)
        self.assertFalse(report["controller_executed"])

    def test_registered_profiles_seeds_and_counts(self):
        manifest = validation_manifest()
        assignments = manifest["trial_assignments"]
        self.assertEqual([item["seed"] for item in assignments], list(range(SEED_START, SEED_END + 1)))
        for profile in PROFILES:
            self.assertEqual(sum(item["profile"] == profile for item in assignments), 50)
        self.assertFalse(manifest["reference_supplied_to_controller"])
        self.assertFalse(manifest["negative_control"]["included_in_500_trials"])

    def test_shape_dimensions_vary_not_only_orientation(self):
        manifests = [course_manifest(seed) for seed in range(81001, 81065)]
        for key in ("room_lengths", "canonical_bounds", "canonical_apertures"):
            self.assertGreater(len({repr(item["layout_parameters"][key]) for item in manifests}), 20)
        self.assertEqual(len({(item["layout_parameters"]["swap_xy"], item["layout_parameters"]["reflect_x"],
                              item["layout_parameters"]["reflect_y"]) for item in manifests}), 8)
        for item in manifests:
            self.assertTrue(item["witness"]["all_segments_planning_clear"])

    def test_same_seed_is_reproducible_and_profiles_do_not_change_geometry(self):
        for seed in (82001, 82002):
            nominal = make_trial(seed)
            for profile in PROFILES:
                scenario = make_trial(seed, profile)
                self.assertEqual(asdict(scenario), asdict(make_trial(seed, profile)))
                self.assertEqual(scenario.home, nominal.home)
                self.assertEqual(scenario.waypoints, nominal.waypoints)
                self.assertEqual(tuple(box for box in scenario.obstacles if box.velocity == (0.0, 0.0, 0.0)),
                                 nominal.obstacles)
                self.assertNotIn("witness", asdict(scenario))

    def test_fault_severity_contract_matches_original_profiles(self):
        for profile in PROFILES:
            scenario, original = make_trial(83001, profile), make_fault_trial(83001, profile)
            for name in ("sensor_noise", "dropout_probability", "latency_steps", "fault_start", "fault_duration",
                         "initial_battery_wh", "max_time", "dt", "sensor_range"):
                self.assertEqual(getattr(scenario, name), getattr(original, name))
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in scenario.wind)),
                                   math.sqrt(sum(v * v for v in original.wind)))

    def test_motion_stays_inside_static_structure_without_penetration(self):
        directions = set()
        for seed in range(84001, 84021):
            scenario = make_trial(seed, "moving_obstacle")
            original = scenario.obstacles[-1]
            directions.add(tuple(0 if abs(v) < 1e-10 else 1 if v > 0 else -1 for v in original.velocity))
            for t in range(1, 301, 7):
                box = obstacles_at_trial(scenario, t)[-1]
                for static in scenario.obstacles[:-1]:
                    penetrating = all(min(box.high[i], static.high[i]) - max(box.low[i], static.low[i]) > 1e-9
                                      for i in range(3))
                    self.assertFalse(penetrating, (seed, t, static.id))
                before, after = obstacles_at_trial(scenario, t - 1e-4)[-1], obstacles_at_trial(scenario, t + 1e-4)[-1]
                for axis in range(3):
                    derivative = (after.low[axis] - before.low[axis]) / 2e-4
                    self.assertAlmostEqual(derivative, box.velocity[axis], places=6)
                self.assertLessEqual(math.dist(original.low, box.low), 4.5 + 1e-9)
        self.assertEqual(len(directions), 4)

    def test_aperture_centers_clear_and_closed_partition_cannot_be_crossed(self):
        for seed in (85001, 85002, 85003, 85004):
            scenario = make_trial(seed)
            gates = apertures_for_seed(seed)
            for gate in gates:
                axis = 0 if gate["partition_axis"] == "x" else 1
                a, b = list(gate["center"]), list(gate["center"])
                a[axis] -= 2
                b[axis] += 2
                self.assertFalse(any(_expanded_box_intersects_segment(tuple(a), tuple(b), box, 0.95)
                                     for box in scenario.obstacles))
            blocked = make_blocked_trial(seed)
            negative = course_certificate(seed, blocked=True)
            self.assertFalse(negative["all_segments_clear"])
            self.assertNotEqual(negative["static_geometry_sha256"], geometry_sha256(seed))
            gate = gates[1]
            barrier = blocked.obstacles[-1]
            axis = 0 if gate["partition_axis"] == "x" else 1
            lateral_axis = 1 - axis
            self.assertLess(barrier.low[axis], gate["center"][axis])
            self.assertGreater(barrier.high[axis], gate["center"][axis])
            floor = next(box for box in blocked.obstacles if box.id == "floor")
            roof = next(box for box in blocked.obstacles if box.id == "roof")
            self.assertEqual(barrier.low[lateral_axis], floor.low[lateral_axis])
            self.assertEqual(barrier.high[lateral_axis], floor.high[lateral_axis])
            self.assertEqual(barrier.low[2], floor.low[2])
            self.assertEqual(barrier.high[2], roof.high[2])

    def test_independent_expanded_slab_predicate_edge_cases(self):
        box = Box("unit", (0, 0, 0), (1, 1, 1))
        self.assertTrue(_expanded_box_intersects_segment((-2, 0.5, 0.5), (2, 0.5, 0.5), box, 0.0))
        self.assertTrue(_expanded_box_intersects_segment((-2, 1.5, 0.5), (2, 1.5, 0.5), box, 0.5))
        self.assertFalse(_expanded_box_intersects_segment((-2, 1.5001, 0.5), (2, 1.5001, 0.5), box, 0.5))
        self.assertFalse(_expanded_box_intersects_segment((3, 2, 2), (4, 2, 2), box, 0.5))

    def test_invalid_inputs_fail_explicitly(self):
        for seed in (True, "1", 1.0):
            with self.assertRaises(TypeError):
                make_trial(seed)
        with self.assertRaises(ValueError):
            make_trial(86001, "unknown")
        with self.assertRaises(ValueError):
            obstacles_at_trial(make_trial(86001), math.inf)


if __name__ == "__main__":
    unittest.main()
