"""Independent course validity and motion checks; no controller tuning."""
import math
import unittest

from flybrain_sim.contracts import Box
from stress.course import (
    APERTURES, BOUNDS, HOME, PROFILES, REFERENCE_ROUTE, STATIC_OBSTACLES,
    WAYPOINTS, _segment_box_distance, make_trial, obstacles_at_trial,
    reference_certificate, static_geometry_sha256,
)


class CourseTests(unittest.TestCase):
    def test_static_course_is_identical_across_profiles_and_seeds(self):
        for seed in (10, 11, 20):
            for profile in PROFILES:
                scenario = make_trial(seed, profile)
                self.assertEqual(tuple(b for b in scenario.obstacles if b.velocity == (0, 0, 0)), STATIC_OBSTACLES)
                self.assertEqual(scenario.home, HOME)
                self.assertEqual(scenario.waypoints, WAYPOINTS)
                self.assertEqual(scenario.bounds, BOUNDS)

    def test_invalid_inputs_rejected(self):
        for seed in (True, 1.2, "10"):
            with self.assertRaises(TypeError):
                make_trial(seed)
        with self.assertRaises(ValueError):
            make_trial(10, "unknown")
        with self.assertRaises(ValueError):
            obstacles_at_trial(make_trial(10), math.nan)

    def test_reference_is_independently_clear(self):
        certificate = reference_certificate()
        self.assertTrue(certificate["all_segments_clear"])
        self.assertTrue(certificate["complete_round_trip"])
        self.assertGreater(certificate["minimum_body_clearance_m"], 1.0)
        self.assertGreater(certificate["minimum_conservative_cube_clearance_lower_bound_m"], 0.7)
        self.assertGreater(certificate["length_m"], 150.0)
        self.assertEqual(REFERENCE_ROUTE[0], REFERENCE_ROUTE[-1])

    def test_exact_segment_distance_has_interior_minimum(self):
        box = Box("test", (0, 0, 0), (1, 1, 1))
        self.assertAlmostEqual(_segment_box_distance((-2, 2, 0.5), (3, 2, 0.5), box), 1)
        self.assertAlmostEqual(_segment_box_distance((-2, -2, 0.5), (2, 2, 0.5), box), 0)
        self.assertAlmostEqual(_segment_box_distance((-1, -1, -1), (-1, -1, -1), box), math.sqrt(3))
        self.assertAlmostEqual(_segment_box_distance((2, 2, 0.5), (3, 3, 0.5), box), math.sqrt(2))

    def test_apertures_cannot_be_flown_over_or_around(self):
        # Sample each complete partition cross-section. Every point outside its
        # advertised hole is physically occupied; no flight-radius assumptions.
        for gate in APERTURES:
            pieces = [box for box in STATIC_OBSTACLES if box.id.startswith(gate["id"])]
            for yi in range(73):
                for zi in range(41):
                    y, z = yi * 0.5, zi * 0.25
                    in_aperture = (abs(y - gate["y"]) < gate["width"] / 2
                                   and abs(z - gate["z"]) < gate["height"] / 2)
                    occupied = any(box.low[1] <= y <= box.high[1]
                                   and box.low[2] <= z <= box.high[2] for box in pieces)
                    self.assertEqual(occupied, not in_aperture)

    def test_dynamic_fixture_remains_in_room_and_has_correct_velocity(self):
        scenario = make_trial(12, "moving_obstacle")
        h = 1e-4
        for t in range(301):
            box = obstacles_at_trial(scenario, float(t))[-1]
            self.assertGreater(box.low[0], 12.4)
            self.assertLess(box.high[0], 23.6)
            self.assertGreaterEqual(box.low[1], 10.5 - 1e-10)
            self.assertLessEqual(box.high[1], 21.5 + 1e-10)
            self.assertGreaterEqual(box.low[2], 0.4)
            self.assertLess(box.high[2], 9.6)
            if t:
                before = obstacles_at_trial(scenario, t - h)[-1]
                after = obstacles_at_trial(scenario, t + h)[-1]
                derivative = (after.low[1] - before.low[1]) / (2 * h)
                self.assertAlmostEqual(derivative, box.velocity[1], places=6)

    def test_trial_determinism_and_fault_variety(self):
        for profile in PROFILES:
            self.assertEqual(make_trial(10, profile), make_trial(10, profile))
            self.assertNotEqual(make_trial(10, profile).fault_start, make_trial(11, profile).fault_start)
        self.assertEqual(len(static_geometry_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
