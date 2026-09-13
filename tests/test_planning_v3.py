"""Geometric regressions independent of the benchmark's witness route."""
from dataclasses import replace
import itertools
import math
import unittest

from flybrain_sim.contracts import Box, Scenario
from flybrain_sim.planning import VisibilityPlanner
from stress.course import make_trial


def enclosure(bounds):
    boxes = []
    for axis in range(3):
        for side in (0, 1):
            low, high = [0.] * 3, list(bounds)
            if side:
                low[axis] = bounds[axis] - .3
            else:
                high[axis] = .3
            boxes.append(Box(f"shell-{axis}-{side}", tuple(low), tuple(high)))
    return tuple(boxes)


def partition(bounds, x, y_interval, z_interval, name):
    y0, y1 = y_interval
    z0, z1 = z_interval
    return (
        Box(name + "-bottom", (x - .5, 0., 0.), (x + .5, bounds[1], z0)),
        Box(name + "-top", (x - .5, 0., z1), (x + .5, bounds[1], bounds[2])),
        Box(name + "-left", (x - .5, 0., z0), (x + .5, y0, z1)),
        Box(name + "-right", (x - .5, y1, z0), (x + .5, bounds[1], z1)),
    )


def unseen_opening_scenario():
    bounds = (35., 27., 19.)
    boxes = enclosure(bounds) + partition(bounds, 17.3, (15.1, 19.3), (8.3, 12.3), "opening")
    return Scenario(77, "planner_test", bounds, (4., 5., 4.), ((30., 7., 5.),), boxes,
                    sensor_noise=0.)


def permute_scenario(scenario, permutation):
    def transform(point):
        return tuple(point[i] for i in permutation)
    return replace(scenario, bounds=transform(scenario.bounds), home=transform(scenario.home),
                   waypoints=tuple(transform(p) for p in scenario.waypoints),
                   obstacles=tuple(Box(b.id, transform(b.low), transform(b.high))
                                   for b in scenario.obstacles))


def intersects_expanded_box_independently(start, end, box, margin):
    # Independent interval clipping check; does not invoke planner or simulator geometry.
    enter, leave = 0., 1.
    for coordinate in range(3):
        low, high = box.low[coordinate] - margin, box.high[coordinate] + margin
        origin, delta = start[coordinate], end[coordinate] - start[coordinate]
        if abs(delta) < 1e-12:
            if not low <= origin <= high:
                return False
        else:
            crossings = sorted(((low - origin) / delta, (high - origin) / delta))
            enter, leave = max(enter, crossings[0]), min(leave, crossings[1])
            if enter > leave:
                return False
    return True


class PlanningV3Tests(unittest.TestCase):
    def assert_route_clear(self, planner, start, goal, boxes=()):
        route = planner.plan(start, goal, boxes)
        self.assertIsNotNone(route, (start, goal))
        self.assertEqual(route[-1], goal)
        for a, b in zip([start] + route, route):
            for obstacle in planner.static + tuple(boxes):
                self.assertFalse(intersects_expanded_box_independently(a, b, obstacle, planner.clearance),
                                 (a, b, obstacle.id))
            for point in (a, b):
                self.assertTrue(all(planner.radius <= point[i] <= planner.bounds[i] - planner.radius
                                    for i in range(3)))
        return route

    def test_main_course_all_legs_and_full_return_at_unchanged_margins(self):
        for noise in (0.02, 0.20, 0.48):
            scenario = replace(make_trial(10), sensor_noise=noise)
            planner = VisibilityPlanner(scenario, margin=.5 + math.sqrt(3.) * noise)
            current = scenario.home
            for goal in (*scenario.waypoints, scenario.home):
                self.assert_route_clear(planner, current, goal)
                current = goal
            self.assertLess(len(planner._static_nodes) + 2, planner.MAX_GRAPH_NODES)

    def test_shifted_opening_all_six_axis_permutations(self):
        for permutation in itertools.permutations(range(3)):
            with self.subTest(permutation=permutation):
                scenario = permute_scenario(unseen_opening_scenario(), permutation)
                planner = VisibilityPlanner(scenario)
                self.assert_route_clear(planner, scenario.home, scenario.waypoints[0])
                self.assert_route_clear(planner, scenario.waypoints[0], scenario.home)

    def test_successive_openings_at_different_heights(self):
        bounds = (35., 27., 19.)
        obstacles = (enclosure(bounds)
                     + partition(bounds, 11., (4.9, 9.1), (4., 8.), "lower")
                     + partition(bounds, 24., (16.9, 21.1), (11., 15.), "upper"))
        scenario = Scenario(80, "planner_test", bounds, (4., 18., 4.), ((30., 7., 4.),), obstacles)
        planner = VisibilityPlanner(scenario)
        self.assert_route_clear(planner, scenario.home, scenario.waypoints[0])
        self.assert_route_clear(planner, scenario.waypoints[0], scenario.home)

    def test_fully_sealed_partition_has_no_route(self):
        scenario = unseen_opening_scenario()
        obstacles = enclosure(scenario.bounds) + (Box("sealed", (17., 0., 0.), (18., 27., 19.)),)
        planner = VisibilityPlanner(replace(scenario, obstacles=obstacles))
        self.assertIsNone(planner.plan(scenario.home, scenario.waypoints[0]))

    def test_sub_clearance_opening_is_not_made_passable(self):
        scenario = unseen_opening_scenario()
        obstacles = enclosure(scenario.bounds) + partition(scenario.bounds, 17., (16., 17.8), (8., 12.), "narrow")
        planner = VisibilityPlanner(replace(scenario, obstacles=obstacles))
        self.assertEqual(planner.clearance, .95)
        self.assertIsNone(planner.plan(scenario.home, scenario.waypoints[0]))

    def test_dynamic_box_blocking_only_opening_cannot_be_ignored(self):
        scenario = unseen_opening_scenario()
        planner = VisibilityPlanner(scenario)
        self.assert_route_clear(planner, scenario.home, scenario.waypoints[0])
        blocker = Box("observed-blocker", (16.8, 15.1, 8.3), (17.8, 19.3, 12.3), (0., .2, 0.))
        self.assertIsNone(planner.plan(scenario.home, scenario.waypoints[0], (blocker,)))

    def test_large_map_is_explicitly_bounded(self):
        scenario = unseen_opening_scenario()
        obstacles = tuple(Box(str(i), (10., 10., 10.), (11., 11., 11.))
                          for i in range(VisibilityPlanner.MAX_MAP_BOXES + 1))
        planner = VisibilityPlanner(replace(scenario, obstacles=obstacles))
        self.assertEqual(planner._static_nodes, ())
        self.assertIsNone(planner.plan(scenario.home, scenario.waypoints[0]))

    def test_route_is_deterministic_and_nonfinite_inputs_fail(self):
        scenario = unseen_opening_scenario()
        planner = VisibilityPlanner(scenario)
        route = planner.plan(scenario.home, scenario.waypoints[0])
        self.assertEqual(route, VisibilityPlanner(scenario).plan(scenario.home, scenario.waypoints[0]))
        self.assertIsNone(planner.plan((math.nan, 5., 4.), scenario.waypoints[0]))
        self.assertIsNone(planner.plan(scenario.home, (math.inf, 7., 5.)))


if __name__ == "__main__":
    unittest.main()
