"""Budget regressions: candidate pruning must not discard collision geometry."""
import unittest

from flybrain_sim.contracts import Box, Scenario
from flybrain_sim.planning import VisibilityPlanner
from stress.runner import run_trial
from validation.holdout import make_trial, obstacles_at_trial
from test_planning_v3 import intersects_expanded_box_independently


class TenNodePlanner(VisibilityPlanner):
    MAX_GRAPH_NODES = 10


def small_map():
    return Scenario(1, "budget_regression", (30., 20., 12.), (3., 10., 5.),
                    ((27., 10., 5.),),
                    (Box("wall", (14., 5., 0.), (16., 15., 9.)),))


def unbudgeted_count(planner, extra):
    candidates = (planner._static_nodes + planner._corners(extra)
                  + planner._portals(planner.static + extra, {b.id for b in extra}))
    return 2 + sum(planner.segment_clear(p, p, extra) for p in dict.fromkeys(candidates))


class BudgetPlanningTests(unittest.TestCase):
    def assert_clear(self, planner, start, goal, extra):
        route = planner.plan(start, goal, extra)
        self.assertIsNotNone(route)
        self.assertEqual(route[-1], goal)
        for a, b in zip([start] + route, route):
            for obstacle in planner.static + extra:
                self.assertFalse(intersects_expanded_box_independently(a, b, obstacle, planner.clearance),
                                 (a, b, obstacle.id))
        return route

    def test_dynamic_augmentation_keeps_usable_static_route(self):
        scenario = small_map()
        planner = TenNodePlanner(scenario)
        extra = (Box("moving", (3., 15., 2.), (5., 17., 5.), (0., .2, 0.)),)
        self.assertLessEqual(len(planner._static_nodes) + 2, planner.MAX_GRAPH_NODES)
        self.assertGreater(unbudgeted_count(planner, extra), planner.MAX_GRAPH_NODES)
        self.assert_clear(planner, scenario.home, scenario.waypoints[0], extra)
        self.assertEqual(planner.MAX_GRAPH_NODES, 10)

    def test_overflow_does_not_ignore_obstacle_on_shorter_route(self):
        scenario = small_map()
        planner = TenNodePlanner(scenario)
        extra = (Box("south-passage-blocker", (10., .5, 0.), (19., 5., 12.), (0., .2, 0.)),
                 Box("distant-moving", (3., 15., 2.), (5., 17., 5.), (0., .2, 0.)))
        self.assertGreater(unbudgeted_count(planner, extra), planner.MAX_GRAPH_NODES)
        route = self.assert_clear(planner, scenario.home, scenario.waypoints[0], extra)
        self.assertTrue(any(p[1] > 15.95 for p in route))
        self.assertEqual(route, TenNodePlanner(scenario).plan(scenario.home, scenario.waypoints[0], tuple(reversed(extra))))

    def test_observed_box_closing_all_passages_still_blocks_route(self):
        scenario = small_map()
        planner = TenNodePlanner(scenario)
        extra = (Box("sealed", (13., 0., 0.), (17., 20., 12.), (0., .2, 0.)),)
        self.assertIsNone(planner.plan(scenario.home, scenario.waypoints[0], extra))

    def test_oversized_static_graph_remains_explicit_failure(self):
        scenario = small_map()
        planner = TenNodePlanner(scenario)
        planner.MAX_GRAPH_NODES = 9
        self.assertGreater(len(planner._static_nodes) + 2, planner.MAX_GRAPH_NODES)
        self.assertIsNone(planner.plan(scenario.home, scenario.waypoints[0]))

    def test_exposed_layout_timeout_is_now_a_completed_development_regression(self):
        # This seed was inspected to diagnose the old timeout; it is no longer
        # held-out evidence. Fresh campaign seeds must remain separate.
        run = run_trial(make_trial(2200186, "moving_obstacle"), "moving_obstacle",
                        record=False, obstacle_function=obstacles_at_trial)
        result = run["result"]
        self.assertTrue(result["mission_complete"])
        self.assertTrue(result["returned_home"])
        self.assertEqual(result["waypoints_completed"], 3)
        self.assertFalse(result["collision"])
        self.assertFalse(result["geofence_violation"])


if __name__ == "__main__":
    unittest.main()
