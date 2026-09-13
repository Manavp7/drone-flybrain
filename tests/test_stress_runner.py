from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import patch

from flybrain_sim.contracts import Box, Scenario, VehicleState
from flybrain_sim.geometry import obstacles_at
from flybrain_sim.runner import run_episode
from flybrain_sim.scenarios import CATEGORIES, generate_scenario
from flybrain_sim.sensors import SensorModel
from stress.runner import run_trial
from stress.sensors import StressSensorModel


def simple_scenario(**kwargs):
    base = Scenario(17, "nominal", (20., 20., 10.), (2., 2., 3.), ((15., 15., 3.),),
                    (Box("wall", (8., 7., 0.), (9., 12., 6.)),), sensor_noise=0.0, max_time=1.)
    return replace(base, **kwargs)


class StressRunnerTests(unittest.TestCase):
    def test_base_sensor_parity_across_all_fault_categories(self):
        for category in CATEGORIES:
            scenario = replace(generate_scenario(177, category), fault_start=0.4, fault_duration=1.0)
            expected, actual = SensorModel(scenario), StressSensorModel(scenario, {"sensor_mode": "base"})
            for tick in range(12):
                state = VehicleState(tick * 0.2, (3. + tick * .1, 3., 3.), (.5, 0., 0.), 20.)
                self.assertEqual(expected.observe(state), actual.observe(state), (category, tick))

    def test_core_scoring_parity_all_fault_categories(self):
        for category in CATEGORIES:
            scenario = replace(generate_scenario(171, category), max_time=8., fault_start=1., fault_duration=5.)
            original = asdict(run_episode(scenario))
            actual = run_trial(scenario, {"sensor_mode": "base"}, obstacle_function=obstacles_at)["result"]
            for key in ("trajectory", "events", "wall_seconds"):
                original.pop(key)
                actual.pop(key, None)
            self.assertEqual(original, actual, category)

    def test_timeout_preserves_final_integrated_endpoint(self):
        trial = run_trial(simple_scenario(max_time=0.2), {}, obstacle_function=obstacles_at)
        self.assertEqual(trial["result"]["outcome"], "timeout")
        self.assertEqual([item["t"] for item in trial["trace"]], [0., .2])
        self.assertEqual(trial["trace"][-1]["sample_kind"], "terminal")
        self.assertIsNone(trial["trace"][-1]["observation"])
        self.assertEqual(trial["trace"][-1]["terminal_outcome"], "timeout")
        self.assertEqual(trial["events"][-1]["outcome"], "timeout")

    def test_home_completion_annotates_without_duplicate_timestamp(self):
        scenario = simple_scenario(waypoints=(), max_time=3.)
        trial = run_trial(scenario, {"sensor_mode": "base"}, obstacle_function=obstacles_at)
        self.assertEqual(trial["result"]["outcome"], "mission_complete")
        times = [item["t"] for item in trial["trace"]]
        self.assertEqual(len(times), len(set(times)))
        self.assertEqual(trial["trace"][-1]["terminal_outcome"], "mission_complete")

    def test_collision_contact_and_endpoint_are_retained(self):
        scenario = simple_scenario(max_time=.2)
        endpoint = VehicleState(.2, (8.5, 8., 3.), (0., 0., 0.), 21., .01)
        with patch("stress.runner.step", return_value=endpoint):
            trial = run_trial(scenario, {}, obstacle_function=obstacles_at)
        self.assertTrue(trial["result"]["collision"])
        self.assertEqual(trial["contacts"][0]["obstacle_id"], "wall")
        self.assertEqual(trial["trace"][-1]["p"], [8.5, 8., 3.])
        self.assertEqual(trial["trace"][-1]["terminal_outcome"], "collision")

    def test_geofence_endpoint_is_retained(self):
        endpoint = VehicleState(.2, (.1, 2., 3.), (0., 0., 0.), 21., .01)
        with patch("stress.runner.step", return_value=endpoint):
            trial = run_trial(simple_scenario(max_time=.2), {}, obstacle_function=obstacles_at)
        self.assertEqual(trial["result"]["outcome"], "geofence_violation")
        self.assertFalse(trial["trace"][-1]["inside_bounds"])

    def test_battery_depletion_endpoint_is_retained(self):
        trial = run_trial(simple_scenario(initial_battery_wh=.00001), {}, obstacle_function=obstacles_at)
        self.assertEqual(trial["result"]["outcome"], "battery_depleted")
        self.assertEqual(trial["trace"][-1]["battery_wh"], 0.)

    def test_static_map_prior_and_centerline_dynamic_occlusion(self):
        wall = Box("occluder", (4., 1., 0.), (4.5, 3., 6.))
        moving = Box("moving", (6., 1.5, 2.5), (7., 2.5, 3.5), (0., 1., 0.))
        scenario = simple_scenario(obstacles=(wall, moving))
        model = StressSensorModel(scenario)
        observation = model.observe(VehicleState(0., scenario.home, (0., 0., 0.), 22.))
        self.assertEqual([box.id for box in observation.obstacles], ["occluder"])
        self.assertEqual(model.last_metadata["hidden_dynamic"][0]["reason"], "static_occlusion")
        clear_model = StressSensorModel(scenario, {"dynamic_occlusion": False})
        self.assertEqual(len(clear_model.observe(VehicleState(0., scenario.home, (0., 0., 0.), 22.)).obstacles), 2)

    def test_dynamic_range_and_capture_time_visibility(self):
        moving = Box("moving", (6., 1.5, 2.5), (7., 2.5, 3.5), (0., 1., 0.))
        scenario = simple_scenario(obstacles=(moving,), latency_steps=2, fault_start=0., fault_duration=10.)
        model = StressSensorModel(scenario, {"dynamic_occlusion": False})
        first = model.observe(VehicleState(0., (2., 2., 3.), (0., 0., 0.), 22.))
        delayed = model.observe(VehicleState(.2, (19., 19., 3.), (0., 0., 0.), 22.))
        self.assertEqual(delayed.capture_time, 0.)
        self.assertEqual(delayed.obstacles, first.obstacles)
        self.assertEqual(model.last_metadata["capture_truth_time"], 0.)
        model.observe(VehicleState(.4, (19., 19., 3.), (0., 0., 0.), 22.))
        model.observe(VehicleState(.6, (19., 19., 3.), (0., 0., 0.), 22.))
        self.assertEqual(model.last_metadata["hidden_dynamic"][0]["reason"], "range")

    def test_trace_is_finite_json_and_invalid_observation_rejected(self):
        scenario = simple_scenario(category="compute_stall", fault_start=0., fault_duration=3.)
        trial = run_trial(scenario, {}, obstacle_function=obstacles_at)
        json.dumps(trial, allow_nan=False)
        first = trial["trace"][0]
        self.assertFalse(first["observation"]["valid"])
        self.assertFalse(first["observation"]["accepted"])
        self.assertIsNone(first["tracker"]["last_valid"])
        self.assertEqual(trial["diagnostics"]["rejected_observation_seconds"], 1.)

    def test_record_flag_does_not_change_functional_metrics(self):
        scenario = simple_scenario()
        first = run_trial(scenario, {}, record=True, obstacle_function=obstacles_at)
        second = run_trial(scenario, {}, record=False, obstacle_function=obstacles_at)
        for value in (first, second):
            value["result"].pop("wall_seconds")
        self.assertEqual(first["result"], second["result"])
        self.assertFalse(second["trace"])
        self.assertTrue(second["events"])


if __name__ == "__main__":
    unittest.main()
