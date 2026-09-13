"""Regression gates for split coverage, scoring, and experimental timing."""
import unittest
import numpy as np
from experiments.fly_tracking_v2_data import specs, generate, DT
from experiments.fly_tracking_v2 import align_trace, score


class TrackingV2Tests(unittest.TestCase):
    def test_seeds_are_disjoint_and_training_direction_size_are_not_locked(self):
        definitions=specs();self.assertEqual(len(definitions),36)
        seeds=[s["seed"] for s in definitions];self.assertEqual(len(set(seeds)),36)
        training=[s for s in definitions if s["split"]=="train"]
        for kind in {s["kind"] for s in training}:
            group=[s for s in training if s["kind"]==kind]
            self.assertEqual({s["sign"] for s in group},{-1,1})
            self.assertEqual({s["sign_y"] for s in group},{-1,1})
            self.assertEqual({s["size"] for s in group},{80,96,112})

    def test_upward_motion_and_quantized_finite_unit_range(self):
        spec=next(s for s in specs() if s["split"]=="train" and s["kind"]=="vertical" and s["sign_y"]<0)
        clip=generate(spec)
        self.assertLess(clip["truth_boxes"][-1,1],clip["truth_boxes"][0,1])
        self.assertEqual(clip["frames"].dtype,np.float32)
        self.assertTrue(np.isfinite(clip["frames"]).all())
        self.assertTrue(np.all((clip["frames"]>=0)&(clip["frames"]<=1)))
        np.testing.assert_allclose(np.diff(clip["truth_boxes"][:,:2],axis=0),clip["velocities"][:-1]*DT,atol=1e-12)

    def test_stationary_gate_checks_full_run_and_no_static_division(self):
        clip=generate(next(s for s in specs() if s["split"]=="test" and s["kind"]=="stationary"))
        boxes=clip["truth_boxes"].copy();boxes[100:,0::2]+=6
        result=score({"boxes":boxes,"statuses":np.full(len(boxes),"tracking")},clip)
        self.assertFalse(result["acceptance_pass"])
        self.assertIsNone(result["improvement_vs_static"])
        self.assertEqual(result["max_center_error"],6)

    def test_uncertain_correct_box_never_counts_tracking_hit(self):
        clip=generate(next(s for s in specs() if s["split"]=="test" and s["kind"]=="stationary"))
        result=score({"boxes":clip["truth_boxes"],"statuses":np.full(len(clip["times"]),"uncertain")},clip)
        self.assertEqual(result["fraction_active_iou_ge_0_5"],0)
        self.assertFalse(result["acceptance_pass"])

    def test_initial_template_cannot_appear_before_neural_response(self):
        clip={"kind":"synthetic","truth_boxes":np.array([[0,0,10,10]]*4),"times":np.array([0.,.01,.02,.04])}
        records=[{"box_xyxy":[2,0,12,10],"status":"tracking"},{"box_xyxy":[4,0,14,10],"status":"uncertain"}]
        meta={"response_times":[.02,.04],"stimulus_times":[0.,.02],"input_indices":[0,2]}
        result=align_trace(records,meta,clip)
        np.testing.assert_array_equal(result["boxes"][:,0],[0,0,2,4])
        np.testing.assert_array_equal(result["statuses"],["initialized","initialized","tracking","uncertain"])

if __name__=="__main__":unittest.main()
