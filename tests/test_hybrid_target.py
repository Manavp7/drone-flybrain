"""Target geometry/clock contracts; fixture detections are not model evidence."""
import unittest

import numpy as np

from experiments.hybrid_target import TargetBridge, salience_mask


def person(tid=1, confidence=.9, box=(100., 80., 180., 280.)):
    return {"track_id": tid, "class_id": 0, "label": "person", "confidence": confidence,
            "bbox_xyxy": list(box), "surface_measurement": None}


def result(sequence=0, timestamp=0., detections=None, **kwargs):
    r = {"sequence": sequence, "capture_time_s": timestamp, "stream_id": "scene",
         "clock_domain": "sim", "frame_id": "optical", "status": "ok",
         "inference_executed": True, "detections": [person()] if detections is None else detections}
    r.update(kwargs)
    return r


class TargetBridgeTests(unittest.TestCase):
    def test_selects_highest_confidence_once_then_preserves_id(self):
        bridge = TargetBridge()
        first = bridge.update(result(detections=[person(3,.8),person(2,.9)]), 0., (391,391))
        self.assertEqual(first["track_id"], 2)
        second = bridge.update(result(1,.1,[person(3,.99),person(2,.51)]), .1, (391,391))
        self.assertTrue(second["valid"])
        self.assertEqual(second["track_id"], 2)
        self.assertEqual(second["confidence"], .51)

    def test_tied_selection_is_deterministic(self):
        bridge = TargetBridge()
        self.assertEqual(bridge.update(result(detections=[person(3),person(2)]),0.,(391,391))["track_id"],2)

    def test_explicit_selection_waits_for_only_that_id(self):
        bridge = TargetBridge(296)
        self.assertFalse(bridge.update(result(),0.,(391,391))["valid"])
        self.assertTrue(bridge.update(result(1,.1,[person(296)]),.1,(391,391))["valid"])

    def test_missing_target_does_not_switch_and_same_id_can_recover(self):
        bridge = TargetBridge()
        bridge.update(result(),0.,(391,391))
        missing = bridge.update(result(1,.1,[person(2)]),.1,(391,391))
        self.assertFalse(missing["valid"])
        self.assertEqual(missing["track_id"],1)
        self.assertIsNone(missing["bbox_xyxy"])
        self.assertTrue(bridge.update(result(2,.2),.2,(391,391))["valid"])

    def test_stream_reset_latches_invalid_even_if_id_matches(self):
        bridge = TargetBridge()
        bridge.update(result(),0.,(391,391))
        changed = bridge.update(result(1,.1,stream_id="other"),.1,(391,391))
        self.assertEqual(changed["reason"],"stream_changed")
        self.assertFalse(bridge.update(result(2,.2),.2,(391,391))["valid"])

    def test_stale_future_reordered_and_unknown_timing_are_invalid(self):
        bridge = TargetBridge()
        self.assertEqual(bridge.update(result(),.251,(391,391))["reason"],"stale_target_frame")
        self.assertEqual(bridge.update(result(timestamp=1.),0.,(391,391))["reason"],"future_capture_time")
        bridge.update(result(),0.,(391,391))
        self.assertEqual(bridge.update(result(),0.,(391,391))["reason"],"duplicate_or_reordered_frame")
        self.assertFalse(bridge.update(result(1,.1,status="timing_unverified"),.1,(391,391))["valid"])

    def test_geometry_is_normalized_without_metric_depth(self):
        observation = TargetBridge().update(result(detections=[person(box=(100,100,300,300))]),0.,(400,400))
        self.assertEqual(observation["center_normalized"],[0.,0.])
        self.assertEqual(observation["height_fraction"],.5)
        self.assertEqual(observation["area_fraction"],.25)
        self.assertIsNone(observation["depth_m"])
        self.assertFalse(observation["control_authority"])

    def test_invalid_duplicate_or_off_image_boxes_do_not_select(self):
        for detections in ([person(1),person(1)], [person(box=(-1,0,10,10))], [person(confidence=float("nan"))]):
            bridge = TargetBridge()
            self.assertFalse(bridge.update(result(detections=detections),0.,(391,391))["valid"])
            self.assertIsNone(bridge.track_id)

    def test_wrong_class_never_becomes_person(self):
        bird = person()
        bird.update(class_id=14,label="bird")
        self.assertFalse(TargetBridge().update(result(detections=[bird]),0.,(391,391))["valid"])

    def test_rejected_inference_cannot_supply_a_target(self):
        for changes in ({"status":"rejected"}, {"inference_executed":False}):
            self.assertFalse(TargetBridge().update(result(**changes),0.,(391,391))["valid"])

    def test_bad_configuration_rejected(self):
        for tid in (True,0,-1,1.1):
            with self.assertRaises(ValueError): TargetBridge(tid)
        with self.assertRaises(ValueError): TargetBridge().update(result(),0.,(0,391))
        with self.assertRaises(ValueError): TargetBridge().update(result(),float("nan"),(391,391))


class SalienceTests(unittest.TestCase):
    def test_only_valid_bbox_is_dark(self):
        observation = TargetBridge().update(result(),0.,(391,391))
        mask = salience_mask(observation,(391,391))
        self.assertEqual(mask.shape,(391,391))
        self.assertEqual(mask.dtype,np.float32)
        self.assertTrue(np.all(mask[80:280,100:180] == np.float32(.1)))
        self.assertEqual(np.count_nonzero(mask < .5),80*200)
        self.assertTrue(np.all(salience_mask({"valid":False},(391,391)) == .5))

    def test_nonsquare_mapping_preserves_aspect_ratio(self):
        mask = salience_mask({"valid":True,"bbox_xyxy":[0.,0.,200.,100.]},(100,200),400)
        ys,xs = np.where(mask<.5)
        self.assertEqual((xs.min(),xs.max(),ys.min(),ys.max()),(0,399,100,299))

    def test_valid_but_malformed_geometry_is_not_silently_rendered(self):
        with self.assertRaises(ValueError):
            salience_mask({"valid":True,"bbox_xyxy":[0.,0.,500.,100.]},(391,391))


if __name__ == "__main__":
    unittest.main()
