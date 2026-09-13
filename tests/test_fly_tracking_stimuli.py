"""Checks for evaluation separation, geometry and causal output alignment."""
import unittest

import numpy as np

from experiments.fly_tracking_stimuli import (CALIBRATION, TEST_SEEDS, DT,
    synthetic_clip, source_to_retina, retina_to_source, velocity_at)
from experiments.flyvis_tracking import display_boxes, display_statuses
from flybrain_sim.research_model import validate_clip


class FlyTrackingStimuliTests(unittest.TestCase):
    def test_source_crop_inverse_and_known_origin(self):
        np.testing.assert_allclose(source_to_retina([420,0,1500,1080]), [0,0,391,391])
        boxes = np.array([[530,474,751,904], [1021,472,1200,919]])
        np.testing.assert_allclose(retina_to_source(source_to_retina(boxes)), boxes)

    def test_control_truth_stops_and_reverses_without_box_size_change(self):
        clip = synthetic_clip("reversal")
        validate_clip(clip["frames"], clip["times"], DT)
        boxes = clip["truth_boxes"]
        np.testing.assert_allclose(boxes[:,2:]-boxes[:,:2], np.full((len(boxes),2),96.))
        np.testing.assert_allclose(np.diff(boxes[:,:2],axis=0), clip["velocities"][:-1]*DT, atol=1e-12)
        self.assertTrue(np.all(clip["velocities"][:40] == 0))
        self.assertGreater(clip["velocities"][60,0],0)
        self.assertLess(clip["velocities"][140,0],0)

    def test_calibration_and_held_out_texture_sets_are_distinct(self):
        self.assertTrue(set(CALIBRATION).isdisjoint(TEST_SEEDS))
        self.assertNotIn(101, TEST_SEEDS.values())
        np.testing.assert_array_equal(velocity_at([.1,2.,2.6], "stop_restart"), [[0,0],[0,0],[-18,24]])

    def test_display_uses_only_available_responses_and_initializes_once(self):
        initial = [0,0,10,10]
        predictions = np.array([[1,0,11,10], [2,0,12,10]])
        result = display_boxes(predictions, [.02,.04], [0.,.019,.02,.039,.04], initial)
        np.testing.assert_array_equal(result, [initial,initial,predictions[0],predictions[0],predictions[1]])

    def test_loss_label_appears_only_when_lost_response_is_available(self):
        result = display_statuses(["tracking", "outside_frame"], [.02,.04], [0.,.02,.039,.04,.05])
        np.testing.assert_array_equal(result, ["initialized","tracking","tracking","outside_frame","outside_frame"])


if __name__ == "__main__":
    unittest.main()
