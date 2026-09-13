"""Perspective and image contracts for the explicit billboard simulation."""
import unittest

import numpy as np

from experiments.hybrid_scene import render_scene


class HybridSceneTests(unittest.TestCase):
    def setUp(self):
        self.patch = np.zeros((200,80,3),np.uint8)
        self.patch[:] = (45,85,130)
        self.patch[30:100,20:60] = (200,40,60)

    def render(self, vehicle=(0.,0.,0.),target=(4.,0.,0.),**kwargs):
        return render_scene(self.patch,vehicle,target,**kwargs)

    def test_projected_height_matches_pinhole_and_width_preserves_aspect(self):
        image,truth = self.render()
        x0,y0,x1,y1=truth["bbox_xyxy"]
        self.assertEqual(image.shape,(391,391,3))
        self.assertEqual(image.dtype,np.uint8)
        self.assertTrue(truth["visible"])
        self.assertAlmostEqual(y1-y0,360*1.9/4)
        self.assertAlmostEqual((x1-x0)/(y1-y0),.4)
        self.assertAlmostEqual((x0+x1)/2,195.5)

    def test_camera_translation_changes_next_image_and_correct_projection(self):
        first,truth0=self.render()
        right,truth1=self.render(vehicle=(0.,.5,0.))
        self.assertFalse(np.array_equal(first,right))
        self.assertAlmostEqual(truth1["bbox_xyxy"][0]-truth0["bbox_xyxy"][0],-45.)
        _,up=self.render(vehicle=(0.,0.,.5))
        self.assertAlmostEqual(up["bbox_xyxy"][1]-truth0["bbox_xyxy"][1],45.)

    def test_approach_enlarges_target(self):
        _,far=self.render()
        _,near=self.render(vehicle=(1.,0.,0.))
        self.assertGreater(near["bbox_xyxy"][3]-near["bbox_xyxy"][1],far["bbox_xyxy"][3]-far["bbox_xyxy"][1])
        self.assertEqual(near["relative_depth_m"],3.)

    def test_hidden_and_behind_camera_emit_background_only(self):
        hidden,truth=self.render(hidden=True)
        behind,backtruth=self.render(target=(-1.,0.,0.))
        self.assertFalse(truth["visible"])
        self.assertIsNone(truth["bbox_xyxy"])
        np.testing.assert_array_equal(hidden,behind)
        self.assertFalse(backtruth["visible"])

    def test_outside_image_is_not_reported_visible(self):
        _,truth=self.render(target=(4.,100.,0.))
        self.assertFalse(truth["visible"])
        self.assertIsNone(truth["bbox_xyxy"])

    def test_near_plane_is_bounded_and_clipped(self):
        image,truth=self.render(target=(.051,0.,0.))
        self.assertEqual(image.shape,(391,391,3))
        self.assertEqual(truth["bbox_xyxy"],[0.,0.,391.,391.])
        self.assertTrue(truth["visible"])

    def test_truth_is_separate_from_rgb(self):
        image,truth=self.render()
        self.assertNotIn("detections",truth)
        self.assertIsInstance(image,np.ndarray)
        self.assertEqual(truth["world_axes"],"+x camera forward, +y image right, +z image up")

    def test_nonfinite_and_invalid_inputs_are_rejected(self):
        for changes in ({"target_position":(float("nan"),0,0)}, {"focal_px":0}, {"physical_patch_height_m":-1}, {"hidden":1}):
            values=dict(person_patch=self.patch,vehicle_position=(0,0,0),target_position=(4,0,0))
            values.update(changes)
            with self.assertRaises(ValueError): render_scene(**values)


if __name__ == "__main__":
    unittest.main()
