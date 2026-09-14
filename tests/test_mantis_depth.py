"""Depth layer tests use sensor arrays, without actor geometry or model weights."""
from dataclasses import replace
import unittest

import numpy as np

from experiments.flight_contracts import CameraFrame
from experiments.mantis_depth import coherent_foreground_surface, diagnose_foreground_surface

BOX = (0., 0., 100., 100.)


def frame_with_torso():
    depth = np.full((100,100), 12., dtype=np.float32)
    depth[22:54,35:65] = 4.
    return CameraFrame(np.zeros((100,100,3),dtype=np.uint8),depth,(100.,100.,49.5,49.5),
                       np.eye(3),np.zeros(3),1.)


class ForegroundSurfaceTests(unittest.TestCase):
    def test_person_layer_excludes_far_background_without_filling_it(self):
        frame = frame_with_torso()
        original = frame.depth_m.copy()
        result = coherent_foreground_surface(frame,BOX)
        self.assertIsNotNone(result)
        self.assertEqual(result['surface_optical_z_m'],4.)
        self.assertEqual(result['depth_spread_p90_p10_m'],0.)
        self.assertEqual(result['valid_depth_fraction'],1.)
        self.assertAlmostEqual(result['foreground_support_fraction'],.48)
        self.assertEqual(result['foreground_pixel_count'],960)
        self.assertIn('not_identity',result['meaning'])
        np.testing.assert_array_equal(frame.depth_m,original)

    def test_single_near_outlier_cannot_replace_a_supported_surface(self):
        frame = frame_with_torso()
        frame.depth_m[30,40] = .4
        result = coherent_foreground_surface(frame,BOX)
        self.assertIsNotNone(result)
        self.assertEqual(result['surface_optical_z_m'],4.)
        self.assertEqual(result['foreground_pixel_count'],959)

    def test_meaningful_unsupported_near_surface_fails_closed(self):
        frame = frame_with_torso()
        frame.depth_m[24:34,35:51] = 1.
        result = diagnose_foreground_surface(frame,BOX)
        self.assertEqual(result['reason'],'unsupported_nearer_depth_layer')
        self.assertIsNone(result['surface'])

    def test_foreground_occluder_is_reported_as_visible_surface(self):
        frame = frame_with_torso()
        frame.depth_m[22:54,35:55] = 2.
        result = coherent_foreground_surface(frame,BOX)
        self.assertIsNotNone(result)
        self.assertEqual(result['surface_optical_z_m'],2.)
        self.assertAlmostEqual(result['foreground_support_fraction'],.32)
        self.assertIn('not_identity',result['meaning'])

    def test_similarly_supported_nearby_layers_are_ambiguous(self):
        frame = frame_with_torso()
        frame.depth_m[18:58,25:50] = 4.
        frame.depth_m[18:58,50:75] = 4.4
        result = diagnose_foreground_surface(frame,BOX)
        self.assertEqual(result['reason'],'ambiguous_nearby_depth_layers')
        self.assertIsNone(result['surface'])

    def test_disconnected_equal_depth_patches_are_not_one_coherent_torso(self):
        frame = frame_with_torso()
        frame.depth_m[:] = 12.
        frame.depth_m[18:58,25:35] = 4.
        frame.depth_m[18:58,65:75] = 4.
        result = diagnose_foreground_surface(frame,BOX)
        self.assertEqual(result['reason'],'incoherent_nearer_depth_layer')
        self.assertIsNone(result['surface'])

    def test_measured_spread_limit_is_not_raised_or_clamped(self):
        frame = frame_with_torso()
        frame.depth_m[18:58,25:75] = np.linspace(4.,5.,50)[None,:]
        result = diagnose_foreground_surface(frame,BOX)
        self.assertEqual(result['reason'],'foreground_depth_spread_unsupported')
        self.assertGreater(result['foreground_depth_spread_p90_p10_m'],.7)
        self.assertIsNone(result['surface'])

    def test_invalid_depth_fraction_remains_the_actual_roi_fraction(self):
        frame = frame_with_torso()
        frame.depth_m[18:22,25:75] = np.nan
        result = coherent_foreground_surface(frame,BOX)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result['valid_depth_fraction'],.9)
        self.assertAlmostEqual(result['foreground_support_fraction'],.48)

    def test_insufficient_valid_depth_is_rejected(self):
        for invalid in (0.,np.nan,np.inf,-2.,81.):
            with self.subTest(invalid=invalid):
                frame = frame_with_torso()
                frame.depth_m[18:28,25:75] = invalid
                result = diagnose_foreground_surface(frame,BOX)
                self.assertEqual(result['reason'],'insufficient_valid_depth')
                self.assertIsNone(result['surface'])

    def test_all_zero_or_nan_depth_returns_none(self):
        for invalid in (0.,np.nan):
            frame = frame_with_torso()
            frame.depth_m[:] = invalid
            self.assertIsNone(coherent_foreground_surface(frame,BOX))

    def test_unregistered_depth_returns_none(self):
        frame = replace(frame_with_torso(),registration_verified=False)
        self.assertIsNone(coherent_foreground_surface(frame,BOX))

    def test_stale_or_misaligned_depth_returns_none(self):
        frame = frame_with_torso()
        for kwargs in ({'now_s':1.101},{'depth_time_s':.899},{'depth_time_s':.95},
                       {'depth_time_s':1.01},{'now_s':.99},{'now_s':float('nan')},
                       {'depth_time_s':True}):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(coherent_foreground_surface(frame,BOX,**kwargs))
        self.assertIsNotNone(coherent_foreground_surface(frame,BOX,now_s=1.08,depth_time_s=.98))

    def test_bad_or_tiny_boxes_fail_closed(self):
        for box in ((0,0,1,1),(0,0,0,100),(0,0,-1,100),(0,0,np.nan,100),
                    (-200,-200,-100,-100),None,(),(0,0,100)):
            with self.subTest(box=box):
                self.assertIsNone(coherent_foreground_surface(frame_with_torso(),box))

    def test_invalid_frame_contract_fails_closed(self):
        frame = replace(frame_with_torso(),intrinsics=(0.,100.,49.5,49.5))
        self.assertIsNone(coherent_foreground_surface(frame,BOX))
        self.assertIsNone(coherent_foreground_surface(None,BOX))


if __name__ == '__main__':
    unittest.main()
