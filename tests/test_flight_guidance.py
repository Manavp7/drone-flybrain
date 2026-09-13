import unittest
import numpy as np
from experiments.flight_contracts import CameraFrame, FlightState, R_BODY_CAMERA, rotation_from_euler
from experiments.flight_guidance import estimate, release, active_request, physical_completion


class FixedReadout:
    def __init__(self, value=(0.,0.,.35)):
        self.value=np.array(value)
    def predict(self, features):
        return self.value.copy()


def frame(yaw=0., rotation=None, t=1., h=391,w=391):
    r=rotation_from_euler(yaw=yaw) if rotation is None else rotation
    return CameraFrame(np.zeros((h,w,3),np.uint8),np.full((h,w),5.,np.float32),
        (280.,280.,w/2,h/2),r@R_BODY_CAMERA,np.array([0.,0.,1.1]),t)


def state(t=1.2,yaw=0.):
    return FlightState(t,np.array([0.,0.,1.1]),np.zeros(3),rotation_from_euler(yaw=yaw),np.zeros(3),np.ones(4)*2)


OBS=dict(valid=True,track_id=1,bbox_xyxy=[165.,125.,225.,262.])
SURFACE=dict(surface_optical_z_m=5.,valid_depth_fraction=1.,depth_spread_p90_p10_m=0.)
FEATURES=np.array([0.,0.,.1,.1,0.,2.,1.,1.])


class FlightGuidanceTests(unittest.TestCase):
    def test_image_right_turns_negative_world_yaw(self):
        r=estimate(FixedReadout((.2,0.,.35)),FEATURES,True,OBS,frame(),SURFACE)
        self.assertTrue(r['valid'])
        self.assertAlmostEqual(r['heading_world_rad'],-np.arctan(.2*391/2/280))

    def test_capture_attitude_rotates_bearing(self):
        r=estimate(FixedReadout(),FEATURES,True,OBS,frame(yaw=np.pi/2),SURFACE)
        self.assertAlmostEqual(r['heading_world_rad'],np.pi/2)

    def test_pitch_roll_compensation_uses_optical_ray(self):
        rotation=rotation_from_euler(.12,-.13,.4)
        value=(.2,.1,.35)
        r=estimate(FixedReadout(value),FEATURES,True,OBS,frame(rotation=rotation),SURFACE)
        ray=rotation@R_BODY_CAMERA@np.array([.2*391/2/280,.1*391/2/280,1.])
        self.assertAlmostEqual(r['heading_world_rad'],np.arctan2(ray[1],ray[0]))

    def test_detector_geometry_cannot_steer_valid_neural_bearing(self):
        a=estimate(FixedReadout((.15,0.,.35)),FEATURES,True,OBS,frame(),SURFACE)
        b=estimate(FixedReadout((.15,0.,.35)),FEATURES,True,dict(OBS,bbox_xyxy=[200.,125.,260.,262.]),frame(),SURFACE)
        self.assertEqual(a['heading_world_rad'],b['heading_world_rad'])

    def test_outside_training_input_rejects_plausible_decoded_center(self):
        r=estimate(FixedReadout(),FEATURES,True,dict(OBS,bbox_xyxy=[10.,125.,50.,262.]),frame(),SURFACE)
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'outside_neural_training_envelope')

    def test_zero_features_cannot_generate_bearing(self):
        r=estimate(FixedReadout(),np.zeros(8),True,OBS,frame(),SURFACE)
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'neural_evidence_unavailable')

    def test_missing_or_spread_depth_brakes(self):
        for surface in [None,dict(SURFACE,depth_spread_p90_p10_m=2.),dict(SURFACE,valid_depth_fraction=.3)]:
            r=estimate(FixedReadout(),FEATURES,True,OBS,frame(),surface)
            self.assertFalse(r['valid'])

    def test_loss_cannot_produce_guidance(self):
        r=estimate(FixedReadout(),FEATURES,True,dict(valid=False,track_id=1),frame(),SURFACE)
        self.assertFalse(r['valid']);self.assertEqual(r['reason'],'target_lost')

    def test_stale_completion_rejects_and_does_not_renew_capture_age(self):
        r=estimate(FixedReadout(),FEATURES,True,OBS,frame(),SURFACE)
        c=release(r,state(t=1.7),1.7,0)
        self.assertFalse(c['valid']);self.assertEqual(c['reason'],'stale_perception_result')
        self.assertAlmostEqual(c['valid_until_s'],1.9)

    def test_commands_cannot_act_before_release_or_at_expiry(self):
        r=estimate(FixedReadout(),FEATURES,True,OBS,frame(),SURFACE)
        c=release(r,state(),1.2,0)
        self.assertGreater(active_request(c,state(1.3))['forward_speed'],0.)
        for t in [1.199,1.9,2.]:
            self.assertEqual(active_request(c,state(t))['forward_speed'],0.)

    def test_heading_acquisition_rotates_without_translation(self):
        r=estimate(FixedReadout((.2,0.,.35)),FEATURES,True,OBS,frame(),SURFACE)
        c=release(r,state(yaw=.3),1.2,0)
        self.assertTrue(c['valid']);self.assertEqual(c['forward_speed'],0.)
        self.assertLess(c['yaw_target'],0.)

    def test_close_target_never_requests_blind_reverse(self):
        r=estimate(FixedReadout(),FEATURES,True,OBS,frame(),dict(SURFACE,surface_optical_z_m=2.))
        c=release(r,state(),1.2,0)
        self.assertEqual(c['forward_speed'],0.)

    def test_heading_alignment_is_rechecked_after_release(self):
        r=estimate(FixedReadout(),FEATURES,True,OBS,frame(),SURFACE)
        c=release(r,state(),1.2,0)
        self.assertGreater(active_request(c,state(1.3))['forward_speed'],0.)
        self.assertEqual(active_request(c,state(1.3,yaw=.2))['forward_speed'],0.)

    def test_completion_includes_measured_delay_and_rounds_up(self):
        self.assertAlmostEqual(physical_completion(1.,.281),1.285)
        self.assertAlmostEqual(physical_completion(1.,.03),1.1)
        self.assertAlmostEqual(physical_completion(1.,.2,.81),2.01)
        for value in [-1.,float('nan')]:
            with self.assertRaises(ValueError):physical_completion(1.,value)

    def test_release_requires_current_completion_state(self):
        r=estimate(FixedReadout(),FEATURES,True,OBS,frame(),SURFACE)
        with self.assertRaises(ValueError):release(r,state(t=1.1),1.2,0)


if __name__=='__main__':unittest.main()
