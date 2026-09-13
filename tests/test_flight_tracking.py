import unittest
import numpy as np
from experiments.flight_contracts import CameraFrame,R_BODY_CAMERA,rotation_from_euler
from experiments.flight_tracking import EgoMotionTracker
from perception.detector import Detection


def frame(t=0.,yaw=0.,position=(0,0,0),color=(220,20,20),depth=5.):
    rgb=np.empty((100,100,3),np.uint8);rgb[:]=color
    return CameraFrame(rgb,np.full((100,100),depth,np.float32),(100.,100.,50.,50.),
        rotation_from_euler(yaw=yaw)@R_BODY_CAMERA,np.array(position,float),t)


def detection(box=(40.,20.,60.,80.)):
    return Detection(tuple(box),.9,0,'person')


def projected_box(f):
    # Independent fixed world rectangle, five metres ahead of initial camera.
    points=np.array([[5.,y,z] for y in (-.5,.5) for z in (-1.5,1.5)])
    optical=(points-f.position_world_camera)@f.rotation_world_camera
    uv=optical[:,:2]/optical[:,2:]*100+50
    return np.r_[uv.min(0),uv.max(0)]


class FlightTrackingTests(unittest.TestCase):
    def setUp(self):self.tracker=EgoMotionTracker(max_age_s=1.2,appearance_threshold=.65)
    def update(self,f,d=None):
        self.tracker.prepare(f)
        return self.tracker.update([d or detection()],f.capture_time_s,f.rgb)

    def test_rotating_camera_keeps_actual_target_id(self):
        self.assertEqual(self.update(frame())[0]['track_id'],1)
        f=frame(.3,yaw=.23)
        self.assertEqual(self.update(f,detection(projected_box(f)))[0]['track_id'],1)
        self.assertTrue(self.tracker.last_reprojections[0]['valid'])

    def test_translation_reprojects_with_measured_depth(self):
        self.update(frame())
        f=frame(.3,position=(1,0,0),depth=4.)
        self.assertEqual(self.update(f,detection(projected_box(f)))[0]['track_id'],1)

    def test_reprojection_does_not_renew_missing_observation(self):
        self.update(frame())
        f=frame(.4,yaw=.1);self.tracker.prepare(f)
        self.assertEqual(self.tracker.update([],f.capture_time_s,f.rgb),[])
        self.assertEqual(self.tracker.tracks[1]['time'],0.)
        self.assertEqual(self.tracker.anchors[1]['capture_time_s'],0.)

    def test_matching_different_clothing_is_still_rejected(self):
        self.update(frame())
        f=frame(.2,color=(20,20,220))
        self.assertEqual(self.update(f)[0]['track_id'],2)

    def test_new_missing_depth_does_not_reuse_old_world_anchor(self):
        self.update(frame())
        self.update(frame(.2,depth=np.nan))
        self.assertNotIn(1,self.tracker.anchors)

    def test_expiry_never_reselects_old_identity(self):
        self.update(frame())
        self.assertEqual(self.update(frame(1.3))[0]['track_id'],2)
        self.assertNotIn(1,self.tracker.anchors)

    def test_reset_clears_anchors_and_does_not_reuse_ids(self):
        self.update(frame());self.tracker.reset()
        self.assertFalse(self.tracker.anchors)
        self.assertEqual(self.update(frame(.2))[0]['track_id'],2)

    def test_exact_capture_pose_is_required(self):
        self.tracker.prepare(frame(.2))
        with self.assertRaises(ValueError):self.tracker.update([detection()],.3,frame(.3).rgb)

    def test_small_independent_target_motion_can_still_associate(self):
        self.update(frame())
        f=frame(.3,yaw=.15)
        box=projected_box(f)+np.array([2.,0.,2.,0.])
        self.assertEqual(self.update(f,detection(box))[0]['track_id'],1)


if __name__=='__main__':unittest.main()
