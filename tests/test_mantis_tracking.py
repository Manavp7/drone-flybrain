"""Mantis association memory tests; actual tracking, no learned inference."""
import unittest

import numpy as np

from experiments.flight_contracts import CameraFrame, FlightState, R_BODY_CAMERA
from experiments.flight_guidance import active_request, release, TRACK_RETENTION_S
from experiments.flight_vision import FlightVision
from experiments.mantis_vision import MantisVision, MANTIS_TRACK_MEMORY_S
from perception.detector import Detection


BOX = (160., 130., 230., 260.)
RED = (220, 20, 20)
BLUE = (20, 20, 220)


class Detector:
    def __init__(self):
        self.output = []

    def detect(self, image_rgb):
        return self.output


class Brain:
    def reset(self):
        pass

    def step(self, mask, hold_s):
        return dict(features=np.array([0., 0., .1, .2, 0., 4., .05, .8]), valid=True)


class Readout:
    def predict(self, features):
        return np.array([0., 0., .33])


def make_vision(kind=MantisVision):
    vision = object.__new__(kind)
    vision.brain, vision.detector, vision.readout = Brain(), Detector(), Readout()
    vision.reset()
    return vision


def sample(vision, time_s, *, box=BOX, color=RED, observed=True, camera_y=0.):
    rgb = np.full((391, 391, 3), 110, np.uint8)
    x0, y0, x1, y1 = map(int, box)
    rgb[y0:y1, x0:x1] = color
    vision.detector.output = [Detection(box, .9, 0, 'person')] if observed else []
    camera = CameraFrame(rgb, np.full((391, 391), 4., np.float32),
                         (280., 280., 195., 195.), R_BODY_CAMERA.copy(),
                         np.array([.25, camera_y, 1.1]), time_s)
    return vision.process(camera)


def state(time_s):
    return FlightState(time_s, np.array([0., 0., 1.1]), np.zeros(3),
                       np.eye(3), np.zeros(3), np.full(4, .8*9.81/4))


class MantisTrackingTests(unittest.TestCase):
    def test_fresh_person_reassociates_after_pipeline_pause_beyond_legacy_memory(self):
        old, revised = make_vision(FlightVision), make_vision()
        for vision in (old, revised):
            first = sample(vision, 0.)
            self.assertTrue(first['observation']['valid'])
            self.assertEqual(first['observation']['track_id'], 1)
        self.assertFalse(sample(old, 1.6)['observation']['valid'])
        recovered = sample(revised, 1.6)
        self.assertTrue(recovered['observation']['valid'])
        self.assertEqual(recovered['observation']['track_id'], 1)
        self.assertEqual(recovered['observation']['capture_time_s'], 1.6)
        self.assertEqual(recovered['detections']['detections'][0]['track_id'], 1)
        self.assertEqual(revised.pipeline.tracker.minimum_iou, old.pipeline.tracker.minimum_iou)
        self.assertEqual(revised.pipeline.tracker.appearance_threshold, old.pipeline.tracker.appearance_threshold)
        self.assertEqual(old.pipeline.tracker.max_age_s, TRACK_RETENTION_S)
        self.assertEqual(revised.pipeline.tracker.max_age_s, MANTIS_TRACK_MEMORY_S)

    def test_missing_frames_neither_renew_observation_age_nor_allow_stale_control(self):
        vision = make_vision()
        first = sample(vision, 0.)
        command = release(first['candidate'], state(.1), .1, first['sequence'])
        self.assertTrue(command['valid'])
        self.assertGreater(command['forward_speed'], 0.)
        self.assertAlmostEqual(command['valid_until_s'], .9)
        for t in (1.6, 2.5):
            missing = sample(vision, t, observed=False)
            self.assertFalse(missing['observation']['valid'])
            self.assertFalse(missing['candidate']['valid'])
            self.assertEqual(missing['detections']['detections'], [])
            self.assertEqual(vision.pipeline.tracker.tracks[1]['time'], 0.)
            self.assertEqual(vision.pipeline.tracker.anchors[1]['capture_time_s'], 0.)
            self.assertEqual(active_request(command, state(t))['forward_speed'], 0.)
        delayed = release(first['candidate'], state(1.6), 1.6, first['sequence'])
        self.assertFalse(delayed['valid'])
        self.assertEqual(delayed['reason'], 'stale_perception_result')
        self.assertEqual(delayed['forward_speed'], 0.)
        recovered = sample(vision, 2.7)
        self.assertTrue(recovered['observation']['valid'])
        self.assertEqual(recovered['observation']['track_id'], 1)
        fresh = release(recovered['candidate'], state(2.8), 2.8, recovered['sequence'])
        self.assertTrue(fresh['valid'])
        self.assertAlmostEqual(fresh['valid_until_s'], 3.6)

    def test_changed_clothing_cannot_reclaim_selected_temporary_id(self):
        vision = make_vision()
        sample(vision, 0.)
        changed = sample(vision, 1.6, color=BLUE)
        self.assertFalse(changed['observation']['valid'])
        self.assertFalse(changed['candidate']['valid'])
        self.assertNotEqual(changed['detections']['detections'][0]['track_id'], 1)
        self.assertEqual(vision.pipeline.tracker.tracks[1]['time'], 0.)

    def test_unmatched_geometry_cannot_reclaim_selected_temporary_id(self):
        vision = make_vision()
        sample(vision, 0.)
        changed = sample(vision, 1.6, box=(270., 130., 340., 260.))
        self.assertFalse(changed['observation']['valid'])
        self.assertFalse(changed['candidate']['valid'])
        self.assertNotEqual(changed['detections']['detections'][0]['track_id'], 1)
        self.assertEqual(vision.pipeline.tracker.tracks[1]['time'], 0.)

    def test_camera_motion_reassociation_uses_observed_depth_anchor(self):
        vision = make_vision()
        sample(vision, 0.)
        # Camera moves 0.8 m left: the measured point moves +56 pixels right.
        # The new box has low raw overlap and must use the old depth/pose anchor.
        recovered = sample(vision, 1.6, box=(216., 130., 286., 260.), camera_y=.8)
        self.assertTrue(recovered['observation']['valid'])
        self.assertEqual(recovered['observation']['track_id'], 1)
        self.assertTrue(vision.pipeline.tracker.last_reprojections[0]['valid'])
        self.assertEqual(vision.pipeline.tracker.last_reprojections[0]['anchor_capture_time_s'], 0.)

    def test_memory_expires_from_last_observation_despite_missing_updates(self):
        vision = make_vision()
        sample(vision, 0.)
        sample(vision, 2.9, observed=False)
        expired = sample(vision, MANTIS_TRACK_MEMORY_S+.01)
        self.assertFalse(expired['observation']['valid'])
        self.assertFalse(expired['candidate']['valid'])
        self.assertNotEqual(expired['detections']['detections'][0]['track_id'], 1)
        self.assertNotIn(1, vision.pipeline.tracker.tracks)
        self.assertNotIn(1, vision.pipeline.tracker.anchors)
        self.assertEqual(expired['observation']['track_id'], 1)

    def test_new_episode_discards_old_descriptors_and_anchors(self):
        vision = make_vision()
        sample(vision, 0.)
        vision.reset()
        self.assertFalse(vision.pipeline.tracker.tracks)
        self.assertFalse(vision.pipeline.tracker.anchors)
        first = sample(vision, 0., color=BLUE)
        self.assertTrue(first['observation']['valid'])
        self.assertEqual(first['sequence'], 0)


if __name__ == '__main__':
    unittest.main()
