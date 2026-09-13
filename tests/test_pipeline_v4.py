"""Perception contract tests use explicit fixtures, never pretend neural accuracy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from perception.detector import Detection
from perception.pipeline import CameraSample, PerceptionPipeline, ShortTermTracker, forward_depth_advisory, surface_measurement
from perception.__main__ import load_sample


class Clock:
    def __init__(self): self.now=100.
    def __call__(self): return self.now


class FixtureDetector:
    def __init__(self, detections=(), clock=None, latency=0):
        self.detections=list(detections); self.clock=clock; self.latency=latency; self.calls=0
    def detect(self,image_rgb):
        self.calls+=1
        if self.clock: self.clock.now+=self.latency
        return self.detections


def sample(**changes):
    values=dict(image_rgb=np.zeros((20,30,3),np.uint8),capture_time_s=1.,received_monotonic_s=100.,sequence=1,
                stream_id='a',clock_domain='sim',frame_id='optical',capture_age_at_receive_s=0.,
                depth_m=np.full((20,30),2.,np.float32),depth_time_s=1.,
                camera_intrinsics=(20.,20.,15.,10.),registration_verified=True)
    values.update(changes)
    return CameraSample(**values)


def detection(box=(5.,5.,15.,15.),cls=0):
    return Detection(box,.8,cls,'person' if cls==0 else 'bicycle')


class PipelineTests(unittest.TestCase):
    def setUp(self): self.clock=Clock()
    def pipeline(self,detector=None): return PerceptionPipeline(detector or FixtureDetector(),clock=self.clock)
    def test_complete_registered_sample(self):
        p=self.pipeline(FixtureDetector([detection()]))
        r=p.process(sample())
        self.assertEqual(r['status'],'ok'); self.assertEqual(r['depth_advisory']['state'],'near_surface')
        self.assertEqual(r['detections'][0]['surface_measurement']['surface_optical_z_m'],2.)
        self.assertFalse(r['control_authority']); json.dumps(r,allow_nan=False)
    def test_empty_detections_do_not_imply_clearance(self):
        r=self.pipeline().process(sample(depth_m=None,depth_time_s=None,registration_verified=False))
        self.assertEqual(r['detections'],[]); self.assertEqual(r['depth_advisory']['state'],'unknown')
    def test_far_depth_never_claims_clear_flight_corridor(self):
        r=forward_depth_advisory(sample(depth_m=np.full((20,30),10.,np.float32)))
        self.assertEqual(r['state'],'no_near_surface_in_sampled_region'); self.assertIsNone(r['flight_corridor_clear'])
    def test_stale_capture_is_rejected_before_detector(self):
        d=FixtureDetector(); r=self.pipeline(d).process(sample(capture_age_at_receive_s=.31))
        self.assertEqual(r['reason'],'stale_camera_frame'); self.assertEqual(d.calls,0)
    def test_stale_local_packet_is_rejected(self):
        self.assertEqual(self.pipeline().process(sample(received_monotonic_s=99.))['status'],'rejected')
    def test_future_local_timestamp_is_rejected(self):
        self.assertEqual(self.pipeline().process(sample(received_monotonic_s=101.))['reason'],'future_receive_time')
    def test_slow_inference_cannot_renew_its_deadline(self):
        d=FixtureDetector([detection()],self.clock,.4)
        r=self.pipeline(d).process(sample())
        self.assertTrue(r['inference_executed']); self.assertEqual(r['reason'],'inference_deadline_missed')
        self.assertEqual(r['detections'],[])
    def test_depth_and_tracking_work_are_inside_final_deadline(self):
        def slow_depth(_):
            self.clock.now += .4
            return {'state':'near_surface'}
        p=self.pipeline(FixtureDetector([detection()]))
        with patch('perception.pipeline.forward_depth_advisory',side_effect=slow_depth):
            r=p.process(sample())
        self.assertEqual(r['reason'],'processing_deadline_missed')
        self.assertEqual(r['detections'],[]); self.assertEqual(r['depth_advisory']['state'],'unknown')
        self.assertGreater(r['processing_ms'],399)
        self.assertEqual(p.tracker.tracks,{})
    def test_unknown_source_capture_age_preserves_only_2d(self):
        r=self.pipeline(FixtureDetector([detection()])).process(sample(capture_age_at_receive_s=None))
        self.assertEqual(r['status'],'timing_unverified'); self.assertEqual(len(r['detections']),1)
        self.assertEqual(r['depth_advisory']['state'],'unknown')
    def test_duplicate_sequence_and_time_rejected(self):
        p=self.pipeline(); p.process(sample())
        self.assertEqual(p.process(sample())['reason'],'duplicate_or_reordered_frame')
    def test_source_clock_rewind_rejected_in_same_stream(self):
        p=self.pipeline(); p.process(sample())
        self.assertEqual(p.process(sample(sequence=2,capture_time_s=.5))['status'],'rejected')
    def test_new_stream_can_restart_clock_without_reusing_track_id(self):
        p=self.pipeline(FixtureDetector([detection()])); a=p.process(sample())
        b=p.process(sample(stream_id='b',capture_time_s=0.,depth_time_s=0.))
        self.assertEqual(b['status'],'ok'); self.assertNotEqual(a['detections'][0]['track_id'],b['detections'][0]['track_id'])
    def test_invalid_detector_result_fails_closed(self):
        r=self.pipeline(FixtureDetector([{'box':'invalid'}])).process(sample())
        self.assertEqual(r['status'],'detector_error')
    def test_outside_image_box_rejected(self):
        r=self.pipeline(FixtureDetector([detection((-1,0,2,3))])).process(sample())
        self.assertEqual(r['status'],'detector_error')
    def test_backend_exception_is_not_empty_success(self):
        class Broken:
            def detect(self,_): raise RuntimeError('weights missing')
        r=self.pipeline(Broken()).process(sample())
        self.assertEqual(r['status'],'detector_error'); self.assertFalse(r['inference_executed'])
    def test_invalid_channels_rejected(self):
        with self.assertRaises(ValueError): sample(image_rgb=np.zeros((20,30,4),np.uint8)).validate()
    def test_invalid_intrinsics_rejected(self):
        with self.assertRaises(ValueError): sample(camera_intrinsics=(-1,20,15,10)).validate()
    def test_nan_depth_is_unknown(self):
        r=forward_depth_advisory(sample(depth_m=np.full((20,30),np.nan,np.float32)))
        self.assertEqual(r['state'],'unknown')
    def test_partial_depth_coverage_is_unknown(self):
        depth=np.full((20,30),2.,np.float32); depth[:,8:25]=0
        self.assertEqual(forward_depth_advisory(sample(depth_m=depth))['state'],'unknown')
    def test_unsynchronized_depth_is_not_used(self):
        r=forward_depth_advisory(sample(depth_time_s=.5))
        self.assertEqual(r['reason'],'rgb_depth_not_synchronized')
    def test_shape_match_does_not_prove_depth_registration(self):
        self.assertEqual(forward_depth_advisory(sample(registration_verified=False))['reason'],'depth_registration_unverified')
    def test_optical_projection_uses_calibrated_pixel_coordinates(self):
        m=surface_measurement(sample(),(14.,9.,17.,12.))
        self.assertIsNotNone(m); self.assertAlmostEqual(m['surface_xyz_optical_m'][0],0.)
        self.assertAlmostEqual(m['surface_xyz_optical_m'][1],0.); self.assertEqual(m['surface_xyz_optical_m'][2],2.)
    def test_spool_load_preserves_metadata_and_depth(self):
        with tempfile.TemporaryDirectory() as d:
            s=sample(); path=Path(d)/'frame.npz'
            np.savez(path,**{k:v for k,v in vars(s).items() if v is not None})
            restored=load_sample(path)
            self.assertEqual(restored.sequence,1); self.assertEqual(restored.frame_id,'optical')
            np.testing.assert_array_equal(restored.image_rgb,s.image_rgb)
    def test_spool_rejects_object_arrays(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'frame.npz'; np.savez(path,image_rgb=np.array([object()],dtype=object))
            with self.assertRaises(ValueError): load_sample(path)


class TrackerTests(unittest.TestCase):
    def test_same_class_overlap_retains_id(self):
        t=ShortTermTracker(); a=t.update([detection()],1.)
        b=t.update([detection((6,5,16,15))],1.1)
        self.assertEqual(a[0]['track_id'],b[0]['track_id'])
    def test_different_class_does_not_steal_id(self):
        t=ShortTermTracker(); a=t.update([detection()],1.); b=t.update([detection(cls=1)],1.1)
        self.assertNotEqual(a[0]['track_id'],b[0]['track_id'])
    def test_expiry_creates_new_id(self):
        t=ShortTermTracker(); a=t.update([detection()],1.); t.update([],1.2)
        b=t.update([detection()],2.)
        self.assertNotEqual(a[0]['track_id'],b[0]['track_id'])
    def test_capacity_bounds_retained_tracks(self):
        t=ShortTermTracker(max_tracks=2)
        for i in range(10): t.update([detection((i*20,0,i*20+10,10))],1+i*.01)
        self.assertLessEqual(len(t.tracks),2)
    def test_one_to_one_association(self):
        t=ShortTermTracker(); t.update([detection()],1.)
        rows=t.update([detection(),detection((6,5,16,15))],1.1)
        self.assertEqual(len({r['track_id'] for r in rows}),2)
    def test_time_rewind_rejected(self):
        t=ShortTermTracker(); t.update([],1.)
        with self.assertRaises(ValueError): t.update([],.9)


if __name__=='__main__': unittest.main()
