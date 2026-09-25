"""Actual sequential detector and recurrent neural processing for flight tests."""
import time
from pathlib import Path
import numpy as np
from experiments.flight_contracts import ROOT, MAX_OBSERVATION_AGE_S
from experiments.flight_guidance import TRACK_RETENTION_S, estimate
from experiments.flight_tracking import EgoMotionTracker
from experiments.hybrid_flyvis import HybridFlyvis
from experiments.hybrid_target import TargetBridge, salience_mask
from experiments.hybrid_video import load_frozen_readout
from perception.detector import YOLOXDetector
from perception.pipeline import CameraSample, PerceptionPipeline

TINY_SHA = '427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7'
MODEL = ROOT/'models/yolox_tiny_official/yolox_tiny.onnx'
MANIFEST = ROOT/'models/flyvis_0000_000.manifest.json'


class FlightVision:
    def __init__(self):
        import cv2
        cv2.setNumThreads(2)
        self.detector=YOLOXDetector(MODEL,expected_sha256=TINY_SHA,confidence=.5,nms_iou=.45,input_size=416)
        self.brain=HybridFlyvis(MANIFEST)
        self.readout,self.frozen=load_frozen_readout()
        self.reset()

    def reset(self):
        self.brain.reset()
        self.pipeline=PerceptionPipeline(self.detector,max_frame_age_s=MAX_OBSERVATION_AGE_S,appearance_tracking=True)
        self.pipeline.tracker=EgoMotionTracker(max_age_s=TRACK_RETENTION_S,appearance_threshold=.65)
        self.bridge=TargetBridge()
        self.sequence=0

    def process(self,frame,zero_neural=False):
        self.pipeline.tracker.prepare(frame)
        sample=CameraSample(frame.rgb,frame.capture_time_s,time.monotonic(),self.sequence,
            'mujoco_flight','mujoco_physics','primary_optical',capture_age_at_receive_s=0.,
            depth_m=frame.depth_m,depth_time_s=frame.capture_time_s,
            camera_intrinsics=frame.intrinsics,registration_verified=frame.registration_verified)
        detections=self.pipeline.process(sample)
        observation=self.bridge.update(detections,frame.capture_time_s,frame.rgb.shape[:2],MAX_OBSERVATION_AGE_S)
        surface=None
        for detection in detections.get('detections',[]):
            if detection['track_id']==observation.get('track_id'):
                surface=detection.get('surface_measurement');break
        mask=salience_mask(observation,frame.rgb.shape[:2])
        neural=self.brain.step(mask,.1)
        supplied=np.zeros(8) if zero_neural else neural['features'].copy()
        candidate=estimate(self.readout,supplied,neural['valid'],observation,frame,surface)
        result=dict(sequence=self.sequence,observation=observation,detections=detections,
            ego_motion=self.pipeline.tracker.last_reprojections,
            surface=surface,mask=mask,neural=neural,supplied_features=supplied,candidate=candidate)
        self.sequence+=1
        return result
