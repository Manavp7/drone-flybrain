"""Actual existing YOLO/neural pipeline with 3D foreground surface sampling."""
from __future__ import annotations

import hashlib
import numpy as np

from experiments.flight_contracts import ROOT
from experiments.flight_guidance import estimate
from experiments.flight_tracking import EgoMotionTracker
from experiments.flight_vision import FlightVision, MANIFEST
from experiments.hybrid_video import load_frozen_readout
from experiments.mantis_depth import coherent_foreground_surface
from experiments.mantis_pipeline import MantisPerceptionPipeline


# Association memory only: a delayed pipeline may resume with a fresh detection
# after the original 1.2-second tracker lifetime. This does not extend the
# .65-second result freshness or .90-second capture-anchored command lifetime.
MANTIS_TRACK_MEMORY_S = 3.0
MANTIS_READOUT_FOLDER = ROOT/'assets/mantis_calibration'
MANTIS_READOUT_SHA = 'ea57f5b38d467f632a9b14e7403b6e1f8288dd86f5da5e47d5356a29fb573663'
MANTIS_SELECTION_SHA = 'ea34209dd4b2a52bc7d1f3628abf59c2b0202f98b05bf594ef169ebd7ab9e7ab'


def load_mantis_readout():
    """Pin the independently validated synthetic-cue fit and neural model."""
    readout, frozen = load_frozen_readout(MANTIS_READOUT_FOLDER,
        expected_selection_sha=MANTIS_SELECTION_SHA, expected_readout_sha=MANTIS_READOUT_SHA)
    if hashlib.sha256(MANIFEST.read_bytes()).hexdigest() != frozen['model_manifest_sha256']:
        raise ValueError('Mantis readout neural-model manifest mismatch')
    return readout, frozen


class MantisVision(FlightVision):
    """Reuse recognition and recurrence; replace photo-only depth sampling.

    New anchors still derive solely from current camera boxes/pose/registered
    depth. Foreground sampling can measure an occluder; it never asserts target
    identity or uses actor geometry. The separate whole-image depth stop gate
    remains unchanged.
    """
    def __init__(self):
        readout, frozen = load_mantis_readout()
        super().__init__()
        self.readout, self.frozen = readout, frozen

    def reset(self):
        super().reset()
        pipeline = self.pipeline
        tracker = self.pipeline.tracker
        self.pipeline = MantisPerceptionPipeline(self.detector,
            max_frame_age_s=pipeline.max_frame_age_s, clock=pipeline.clock,
            appearance_tracking=pipeline.appearance_tracking)
        # Retain bounded descriptors/observed world anchors, not predictions
        # exposed as detections. Reassociation still requires a current YOLO
        # box passing unchanged class, IoU and clothing checks. Similar-looking
        # people remain ambiguous: this is temporary tracking, not identity.
        self.pipeline.tracker = EgoMotionTracker(
            minimum_iou=tracker.minimum_iou,
            max_age_s=MANTIS_TRACK_MEMORY_S,
            max_tracks=tracker.max_tracks,
            appearance_threshold=tracker.appearance_threshold)

    def process(self, frame, zero_neural=False):
        result = super().process(frame, zero_neural)
        selected_id = result['observation'].get('track_id')
        selected_surface = None
        fx, fy, cx, cy = frame.intrinsics
        for detection in result['detections'].get('detections', []):
            if detection['class_id'] != 0:
                continue
            box, track_id = detection['bbox_xyxy'], detection['track_id']
            surface = coherent_foreground_surface(frame, box)
            detection['original_central_box_surface'] = detection.get('surface_measurement')
            detection['surface_measurement'] = surface
            self.pipeline.tracker.anchors.pop(track_id, None)
            if surface is not None:
                z = surface['surface_optical_z_m']
                pixels = np.array([[u, v] for u in (box[0], box[2]) for v in (box[1], box[3])])
                optical = np.column_stack([(pixels[:, 0]-cx)*z/fx,
                                           (pixels[:, 1]-cy)*z/fy, np.full(4, z)])
                corners = optical @ frame.rotation_world_camera.T + frame.position_world_camera
                self.pipeline.tracker.anchors[track_id] = dict(corners=corners,
                                                               capture_time_s=float(frame.capture_time_s))
            if track_id == selected_id and result['observation']['valid']:
                selected_surface = surface
        result['surface'] = selected_surface
        result['candidate'] = estimate(self.readout, result['supplied_features'],
                                       result['neural']['valid'], result['observation'],
                                       frame, selected_surface)
        return result
