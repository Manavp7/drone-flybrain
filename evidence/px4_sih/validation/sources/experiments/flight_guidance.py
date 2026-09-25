"""Frozen Flyvis bearing -> bounded yaw/forward requests for simulation.

YOLO geometry may reject unsupported input, but never supplies steering bearing.
Metric target standoff uses registered visible-surface depth. Altitude and
attitude stabilization are conventional simulated autopilot responsibilities.
"""
from __future__ import annotations
import math
import numpy as np
from experiments.flight_contracts import (CameraFrame, FlightState, MAX_SPEED_M_S,
    MAX_OBSERVATION_AGE_S, COMMAND_MAX_CAPTURE_AGE_S, PHYSICS_DT, wrap_angle)

STANDOFF_M = 3.50
YAW_CENTERED_RAD = .12
TRAIN_X = .4
TRAIN_Y = .3
TRAIN_HEIGHT = (.22, .52)
TRACK_RETENTION_S = 1.2


def physical_completion(capture_s, measured_s, injected_delay_s=0.):
    values = [capture_s, measured_s, injected_delay_s]
    if not np.isfinite(values).all() or min(values) < 0:
        raise ValueError('Finite nonnegative physical timing required')
    delay = max(.1, measured_s+injected_delay_s)
    return math.ceil((capture_s+delay-1e-10)/PHYSICS_DT)*PHYSICS_DT


def cue_geometry(observation, image_hw):
    """Reject-only input geometry; these coordinates must not steer the drone."""
    if observation.get('valid') is not True:
        return None
    box = np.asarray(observation.get('bbox_xyxy'),float)
    h,w = image_hw
    if (box.shape != (4,) or not np.isfinite(box).all()
            or not 0 <= box[0] < box[2] <= w or not 0 <= box[1] < box[3] <= h):
        return None
    scale = 391/max(h,w)
    offset = np.array([(391-w*scale)/2,(391-h*scale)/2])
    center = ((box[:2]+box[2:])/2*scale+offset)*2/391-1
    return np.r_[center,(box[3]-box[1])*scale/391]


def estimate(readout, features, neural_valid, observation, frame, surface):
    """Make a capture-time estimate without vehicle/target/world truth."""
    frame.validate()
    base = dict(valid=False, reason='target_lost', heading_world_rad=None,
        surface_optical_z_m=None, decoded=None, cue_input=None,
        capture_time_s=float(frame.capture_time_s), track_id=observation.get('track_id'))
    cue = cue_geometry(observation,frame.rgb.shape[:2])
    if cue is None:
        return base
    base['cue_input'] = cue.tolist()
    if abs(cue[0]) > TRAIN_X+1e-9 or abs(cue[1]) > TRAIN_Y+1e-9 or not TRAIN_HEIGHT[0] <= cue[2] <= TRAIN_HEIGHT[1]:
        return dict(base,reason='outside_neural_training_envelope')
    f = np.asarray(features,float)
    if (not neural_valid or f.shape != (8,) or not np.isfinite(f).all()
            or f[7] < .05 or f[5] < np.log1p(.05)):
        return dict(base,reason='neural_evidence_unavailable')
    decoded = readout.predict(f)
    if (decoded.shape != (3,) or not np.isfinite(decoded).all()
            or abs(decoded[0]) > .5 or abs(decoded[1]) > .4 or not .15 <= decoded[2] <= .60):
        return dict(base,reason='neural_readout_out_of_range')
    if not frame.registration_verified or not isinstance(surface,dict):
        return dict(base,reason='target_depth_unavailable')
    z = surface.get('surface_optical_z_m')
    fraction = surface.get('valid_depth_fraction',0.)
    spread = surface.get('depth_spread_p90_p10_m',float('inf'))
    if (isinstance(z,bool) or not isinstance(z,(float,int)) or not np.isfinite([z,fraction,spread]).all()
            or not .3 <= z <= 20 or fraction < .8 or spread > .35):
        return dict(base,reason='target_depth_unsupported')
    h,w = frame.rgb.shape[:2]; scale = 391/max(h,w)
    offset = np.array([(391-w*scale)/2,(391-h*scale)/2])
    uv = (391*(decoded[:2]+1)/2-offset)/scale
    fx,fy,cx,cy = frame.intrinsics
    ray = frame.rotation_world_camera @ np.array([(uv[0]-cx)/fx,(uv[1]-cy)/fy,1.])
    if np.linalg.norm(ray[:2]) < .2 or not np.isfinite(ray).all():
        return dict(base,reason='invalid_level_bearing')
    return dict(base,valid=True,reason='neural_bearing_and_registered_depth',
        heading_world_rad=float(np.arctan2(ray[1],ray[0])),surface_optical_z_m=float(z),decoded=decoded.tolist())


def release(estimate_record, state, completed_s, sequence):
    """Release only at physical completion; expiry stays anchored to capture."""
    capture = estimate_record['capture_time_s']
    if not np.isfinite([completed_s,capture,state.time_s]).all() or completed_s < capture or abs(state.time_s-completed_s) > 1e-7:
        raise ValueError('Release state must be at physical completion, after capture')
    result = dict(sequence=int(sequence),capture_time_s=capture,issued_at_s=float(completed_s),
        valid_until_s=float(capture+COMMAND_MAX_CAPTURE_AGE_S),forward_speed=0.,yaw_target=state.yaw,
        reason=estimate_record['reason'],valid=False,estimate=estimate_record)
    if completed_s-capture > MAX_OBSERVATION_AGE_S+1e-9:
        return dict(result,reason='stale_perception_result')
    if not estimate_record['valid']:
        return result
    yaw = estimate_record['heading_world_rad']
    centered = abs(wrap_angle(yaw-state.yaw)) <= YAW_CENTERED_RAD
    speed = min(MAX_SPEED_M_S,max(0.,.6*(estimate_record['surface_optical_z_m']-STANDOFF_M))) if centered else 0.
    return dict(result,valid=True,forward_speed=float(speed),yaw_target=float(yaw))


def active_request(command, state):
    if (command is None or not command.get('valid') or state.time_s < command['issued_at_s']-1e-9
            or state.time_s >= command['valid_until_s']-1e-9):
        return dict(forward_speed=0.,yaw_target=state.yaw,sequence=None,reason='no_fresh_guidance')
    centered = abs(wrap_angle(command['yaw_target']-state.yaw)) <= YAW_CENTERED_RAD
    return dict(forward_speed=command['forward_speed'] if centered else 0.,yaw_target=command['yaw_target'],
                sequence=command['sequence'],reason=command['reason'])
