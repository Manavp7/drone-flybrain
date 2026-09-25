"""Actual YOLO/Flyvis with motor-driven, measured-delay MuJoCo camera flight.

This is offline simulation with measured computation delay. It is not a PX4
transport, an aircraft interface, or a claim of concurrent real-time execution.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

from experiments.flight_contracts import (ROOT,PHYSICS_DT,DEPTH_PERIOD_S,ALTITUDE_M,
    MAX_OBSERVATION_AGE_S,COMMAND_MAX_CAPTURE_AGE_S,MAX_SPEED_M_S)
from experiments.flight_autopilot import Autopilot
from experiments.flight_world import QuadWorld,TARGET_HEIGHT_M
from experiments.flight_guidance import physical_completion,release,active_request
from experiments.flight_safety import DepthGuardian
from experiments.flight_vision import FlightVision,MODEL,MANIFEST,TINY_SHA

PATCH=ROOT/'results/hybrid_flight_run01/source_patch.png'
SPECS=[
    dict(name='approach_left',kind='normal',duration_s=10.,target=[5.3,.6,.95]),
    dict(name='approach_right',kind='normal',duration_s=10.,target=[5.2,-.8,.95]),
    dict(name='moving_target',kind='normal',duration_s=12.,target=[4.9,.2,.95],moving=True),
    *[dict(name=k,kind=k,duration_s=8.,target=[5.,0.,.95],event_s=3.5)
      for k in ('target_loss','obstacle_stop','depth_loss','stale_depth','delayed_inference')],
    dict(name='zero_neural_features',kind='ablation',duration_s=8.,target=[5.,.7,.95],ablation=True),
]
RUNTIME_SOURCES=['experiments/'+name for name in ('flight_contracts.py','flight_world.py','flight_autopilot.py',
    'flight_guidance.py','flight_safety.py','flight_tracking.py','flight_vision.py','flight_demo.py','hybrid_flyvis.py',
    'hybrid_target.py','hybrid_control.py','hybrid_video.py','runtime.py')]+[
    'perception/detector.py','perception/pipeline.py','flybrain_sim/research_model.py']


def serial(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value


def save_json(path,value):
    Path(path).write_text(json.dumps(serial(value),indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_rows(path,rows):
    with Path(path).open('x') as f:
        for row in rows:f.write(json.dumps(serial(row),allow_nan=False)+'\n')


def update_fixture(world,spec):
    """Evaluator/scenario-owned target and obstacle; no truth enters guidance."""
    t=world.state().time_s
    target=np.array(spec['target'],float)
    if spec.get('moving'):target[1]+=.3*np.sin(.45*t)
    hidden=spec['kind']=='target_loss' and t>=spec['event_s']-1e-8
    world.set_target(target,hidden=hidden)
    enabled=spec['kind']=='obstacle_stop' and t>=spec['event_s']-1e-8
    world.set_obstacle((1.85,0.,.7),enabled=enabled)


def evaluation_projection(world,frame):
    """Project known photo subject for scoring only; never passed to vision."""
    truth=world.truth()
    if truth['target_hidden']:return None,None
    # Same foreground box and original crop recorded by the prior photo source.
    crop=np.array([1113.,416.,1322.,959.]); person=np.array([1128.,431.,1307.,944.])
    uv=(person-np.r_[crop[:2],crop[:2]])/np.tile(crop[2:]-crop[:2],2)
    center=np.array(truth['target_position'])
    corners=np.array([center+[-.01,world.target_half_width*(1-2*u),TARGET_HEIGHT_M*(.5-v)]
                      for u in (uv[0],uv[2]) for v in (uv[1],uv[3])])
    optical=(corners-frame.position_world_camera)@frame.rotation_world_camera
    if np.min(optical[:,2])<=0:return None,None
    fx,fy,cx,cy=frame.intrinsics
    pixels=optical[:,:2]/optical[:,2:]*[fx,fy]+[cx,cy]
    bbox=np.r_[pixels.min(0),pixels.max(0)]
    h,w=frame.rgb.shape[:2]
    bbox=np.clip(bbox,[0,0,0,0],[w,h,w,h])
    return bbox.tolist(),float(np.mean(optical[:,2]))


def run_episode(spec,folder,vision):
    folder.mkdir()
    save_json(folder/'spec.json',spec)
    world=QuadWorld(PATCH);pilot=Autopilot();guardian=DepthGuardian()
    ticks=[];observations=[];depth_rows=[];depth_arrays=[];observation_arrays=[]
    current_command=None;latest_depth=None;initial=None
    started=time.perf_counter();tick=0
    duration=spec['duration_s'];total_ticks=round(duration/PHYSICS_DT)
    if abs(total_ticks*PHYSICS_DT-duration)>1e-9:raise ValueError('Duration must align to physics ticks')
    vision.reset()
    world.reset();update_fixture(world,spec)
    (folder/'world.xml').write_text(world.mjcf)
    np.savez_compressed(folder/'neural_binding.npz',baseline=vision.brain.baseline_activity,
                        cell_indices=vision.brain.cell_indices,centers_rc=vision.brain.centers_rc)
    initial=world.overview()
    np.savez_compressed(folder/'initial.npz',overview=initial)
    frame_directory=folder/'observations';frame_directory.mkdir()
    depth_directory=folder/'depth';depth_directory.mkdir()

    def advance(end_s):
        nonlocal tick,latest_depth
        goal=min(total_ticks,round(end_s/PHYSICS_DT))
        while tick<goal:
            update_fixture(world,spec)
            state=world.state()
            if tick%round(DEPTH_PERIOD_S/PHYSICS_DT)==0:
                fault=state.time_s>=spec.get('event_s',float('inf'))-1e-8
                sensor=None
                if fault and spec['kind']=='depth_loss':
                    latest_depth=None
                elif not (fault and spec['kind']=='stale_depth'):
                    latest_depth=world.capture(safety=True);sensor=latest_depth
                index=len(depth_rows)
                depth_rows.append(dict(sequence=index,scheduled_time_s=state.time_s,
                    sample_capture_time_s=None if latest_depth is None else latest_depth.capture_time_s,
                    fresh_capture=sensor is not None,fault_applied=fault and spec['kind'] in ('depth_loss','stale_depth'),
                    camera_intrinsics=None if sensor is None else sensor.intrinsics,
                    camera_rotation=None if sensor is None else sensor.rotation_world_camera,
                    camera_position=None if sensor is None else sensor.position_world_camera,
                    file=None if sensor is None else 'depth/%06d.npz'%index))
                if sensor is not None:depth_arrays.append((index,sensor))
            request=active_request(current_command,state)
            safety=guardian.check(latest_depth,state,request['forward_speed'],state.time_s)
            motors=pilot.command(state,safety['forward_speed'],request['yaw_target'],ALTITUDE_M)
            after=world.step(motors)
            ticks.append(dict(index=tick,time_s=state.time_s,state_before=asdict(state),state_after=asdict(after),
                request=request,guardian=safety,motor_targets=motors,
                applied_command_sequence=request['sequence'],truth_after=world.truth()))
            tick+=1

    failure=None
    try:
        while tick<total_ticks:
            update_fixture(world,spec)
            t0=time.perf_counter();frame=world.capture()
            bundle=vision.process(frame,zero_neural=spec.get('ablation',False))
            elapsed=time.perf_counter()-t0
            # Synthetic delay is an explicit failure injection, never measured speed.
            injected=1.0 if spec['kind']=='delayed_inference' and frame.capture_time_s>=spec['event_s']-1e-8 else 0.
            completion=physical_completion(frame.capture_time_s,elapsed,injected)
            truth_bbox,truth_depth=evaluation_projection(world,frame)
            before_tick=tick
            advance(completion)
            discarded=completion>duration+1e-8
            command=None if discarded else release(bundle['candidate'],world.state(),completion,bundle['sequence'])
            if command is not None:current_command=command
            neural=bundle['neural']
            row=dict(sequence=bundle['sequence'],capture_time_s=frame.capture_time_s,completed_time_s=completion,
                inference_wall_s=elapsed,injected_delay_s=injected,delay_physics_tick_start=before_tick,
                delay_physics_tick_end=tick,discarded_after_episode=discarded,
                observation=bundle['observation'],detections=bundle['detections'],surface=bundle['surface'],
                ego_motion=bundle['ego_motion'],
                estimate=bundle['candidate'],candidate=command,target_truth_bbox=truth_bbox,
                target_surface_optical_z_m=truth_depth,camera_intrinsics=frame.intrinsics,
                camera_rotation=frame.rotation_world_camera,camera_position=frame.position_world_camera,
                neural_valid=neural['valid'],neural_stimulus_time_s=neural['stimulus_time_s'],
                neural_response_time_s=neural['response_time_s'],neural_elapsed_wall_s=neural['elapsed_wall_s'],
                file='observations/%06d.npz'%bundle['sequence'])
            observations.append(row)
            observation_arrays.append((bundle['sequence'],dict(rgb=frame.rgb,depth_m=frame.depth_m,mask=bundle['mask'],
                activity=neural['activity'],retina=neural['retina'],features=neural['features'],
                supplied_features=bundle['supplied_features'],overview=world.overview())))
            if len(observations)%20==0:
                print(json.dumps(dict(case=spec['name'],sim_s=round(world.state().time_s,3),
                    captures=len(observations),target=bundle['observation']['valid'],
                    guidance=bundle['candidate']['reason'],depth=ticks[-1]['guardian']['reason'])),flush=True)
    except Exception as exc:
        failure=dict(type=type(exc).__name__,reason=str(exc))
        raise
    finally:
        # Evidence serialization is outside measured inference. Full loop wall time
        # is recorded separately; no compressed logging cost is called realtime.
        write_rows(folder/'ticks.jsonl',ticks)
        write_rows(folder/'observations.jsonl',observations)
        write_rows(folder/'depth.jsonl',depth_rows)
        for index,frame in depth_arrays:
            np.savez_compressed(depth_directory/('%06d.npz'%index),rgb=frame.rgb,depth_m=frame.depth_m)
        for index,arrays in observation_arrays:
            np.savez_compressed(frame_directory/('%06d.npz'%index),**arrays)
        save_json(folder/'episode.json',dict(status='failed' if failure else 'completed',failure=failure,
            whole_wall_s=time.perf_counter()-started,physics_ticks=len(ticks),observations=len(observations),
            depth_events=len(depth_rows),fresh_depth_frames=len(depth_arrays),final_state=asdict(world.state()),
            actual_yolo_calls=sum(r['detections']['inference_executed'] for r in observations),
            actual_flyvis_observations=len(observations),neural_steps=len(observations)*5,
            mode='offline_six_dof_measured_inference_delay',real_time=False,physical_flight=False,
            full_fly_motor_brain=False,autopilot='conventional_geometric_controller',
            rotor_forces_are_the_only_aircraft_actuation=True))
        world.close()
    return json.loads((folder/'episode.json').read_text())


def run(output,specs,development=False):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    sources={p:sha(ROOT/p) for p in RUNTIME_SOURCES}
    criteria=None
    if not development:
        from experiments.flight_report import LIMITS
        sources['experiments/flight_report.py']=sha(ROOT/'experiments/flight_report.py')
        criteria=dict(LIMITS)
    (output/'sources').mkdir()
    for p in sources:shutil.copy2(ROOT/p,output/'sources'/p.replace('/','__'))
    save_json(output/'definition.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),specs=specs,
        development=development,physics_dt_s=PHYSICS_DT,depth_period_s=DEPTH_PERIOD_S,
        maximum_observation_age_s=MAX_OBSERVATION_AGE_S,maximum_command_capture_age_s=COMMAND_MAX_CAPTURE_AGE_S,
        requested_speed_cap_m_s=MAX_SPEED_M_S,source_hashes=sources,
        detector_model=str(MODEL),detector_sha256=TINY_SHA,detector_confidence=.5,detector_input_size=416,
        flyvis_manifest_sha256=sha(MANIFEST),source_patch_sha256=sha(PATCH),
        acceptance_criteria=criteria,
        mode='offline_six_dof_measured_inference_delay',physical_control_authority=False,
        acceptance_source='experiments/flight_report.py; frozen before final held-out runs',
        source_representation='Fixed vertical unsegmented photograph; ideal enclosed room and ideal state/depth sensors'))
    shutil.copy2(ROOT/'results/hybrid_flight_run01/provenance.json',output/'photo_provenance.json')
    shutil.copy2(PATCH,output/'source_patch.png')
    try:
        vision=FlightVision()
        save_json(output/'runtime.json',dict(packages=vision.brain.adapter.runtime,device=str(vision.brain.adapter.device),
            mujoco='3.2.7',detector='YOLOX-tiny',readout_frozen=vision.frozen))
        save_json(output/'readout.json',vision.readout.to_dict())
        for spec in specs:
            result=run_episode(spec,output/spec['name'],vision)
            print(json.dumps(dict(case=spec['name'],**result)),flush=True)
        if any(sha(ROOT/p)!=h for p,h in sources.items()):raise RuntimeError('Executed sources changed during run')
        save_json(output/'execution_complete.json',dict(cases=[s['name'] for s in specs],complete=True,
            finished_utc=datetime.now(timezone.utc).isoformat(),development=development))
    except Exception as exc:
        save_json(output/'failure.json',dict(type=type(exc).__name__,reason=str(exc)))
        raise
    finally:
        save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name!='artifact_hashes.json'})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=[s['name'] for s in SPECS])
    parser.add_argument('--duration',type=float)
    parser.add_argument('--development',action='store_true')
    args=parser.parse_args()
    if args.duration is not None and not args.development:parser.error('--duration requires --development')
    specs=[dict(s) for s in SPECS if args.case is None or s['name']==args.case]
    if args.duration is not None:
        if not .1<=args.duration<=30:parser.error('development duration must be .1..30s')
        for spec in specs:spec['duration_s']=args.duration
    run(args.output,specs,args.development)


if __name__=='__main__':main()
