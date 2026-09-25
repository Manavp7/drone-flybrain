"""Actual YOLOX -> actual Flyvis -> image-servo point-mass feedback experiment.

Run from the project root: .venv/bin/python -m experiments.hybrid_demo --output NEW_DIR
The simulator pauses for host computation. This is not a real-time flight loop.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
import numpy as np

from experiments.hybrid_control import CONTROL_CONFIG, HybridController, NeuralReadout, active_velocity
from experiments.hybrid_flyvis import HybridFlyvis
from experiments.hybrid_scene import extract_person_patch, render_scene
from experiments.hybrid_target import TargetBridge, salience_mask
from flybrain_sim.contracts import Scenario, VehicleState
from flybrain_sim.physics import step as plant_step
from perception.detector import YOLOXDetector
from perception.pipeline import CameraSample, PerceptionPipeline

ROOT = Path(__file__).resolve().parents[1]
YOLO_SHA = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
OBS_DT = .1
SPECS = [
    dict(name='approach_left', target=[5., -.85, 1.25], duration_s=10., kind='static', ablation=False),
    dict(name='approach_right', target=[5.5, .95, .75], duration_s=10., kind='static', ablation=False),
    dict(name='moving_target', target=[4.6, .35, 1.], duration_s=12., kind='moving', ablation=False),
    dict(name='target_loss', target=[5., .8, 1.2], duration_s=8., kind='loss', hidden_after_s=4., ablation=False),
    dict(name='zero_neural_features', target=[5., -.85, 1.25], duration_s=10., kind='static', ablation=True),
]
CRITERIA = dict(tail_seconds=2., static_center_norm=.10, static_height_error=.055,
                moving_center_norm=.15, moving_height_error=.07,
                observed_fraction=.9, minimum_optical_depth_m=2.,
                loss_brake_response_s=4.1, loss_settle_by_s=6., loss_speed_m_s=.1,
                loss_pre_observed_fraction=.9, loss_pre_max_speed_min_m_s=.1,
                validation_max_center_abs=.08, validation_max_height_abs=.055)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def calibration_specs(validation=False):
    xys = [-.2, .2] if validation else [-.4, 0., .4]
    ys = [-.2, .2] if validation else [-.3, 0., .3]
    heights = [.27, .37, .47] if validation else [.22, .32, .42, .52]
    specs = [dict(x=x, y=y, height=h, aspect=.47 if validation else .40)
             for x, y, h in itertools.product(xys, ys, heights)]
    np.random.default_rng(1702 if validation else 1701).shuffle(specs)
    return specs


def calibration(brain, specs, output):
    output.mkdir()
    brain.reset()
    features, labels, activity, retina, boxes, validity = [], [], [], [], [], []
    for spec in specs:
        cx, cy = (spec['x']+1)*391/2, (spec['y']+1)*391/2
        h = spec['height']*391; w = h*spec['aspect']
        box = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
        mask = salience_mask(dict(valid=True, bbox_xyxy=box), (391, 391))
        for _ in range(4):
            response = brain.step(mask, OBS_DT)
            features.append(response['features']); activity.append(response['activity'])
            retina.append(response['retina']); validity.append(response['valid'])
            labels.append([spec['x'], spec['y'], spec['height']]); boxes.append(box)
    np.savez_compressed(output/'neural.npz', features=features, labels=labels, activity=activity,
                        retina=retina, boxes=boxes, valid=validity,
                        baseline=brain.baseline_activity, cell_indices=brain.cell_indices,
                        centers_rc=brain.centers_rc)
    return np.array(features), np.array(labels), bool(all(validity))


def target_at(spec, timestamp):
    target = np.array(spec['target'], float)
    if spec['kind'] == 'moving':
        target[1] += .45*np.sin(.6*timestamp)
        target[2] += .18*np.sin(.4*timestamp)
    return target


def person_truth(patch_truth, patch_meta):
    """Evaluator-only projection of original source person inside photo patch."""
    bounds = patch_truth['unclipped_bbox_xyxy']
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    crop = patch_meta['crop_xyxy']; src = patch_meta['person_bbox_source_xyxy']
    sx, sy = (right-left)/(crop[2]-crop[0]), (bottom-top)/(crop[3]-crop[1])
    return [left+(src[0]-crop[0])*sx, top+(src[1]-crop[1])*sy,
            left+(src[2]-crop[0])*sx, top+(src[3]-crop[1])*sy]


def image_errors(box):
    if box is None:
        return None, None
    x0, y0, x1, y1 = box
    center = np.array([(x0+x1)/391-1, (y0+y1)/391-1])
    return float(np.linalg.norm(center)), float(abs((y1-y0)/391-CONTROL_CONFIG['desired_height_fraction']))


def score_episode(rows, spec):
    expected = round(spec['duration_s']/OBS_DT)
    if (len(rows) != expected or not np.allclose(
            [r['capture_time_s'] for r in rows], np.arange(expected)*OBS_DT,
            rtol=0, atol=1e-8)):
        raise ValueError('Scoring requires a complete regularly sampled episode')
    depth = min(r['truth']['relative_depth_m'] for r in rows)
    tail = [r for r in rows if r['capture_time_s'] >= spec['duration_s']-CRITERIA['tail_seconds']-1e-9]
    observed = float(np.mean([r['target_observation']['valid'] for r in rows]))
    center = [image_errors(r['person_truth_bbox'])[0] for r in tail if r['person_truth_bbox'] is not None]
    size = [image_errors(r['person_truth_bbox'])[1] for r in tail if r['person_truth_bbox'] is not None]
    out = dict(observed_fraction=observed, minimum_optical_depth_m=depth,
               tail_truth_fraction=float(np.mean([r['person_truth_bbox'] is not None for r in tail])),
               initial_center_norm=image_errors(rows[0]['person_truth_bbox'])[0],
               tail_mean_center_norm=float(np.mean(center)) if center else None,
               tail_mean_height_error=float(np.mean(size)) if size else None,
               final_position=rows[-1]['state_after']['position'],
               maximum_speed_m_s=max(float(np.linalg.norm(r['state_after']['velocity'])) for r in rows),
               selected_ids=sorted({r['target_observation']['track_id'] for r in rows
                                    if r['target_observation']['track_id'] is not None}))
    if spec['kind'] == 'loss':
        lost = [r for r in rows if r['capture_time_s'] >= spec['hidden_after_s']-1e-9]
        before_loss = [r for r in rows if r['capture_time_s'] < spec['hidden_after_s']-1e-9]
        settled = [r for r in rows if r['state_after']['time'] >= CRITERIA['loss_settle_by_s']-1e-9]
        out.update(all_loss_commands_brake=all(r['new_command']['velocity'] == [0.,0.,0.] for r in lost),
                   pre_loss_observed_fraction=float(np.mean([r['target_observation']['valid'] for r in before_loss])),
                   pre_loss_max_speed_m_s=max(float(np.linalg.norm(r['state_after']['velocity'])) for r in before_loss),
                   first_loss_response_s=lost[0]['new_command']['issued_at_s'],
                   settled_max_speed_m_s=max(float(np.linalg.norm(r['state_after']['velocity'])) for r in settled))
        out['passed'] = bool(out['all_loss_commands_brake'] and out['first_loss_response_s'] <= 4.1+1e-9
                             and out['settled_max_speed_m_s'] < .1 and depth >= 2. and len(out['selected_ids']) == 1
                             and out['pre_loss_observed_fraction'] >= CRITERIA['loss_pre_observed_fraction']
                             and out['pre_loss_max_speed_m_s'] > CRITERIA['loss_pre_max_speed_min_m_s'])
    else:
        moving = spec['kind'] == 'moving'
        out['passed'] = bool(center and size and out['tail_truth_fraction'] == 1. and observed >= .9 and depth >= 2.
                             and out['tail_mean_center_norm'] <= CRITERIA['moving_center_norm' if moving else 'static_center_norm']
                             and out['tail_mean_height_error'] <= CRITERIA['moving_height_error' if moving else 'static_height_error'])
    out['meaning'] = 'causal_integrity_ablation' if spec['ablation'] else 'rendered_billboard_control_test'
    return out


def overlay_frame(rgb, mask, response, row, history, spec):
    image = np.full((720, 1100, 3), (21, 25, 32), np.uint8)
    image[78:469, 25:416] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    obs = row['target_observation']
    if obs['valid']:
        x0,y0,x1,y1 = np.rint(obs['bbox_xyxy']).astype(int)
        cv2.rectangle(image, (25+x0,78+y0),(25+x1,78+y1),(70,230,80),2)
    cv2.drawMarker(image, (220,273),(255,255,255),cv2.MARKER_CROSS,18,1)
    tiny = cv2.resize(np.uint8(mask*255),(220,220))
    image[78:298,440:660] = cv2.cvtColor(tiny,cv2.COLOR_GRAY2BGR)
    # Actual L2 response evidence, not simulated pixel activity.
    evidence = response['evidence_for_preview']
    for (rr,cc), value in zip(response['centers_rc'], evidence):
        px,py = 695+int(cc/391*350), 80+int(rr/391*350)
        c = int(np.clip(value / max(float(evidence.max()),1e-8)*255,0,255))
        if 0 <= px < 1100 and 0 <= py < 720:
            cv2.circle(image,(px,py),3,(40,c,255-c),-1)
    font=cv2.FONT_HERSHEY_SIMPLEX
    def text(s,x,y,scale=.55,color=(225,230,240)):
        cv2.putText(image,s,(x,y),font,scale,color,1,cv2.LINE_AA)
    text('YOLO + FLYVIS / CLOSED-LOOP SIMULATION',25,30,.75)
    text(spec['name']+'  |  t=%.1fs'%row['capture_time_s'],25,57,.60)
    text('Actual YOLO camera detections',25,492)
    text('Encoded target cue',440,323)
    text('Actual L2 neural activity',695,457)
    text('CAMERA -> YOLO -> FLYVIS -> READOUT -> VELOCITY',25,530,.63)
    text('Target: '+(str(obs['track_id']) if obs['valid'] else 'LOST / BRAKE'),25,561)
    text('Command active now: '+str(np.round(row['applied_velocity_first'],3).tolist()),25,590)
    text('Position (m): '+str(np.round(row['state_before']['position'],3).tolist()),25,619)
    text('Source photo: Vicente Quintero / QuinteroP, CC BY 3.0',25,659,.46)
    text('Photo billboard + point-mass plant. Offline compute; not PX4 or physical flight.',25,687,.48)
    # Top-down trajectory inset, scaled identically in each episode.
    def point(p): return (int(870+p[1]*60),int(645-p[0]*20))
    for a,b in zip(history[:-1],history[1:]):
        cv2.line(image,point(a),point(b),(220,190,40),2)
    cv2.circle(image,point(row['state_after']['position']),5,(60,240,90),-1)
    cv2.circle(image,point(row['target_truth_position']),6,(100,120,255),2)
    text('Top-down motion (truth display)',695,510,.46)
    if spec['ablation']:
        text('ABLATION: neural features zeroed',695,478,.47,(80,180,255))
    return image


def run_episode(brain, detector, readout, patch, patch_meta, spec, output, writer):
    output.mkdir()
    pipeline = PerceptionPipeline(detector, max_frame_age_s=30., appearance_tracking=True)
    bridge, controller = TargetBridge(), HybridController(readout)
    brain.reset()
    scenario = Scenario(seed=1710, category='hybrid_billboard', bounds=(12.,12.,6.), home=(0.,0.,1.),
                        waypoints=(), obstacles=(), sensor_noise=0., dt=.02)
    state = VehicleState(0.,(0.,0.,1.),(0.,0.,0.),22.)
    command=None; rows=[]; frames=[]; activities=[]; retinas=[]; features=[]; supplied=[]; history=[]
    pipeline_rows=[]; neural_seconds=0.; start=time.perf_counter()
    for i in range(round(spec['duration_s']/OBS_DT)):
        capture_time=i*OBS_DT
        target=target_at(spec,capture_time)
        hidden=spec['kind']=='loss' and capture_time >= spec['hidden_after_s']-1e-9
        rgb,truth=render_scene(patch,state.position,target,hidden=hidden)
        detector_result=pipeline.process(CameraSample(rgb,capture_time,time.monotonic(),i,spec['name'],
            'offline_paused_simulation','fixed_forward_camera',capture_age_at_receive_s=0.))
        obs=bridge.update(detector_result,capture_time,(391,391))
        mask=salience_mask(obs,(391,391))
        response=brain.step(mask,OBS_DT)
        control_features=np.zeros(8) if spec['ablation'] else response['features'].copy()
        new_command=controller.command(control_features,target_valid=obs['valid'],neural_valid=response['valid'],
                                       capture_time_s=capture_time,response_time_s=response['response_time_s'])
        before=asdict(state); applied=active_velocity(command,state.time)
        # Neural processing has a declared .1s model delay. Until then only the
        # PREVIOUS command can move the plant; expire it at each .02s substep.
        for _ in range(5):
            state=plant_step(state,active_velocity(command,state.time),scenario)
        assert abs(state.time-response['response_time_s']) < 1e-8
        row=dict(sequence=i,capture_time_s=capture_time,state_before=before,state_after=asdict(state),
                 target_observation=obs,new_command=new_command,applied_velocity_first=list(applied),
                 neural_valid=response['valid'],neural_elapsed_wall_s=response['elapsed_wall_s'],
                 neural_stimulus_time_s=response['stimulus_time_s'],
                 neural_response_time_s=response['response_time_s'],
                 truth=truth,person_truth_bbox=person_truth(truth,patch_meta),target_truth_position=target.tolist())
        # Only after integration may this response's command become active.
        command=new_command
        pipeline_rows.append(detector_result); rows.append(row); frames.append(rgb)
        activities.append(response['activity']); retinas.append(response['retina'])
        features.append(response['features']); supplied.append(control_features)
        neural_seconds+=response['elapsed_wall_s']; history.append(state.position)
        response['centers_rc']=brain.centers_rc
        response['evidence_for_preview']=np.abs(response['activity'][brain.cell_indices]-brain.baseline_activity[brain.cell_indices])
        writer.write(overlay_frame(rgb,mask,response,row,history,spec))
        if i%20==0:
            print(json.dumps(dict(episode=spec['name'],frame=i,target=obs['valid'],position=state.position)),flush=True)
    np.savez_compressed(output/'neural_camera.npz',activity=activities,retina=retinas,features=features,
                        supplied_features=supplied,rgb=frames,baseline=brain.baseline_activity,
                        cell_indices=brain.cell_indices,centers_rc=brain.centers_rc)
    for filename, values in [('trace.jsonl',rows),('detections.jsonl',pipeline_rows)]:
        (output/filename).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in values))
    metrics=score_episode(rows,spec)
    metrics.update(wall_s=time.perf_counter()-start,neural_wall_s=neural_seconds,frames=len(rows),
                   actual_yolo_calls=sum(r['inference_executed'] for r in pipeline_rows),
                   pipeline_status_counts={s:sum(r['status']==s for r in pipeline_rows)
                                           for s in sorted({r['status'] for r in pipeline_rows})})
    save_json(output/'summary.json',metrics)
    print(json.dumps(dict(episode=spec['name'],metrics=metrics)),flush=True)
    return metrics


def finalize_run(output, *, whole_wall_s=None):
    """Finalize COMPLETE saved episodes without replaying inference or motion.

    This also recovers an interrupted video-export stage. Saved runtime source
    snapshots remain immutable; current scoring/export source is recorded apart.
    """
    if (output/'summary.json').exists():
        raise ValueError('Run already finalized; preserve existing evidence')
    definition=json.loads((output/'definition.json').read_text())
    frozen=json.loads((output/'selection_frozen.json').read_text())
    if (sha(output/'definition.json') != frozen['definition_sha256']
            or sha(output/'readout.json') != frozen['readout_sha256']):
        raise ValueError('Frozen definition or readout changed')
    for name,digest in definition['source_hashes'].items():
        if sha(output/'sources'/name.replace('/','__')) != digest:
            raise ValueError('Executed source snapshot changed: '+name)
    scores={}
    for spec in definition['specs']:
        episode=output/spec['name']
        rows=[json.loads(s) for s in (episode/'trace.jsonl').read_text().splitlines()]
        previous=json.loads((episode/'summary.json').read_text())
        corrected=score_episode(rows,spec)
        # Hardening only rejects inadequate input coverage, never changes a
        # complete run's original geometric score or outcome.
        for key,value in corrected.items():
            if key in previous and value != previous[key]:
                raise ValueError('Rescoring changed saved metric: '+spec['name']+'/'+key)
        scores[spec['name']]={**previous,**corrected}
    preview=output/'hybrid_preview.mp4'
    if not preview.exists():
        # Explicit environment avoids inheriting C-library OpenMP registration
        # variables added after Python startup by an inference runtime.
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','2',
                        '-i',str(output/'preview_raw.mp4'),'-an','-c:v','libx264','-threads','2',
                        '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(preview)],
                       env=dict(os.environ),check=True)
    cap=cv2.VideoCapture(str(preview)); count=0
    try:
        while True:
            ok,frame=cap.read()
            if not ok: break
            if frame.shape != (720,1100,3): raise ValueError('Unexpected preview dimensions')
            count+=1
    finally: cap.release()
    if count != sum(s['frames'] for s in scores.values()):
        raise ValueError('Preview does not cover all saved episode frames')
    final_source=Path(__file__)
    shutil.copy2(final_source,output/'finalization_source.py')
    summary=dict(status='executed_and_finalized',episodes=scores,
                 validation=json.loads((output/'validation_metrics.json').read_text()),
                 whole_wall_s=whole_wall_s,episode_wall_s=sum(s['wall_s'] for s in scores.values()),
                 timing_note='whole_wall_s unavailable when recovering a completed inference run after export failure',
                 primary_passes=sum(scores[s['name']]['passed'] for s in definition['specs'] if not s['ablation']),
                 primary_cases=4,total_yolo_calls=sum(s['actual_yolo_calls'] for s in scores.values()),
                 actual_flyvis=True,actual_yolo=True,physical_flight=False,px4_flight=False,real_time=False,
                 control_authority='local point-mass simulation only',
                 finalization_source_sha256=sha(final_source),preview_decoded_frames=count,
                 ablation_note='Zero features invokes same evidence gate; verifies dependency, not neural superiority')
    save_json(output/'summary.json',summary)
    save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
                                           if p.is_file() and p.name!='artifact_hashes.json'})
    print(json.dumps(summary),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--finalize-only',action='store_true',help='Finalize already completed saved episodes without inference')
    args=parser.parse_args(); output=args.output.resolve()
    if args.finalize_only:
        finalize_run(output)
        return 0
    output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(2); started=time.perf_counter()
    sources=output/'sources'; sources.mkdir()
    paths=[ROOT/p for p in
        ['experiments/hybrid_demo.py','experiments/hybrid_control.py','experiments/hybrid_flyvis.py',
         'experiments/hybrid_scene.py','experiments/hybrid_target.py','experiments/runtime.py',
         'perception/detector.py','perception/pipeline.py','flybrain_sim/geometry.py',
         'flybrain_sim/physics.py','flybrain_sim/contracts.py','flybrain_sim/research_model.py']]
    source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for path in paths: shutil.copy2(path,sources/str(path.relative_to(ROOT)).replace('/','__'))
    definition=dict(created_utc=datetime.now(timezone.utc).isoformat(),specs=SPECS,criteria=CRITERIA,
                    calibration_train=calibration_specs(),calibration_validation=calibration_specs(True),
                    calibration_steps_per_mask=4,training_uses_all_steps=True,ridge=1e-4,
                    config=CONTROL_CONFIG,observation_dt_s=OBS_DT,plant_dt_s=.02,neural_dt_s=.02,
                    source_hashes=source_hashes,mode='offline_paused_point_mass_simulation',
                    real_time=False,physical_control_authority=False,px4_executed=False,
                    neural_input='YOLO-derived dark target rectangle on gray; not raw camera luminance',
                    limitations=['Visual Flyvis network plus engineered readout and velocity controller; not whole fly brain',
                                 'Fixed-orientation camera, one unsegmented photographic billboard, no obstacles or route planning',
                                 'Host computation pauses the simulator; .1s model delay is not measured onboard latency',
                                 'Apparent size setpoint is not a measured metric distance'])
    save_json(output/'definition.json',definition)
    provenance=json.loads((ROOT/'inputs/sabana_grande/provenance.json').read_text())
    video=ROOT/'inputs/sabana_grande/source.webm'
    if sha(video)!=provenance['sha256']: raise ValueError('Source video changed')
    patch,meta=extract_person_patch(video)
    cv2.imwrite(str(output/'source_patch.png'),cv2.cvtColor(patch,cv2.COLOR_RGB2BGR))
    save_json(output/'provenance.json',dict(video=provenance,crop=meta,yolo_sha256=YOLO_SHA,
                flyvis_manifest_sha256=sha(ROOT/'models/flyvis_0000_000.manifest.json'),
                changes='Cropped and projected photo into synthetic views; annotated and muted preview; no endorsement'))
    brain=HybridFlyvis(ROOT/'models/flyvis_0000_000.manifest.json')
    save_json(output/'runtime.json',dict(packages=brain.adapter.runtime,
             cv2_version=cv2.__version__,device=str(brain.adapter.device),
             flyvis_model_manifest=brain.adapter.locked.manifest,
             model_manifest_sha256=brain.adapter.locked.manifest_sha256))
    x,y,valid=calibration(brain,calibration_specs(),output/'training')
    if not valid: raise ValueError('Insufficient neural calibration evidence')
    readout=NeuralReadout.fit(x,y,ridge=1e-4)
    vx,vy,vvalid=calibration(brain,calibration_specs(True),output/'validation')
    errors=np.abs(readout.predict(vx)-vy)
    validation=dict(all_neural_valid=vvalid,max_abs_error=errors.max(0).tolist(),
                    mean_abs_error=errors.mean(0).tolist(),passed=bool(vvalid and errors[:,:2].max()<=.08
                                                                      and errors[:,2].max()<=.055))
    save_json(output/'validation_metrics.json',validation)
    print(json.dumps(dict(validation=validation)),flush=True)
    save_json(output/'readout.json',readout.to_dict())
    if not validation['passed']:
        save_json(output/'summary.json',dict(status='calibration_validation_failed',validation=validation))
        return 2
    save_json(output/'selection_frozen.json',dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
             readout_sha256=sha(output/'readout.json'),definition_sha256=sha(output/'definition.json'),
             source_hashes=source_hashes,validation=validation))
    detector=YOLOXDetector(ROOT/'models/yolox_s_official/yolox_s.onnx',expected_sha256=YOLO_SHA,
                          input_size=640,confidence=.5,output_format='raw_yolox')
    raw=output/'preview_raw.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),10.,(1100,720))
    if not writer.isOpened(): raise RuntimeError('Video output unavailable')
    scores={}
    try:
        for spec in SPECS:
            scores[spec['name']]=run_episode(brain,detector,readout,patch,meta,spec,output/spec['name'],writer)
    finally: writer.release()
    if any(sha(ROOT/path)!=digest for path,digest in source_hashes.items()):
        raise ValueError('Source changed during execution; do not certify run')
    finalize_run(output,whole_wall_s=time.perf_counter()-started)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
