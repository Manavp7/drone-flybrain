"""Fresh YOLOX and recurrent Flyvis on an offline video; no flight authority.

python -m experiments.hybrid_video --video VIDEO --provenance JSON --output NEW_DIR
The frozen billboard-trained readout is reused unchanged. Its commands here
are diagnostic cue-space requests, not measured 3D motion or collision advice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import cv2
import numpy as np

from experiments.hybrid_control import CONTROL_CONFIG, HybridController, NeuralReadout
from experiments.hybrid_flyvis import HybridFlyvis
from experiments.hybrid_target import TargetBridge, salience_mask
from flybrain_sim.research_model import validate_manifest
from perception.detector import YOLOXDetector
from perception.pipeline import CameraSample, PerceptionPipeline

ROOT = Path(__file__).resolve().parents[1]
OBS_DT = .1
READOUT_RUN = ROOT/'results/hybrid_flight_run01'
SELECTION_SHA = 'e00cfc297ff5b28d6ecb258f83f9663ea2efb768cade54891d3c0dafad4d445f'
READOUT_SHA = '0b69cf817f881cb1cd9064f50cb7deb32a22edb3c53ee06dad78245f0c4a898d'
YOLO_SHA = 'c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063'
MODEL = ROOT/'models/yolox_s_official/yolox_s.onnx'
FLYVIS_MANIFEST = ROOT/'models/flyvis_0000_000.manifest.json'
SOURCE_FILES = ('experiments/hybrid_video.py', 'experiments/hybrid_control.py',
    'experiments/hybrid_flyvis.py', 'experiments/hybrid_target.py', 'experiments/runtime.py',
    'flybrain_sim/research_model.py', 'perception/detector.py', 'perception/pipeline.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_frozen_readout(folder=READOUT_RUN, *, expected_selection_sha=SELECTION_SHA,
                        expected_readout_sha=READOUT_SHA):
    """Pin both the selection receipt and readout, rather than trusting either alone."""
    folder = Path(folder)
    if sha(folder/'selection_frozen.json') != expected_selection_sha:
        raise ValueError('Frozen readout selection receipt hash changed')
    frozen = json.loads((folder/'selection_frozen.json').read_text())
    if (frozen.get('readout_sha256') != expected_readout_sha
            or sha(folder/'readout.json') != expected_readout_sha
            or frozen.get('validation', {}).get('passed') is not True):
        raise ValueError('Frozen readout integrity or original validation failed')
    readout = NeuralReadout(**json.loads((folder/'readout.json').read_text()))
    return readout, frozen


def causal_schedule(pts, start_s=0., duration_s=12.):
    """Select the latest past source image on a .1-second grid; preserve gaps.

    start_s is relative to the first source PTS. The actual grid begins at the
    selected source PTS (at or before the requested start), which is recorded.
    Repeated indices during a gap retain their old capture timestamp; they do
    not become fresh camera images merely because neural time advances.
    """
    values = np.asarray(pts)
    if (values.ndim != 1 or len(values) < 2 or values.dtype.kind not in 'fiu'
            or not np.isfinite(values).all() or np.any(values < 0)
            or np.any(np.diff(values) <= 0)):
        raise ValueError('Source PTS must be finite, nonnegative and strictly increasing')
    for name, value in (('start_s', start_s), ('duration_s', duration_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(name+' must be finite')
    if start_s < 0 or not OBS_DT <= duration_s <= 20:
        raise ValueError('Require start_s>=0 and duration_s between .1 and20 seconds')
    count = round(duration_s/OBS_DT)
    if abs(count*OBS_DT-duration_s) > 1e-9:
        raise ValueError('duration_s must be a multiple of .1 seconds')
    values = values.astype(np.float64)
    requested = values[0]+start_s
    if requested > values[-1]+1e-10:
        raise ValueError('Start is beyond video timestamps')
    first = int(np.searchsorted(values, requested+1e-10, side='right')-1)
    origin = float(values[first])
    grid = origin+np.arange(count, dtype=np.float64)*OBS_DT
    if grid[-1] > values[-1]+1e-10:
        raise ValueError('Requested duration exceeds available video timestamps')
    indices = np.searchsorted(values, grid+1e-10, side='right')-1
    selected = values[indices]
    if np.any(selected > grid+1e-10):
        raise ValueError('Future source frame selected')
    return dict(source_indices=indices, source_pts_s=selected, grid_pts_s=grid,
                capture_times_s=selected-origin, grid_times_s=grid-origin,
                origin_pts_s=origin, requested_start_pts_s=float(requested))


def probe_pts(video):
    # Called before any Torch/Flyvis initialization. Do not infer PTS from FPS.
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_frames', '-show_entries', 'frame=best_effort_timestamp_time,width,height',
        '-of', 'json', str(video)], capture_output=True, text=True, check=True,
        env=dict(os.environ), timeout=120)
    frames = json.loads(result.stdout).get('frames', [])
    if not frames or any('best_effort_timestamp_time' not in row for row in frames):
        raise ValueError('ffprobe did not provide every source presentation timestamp')
    pts = np.array([float(row['best_effort_timestamp_time']) for row in frames], np.float64)
    # Validation does not constrain source frame gaps: stale held images are
    # explicitly rejected by the existing perception/target timing contracts.
    if len(pts) < 2 or not np.isfinite(pts).all() or np.any(pts < 0) or np.any(np.diff(pts) <= 0):
        raise ValueError('Invalid or ambiguous ffprobe presentation timestamps')
    dimensions = {(row['height'], row['width']) for row in frames}
    if len(dimensions) != 1:
        raise ValueError('Changing source video dimensions are unsupported')
    return pts, dict(frames=frames, original_hw=list(next(iter(dimensions))),
                     timestamp_source='ffprobe best_effort_timestamp_time; display order')


def letterbox_metadata(image_hw, original_hw=None):
    height, width = image_hw
    if any(type(v) is not int or v < 2 for v in (height, width)):
        raise ValueError('Image dimensions must be integers>=2')
    scale = 391/max(height, width)
    out = dict(image_hw=[height, width], cue_hw=[391, 391],
        image_to_cue_scale=scale,
        image_to_cue_offset_xy=[(391-width*scale)/2, (391-height*scale)/2],
        cue_coordinates='normalized square coordinates: x right, y down',
        cue_height='box height after letterbox divided by391, not original image height',
        metric_distance_available=False)
    if original_hw is not None:
        oh, ow = original_hw
        if any(type(v) is not int or v < 2 for v in (oh, ow)):
            raise ValueError('Original dimensions must be integers>=2')
        out.update(original_hw=[oh, ow], original_to_image_scale_xy=[width/ow, height/oh])
    return out


class FirstFrameSelector:
    """Optional source-image point is used once; a miss latches explicit loss."""
    def __init__(self, target_point=None):
        if target_point is not None:
            p = np.asarray(target_point)
            if p.shape != (2,) or p.dtype.kind not in 'fiu' or not np.isfinite(p).all() or np.any(p < 0):
                raise ValueError('target_point requires two finite nonnegative coordinates')
            target_point = p.astype(float).tolist()
        self.point = target_point
        self.bridge = None
        self.failed_initialization = False

    def update(self, result, now_s, image_hw, original_hw):
        if self.failed_initialization:
            return TargetBridge()._invalid('manual_target_not_detected_in_first_frame')
        if self.bridge is None:
            selected = None
            if self.point is not None:
                oh, ow = original_hw; h, w = image_hw
                if not 0 <= self.point[0] < ow or not 0 <= self.point[1] < oh:
                    raise ValueError('target_point lies outside the original image')
                x, y = self.point[0]*w/ow, self.point[1]*h/oh
                people = [d for d in result.get('detections', [])
                    if d.get('class_id') == 0 and d.get('label') == 'person'
                    and d['bbox_xyxy'][0] <= x < d['bbox_xyxy'][2]
                    and d['bbox_xyxy'][1] <= y < d['bbox_xyxy'][3]]
                if result.get('status') == 'ok' and people:
                    selected = min(people, key=lambda d: (-d['confidence'], d['track_id']))['track_id']
                else:
                    self.failed_initialization = True
                    return TargetBridge()._invalid('manual_target_not_detected_in_first_frame')
            self.bridge = TargetBridge(selected)
        return self.bridge.update(result, now_s, image_hw)


def decode_selected(video, schedule, original_hw):
    """Decode sequentially from frame zero; no approximate time seek."""
    needed = sorted(set(schedule['source_indices'].tolist()))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError('Cannot open source video')
    frames, index, wanted = {}, 0, 0
    h, w = original_hw
    ratio = min(1., 960/max(h, w))
    image_hw = [max(2, round(h*ratio)), max(2, round(w*ratio))]
    try:
        while wanted < len(needed):
            ok, frame = capture.read()
            if not ok:
                raise ValueError('Sequential video decode ended before selected source PTS')
            if frame.shape[:2] != tuple(original_hw):
                raise ValueError('Decoded source dimensions differ from ffprobe')
            if index == needed[wanted]:
                resized = cv2.resize(frame, (image_hw[1], image_hw[0]), interpolation=cv2.INTER_AREA)
                frames[index] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                wanted += 1
            index += 1
    finally:
        capture.release()
    return frames, image_hw


def preview_frame(rgb, mask, response, observation, detections, command, row, provenance, brain):
    canvas = np.full((720, 1100, 3), (24, 28, 35), np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    def text(s, x, y, scale=.5, color=(230,230,230)):
        cv2.putText(canvas, s, (x,y), font, scale, color, 1, cv2.LINE_AA)
    h, w = rgb.shape[:2]; ratio = min(640/w, 450/h)
    wh = (round(w*ratio), round(h*ratio)); left, top = 20+(640-wh[0])//2, 90+(450-wh[1])//2
    image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for d in detections.get('detections', []):
        x0,y0,x1,y1 = np.rint(d['bbox_xyxy']).astype(int)
        selected = observation['valid'] and d['track_id'] == observation['track_id']
        color = (70,230,90) if selected else (140,145,155)
        cv2.rectangle(image, (x0,y0), (x1,y1), color, 2 if selected else 1)
        cv2.putText(image, '%s #%s %.2f'%(d['label'],d['track_id'],d['confidence']),
                    (x0,max(14,y0-4)), font, .4, color, 1, cv2.LINE_AA)
    canvas[top:top+wh[1],left:left+wh[0]] = cv2.resize(image,wh)
    canvas[90:275,680:865] = cv2.cvtColor(cv2.resize(np.uint8(mask*255),(185,185)),cv2.COLOR_GRAY2BGR)
    evidence = np.abs(response['activity'][brain.cell_indices]-brain.baseline_activity[brain.cell_indices])
    for (r,c), value in zip(brain.centers_rc,evidence):
        x,y=885+int(c/391*190),90+int(r/391*190)
        intensity=int(np.clip(value/max(float(evidence.max()),1e-8)*255,0,255))
        if 0 <= x < 1100 and 0 <= y < 720:
            cv2.circle(canvas,(x,y),2,(30,intensity,255-intensity),-1)
    text('OFFLINE VIDEO / YOLO + ACTUAL FLYVIS',20,32,.8)
    text('Guidance signals only. No simulated or physical drone flight.',20,60,.6)
    text('Encoded YOLO cue',680,300,.46); text('Actual L2 evidence',885,300,.46)
    text('Target: '+(str(observation['track_id']) if observation['valid'] else 'LOST / BRAKE'),680,340,.6)
    velocity=command['velocity']; origin=(860,410)
    cv2.arrowedLine(canvas,origin,(int(origin[0]+velocity[1]*65),int(origin[1]-velocity[2]*65)),(60,220,240),3,tipLength=.2)
    text('Image right/up request',700,470,.5)
    text('Approach-size request: %+.3f'%velocity[0],700,500,.48)
    text('Command available at %.2fs'%command['issued_at_s'],680,535,.48)
    text(command['reason'],680,565,.43)
    text('Source PTS %.3fs | grid %.2fs | neural response %.2fs'%(
        row['source_pts_s'],row['grid_time_s'],row['neural_response_time_s']),20,584,.55)
    text('Cue-space bearing and apparent size; no metric range or collision estimate.',20,615,.5)
    text(('Source: '+provenance['title'])[:120],20,651,.44)
    text((provenance['author']+' | '+provenance['license'])[:130],20,680,.44)
    return canvas


def run(video, provenance_path, output, start_s=0., duration_s=12., target_point=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    started=time.perf_counter(); rows=[]; writer=None
    try:
        video=Path(video).resolve(); provenance=json.loads(Path(provenance_path).read_text())
        for field in ('source_url','title','author','license','sha256'):
            if not isinstance(provenance.get(field),str) or not provenance[field].strip():
                raise ValueError('Provenance requires '+field)
        if (not re.fullmatch('[0-9a-f]{64}',provenance['sha256'])
                or sha(video) != provenance['sha256']):
            raise ValueError('Source video does not match provenance SHA256')
        pts, probe=probe_pts(video)
        schedule=causal_schedule(pts,start_s,duration_s)
        frames,image_hw=decode_selected(video,schedule,probe['original_hw'])
        selector=FirstFrameSelector(target_point)
        if target_point is not None and not (0 <= target_point[0] < probe['original_hw'][1]
                                             and 0 <= target_point[1] < probe['original_hw'][0]):
            raise ValueError('target_point outside original image')
        readout,frozen=load_frozen_readout()
        locked=validate_manifest(FLYVIS_MANIFEST)
        if sha(MODEL) != YOLO_SHA:
            raise ValueError('YOLO model hash changed')
        (output/'sources').mkdir(); (output/'samples').mkdir()
        source_hashes={name:sha(ROOT/name) for name in SOURCE_FILES}
        for name in SOURCE_FILES:
            shutil.copy2(ROOT/name,output/'sources'/name.replace('/','__'))
        for name in ('readout.json','selection_frozen.json'):
            shutil.copy2(READOUT_RUN/name,output/('prior_'+name))
        save_json(output/'provenance.json',provenance); save_json(output/'ffprobe.json',probe)
        definition=dict(frozen_utc=datetime.now(timezone.utc).isoformat(),video=str(video),
            video_sha256=provenance['sha256'],source_hashes=source_hashes,
            source_original_hw=probe['original_hw'],letterbox=letterbox_metadata(image_hw,probe['original_hw']),
            start_s=start_s,duration_s=duration_s,observation_dt_s=OBS_DT,target_point_original_xy=target_point,
            target_selection='first_frame_point_then_fixed_track_id' if target_point is not None else 'first_highest_confidence_person_then_fixed_track_id',
            sampling={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in schedule.items()},
            source_timestamps_s=pts.tolist(),readout_sha256=READOUT_SHA,selection_sha256=SELECTION_SHA,
            flyvis_manifest_sha256=locked.manifest_sha256,flyvis_model_manifest=locked.manifest,
            yolo_model_sha256=YOLO_SHA,yolo_confidence=.5,yolo_iou=.45,yolo_classes='all80COCO',
            control_config=CONTROL_CONFIG,mode='offline_recorded_video_guidance',real_time=False,
            physical_control_authority=False,flight_simulated=False,
            command_units='unapplied legacy servo units; not validated metric3D velocity',
            neural_input='YOLO target rectangle ongray391square with isotropic letterbox; not rawcamera',
            limitations=['Frozen readout trained on synthetic target cues; no refitting to this video',
                         'Temporary track IDs do not establish identity or accuracy',
                         'No drone physics, world coordinates, obstacles, collision assessment or flight authority',
                         'Commands available only at neural response time; host processing is offline'])
        save_json(output/'definition.json',definition)
        save_json(output/'frozen_inputs.json',dict(definition_sha256=sha(output/'definition.json'),
            source_hashes=source_hashes,input_sha256=provenance['sha256'],readout_sha256=READOUT_SHA))
        cv2.setNumThreads(2)
        detector=YOLOXDetector(MODEL,expected_sha256=YOLO_SHA,confidence=.5,nms_iou=.45,
                               input_size=640,output_format='raw_yolox')
        pipeline=PerceptionPipeline(detector,max_frame_age_s=30.,appearance_tracking=True)
        brain=HybridFlyvis(FLYVIS_MANIFEST); controller=HybridController(readout)
        np.savez_compressed(output/'neural_binding.npz',baseline=brain.baseline_activity,
                            cell_indices=brain.cell_indices,centers_rc=brain.centers_rc)
        save_json(output/'runtime.json',dict(packages=brain.adapter.runtime,device=str(brain.adapter.device),
            reset_elapsed_wall_s=brain.reset_elapsed_wall_s,opencv_version=cv2.__version__))
        writer=cv2.VideoWriter(str(output/'preview_raw.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),10.,(1100,720))
        if not writer.isOpened(): raise ValueError('Cannot create preview writer')
        with (output/'trace.jsonl').open('w') as trace_file, (output/'detections.jsonl').open('w') as detection_file:
            for i,source_index in enumerate(schedule['source_indices']):
                rgb=frames[int(source_index)]; capture=float(schedule['capture_times_s'][i]); grid=i*OBS_DT
                if i and source_index == schedule['source_indices'][i-1]:
                    # A held old image is not new detector evidence. Avoid
                    # resetting the tracker (and reusing IDs) on duplicate PTS.
                    result=dict(sequence=i,stream_id='hybrid_offline_video',
                        clock_domain='normalized_source_pts',frame_id='recorded_video_image',
                        capture_time_s=capture,status='held_source_frame',
                        inference_executed=False,detections=[],control_authority=False)
                else:
                    result=pipeline.process(CameraSample(rgb,capture,time.monotonic(),i,'hybrid_offline_video',
                        'normalized_source_pts','recorded_video_image',capture_age_at_receive_s=max(0.,grid-capture)))
                observation=selector.update(result,grid,image_hw,probe['original_hw'])
                mask=salience_mask(observation,image_hw)
                response=brain.step(mask,OBS_DT)
                command=controller.command(response['features'],target_valid=observation['valid'],
                    neural_valid=response['valid'],capture_time_s=capture,response_time_s=response['response_time_s'])
                row=dict(sequence=i,source_index=int(source_index),source_pts_s=float(schedule['source_pts_s'][i]),
                    capture_time_s=capture,grid_time_s=grid,source_age_at_grid_s=max(0.,grid-capture),
                    neural_stimulus_time_s=response['stimulus_time_s'],neural_response_time_s=response['response_time_s'],
                    neural_elapsed_wall_s=response['elapsed_wall_s'],neural_valid=response['valid'],
                    target_observation=observation,command=command,command_applied=False,
                    inference_executed=result['inference_executed'],perception_status=result['status'],
                    guidance_space='letterboxed visual cue',metric_distance_m=None)
                np.savez_compressed(output/'samples'/('%06d.npz'%i),rgb=rgb,mask=mask,
                    activity=response['activity'],retina=response['retina'],features=response['features'])
                trace_file.write(json.dumps(row,allow_nan=False)+'\n');trace_file.flush()
                detection_file.write(json.dumps(result,allow_nan=False)+'\n');detection_file.flush()
                rows.append(row)
                writer.write(preview_frame(rgb,mask,response,observation,result,command,row,provenance,brain))
                if i%20 == 0: print(json.dumps(dict(frame=i,target=observation['valid'],reason=command['reason'])),flush=True)
        writer.release();writer=None
        if sha(video) != provenance['sha256'] or any(sha(ROOT/n)!=v for n,v in source_hashes.items()):
            raise ValueError('Input or executed source changed during run')
        validate_manifest(FLYVIS_MANIFEST)
        subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-threads','2',
            '-i',str(output/'preview_raw.mp4'),'-an','-c:v','libx264','-threads','2',
            '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'hybrid_video_preview.mp4')],
            check=True,env=dict(os.environ))
        preview=cv2.VideoCapture(str(output/'hybrid_video_preview.mp4'));count=0
        try:
            while True:
                ok,frame=preview.read()
                if not ok: break
                if frame.shape != (720,1100,3): raise ValueError('Preview dimensions changed')
                count+=1
        finally: preview.release()
        if count != len(rows): raise ValueError('Preview frame count does not match all observations')
        observed=[r['target_observation']['valid'] for r in rows]
        status_counts=dict(Counter(r['perception_status'] for r in rows))
        summary=dict(status='completed' if all(r['perception_status']=='ok' for r in rows) else 'completed_with_errors',
            frames=len(rows),preview_frames=count,actual_yolo_calls=sum(r['inference_executed'] for r in rows),
            actual_flyvis_observations=len(rows),neural_steps=len(rows)*5,observed_frames=sum(observed),
            observed_fraction=float(np.mean(observed)),selected_track_ids=sorted({r['target_observation']['track_id']
                for r in rows if r['target_observation']['track_id'] is not None}),
            command_reason_counts=dict(Counter(r['command']['reason'] for r in rows)),
            perception_status_counts=status_counts,neural_wall_s=sum(r['neural_elapsed_wall_s'] for r in rows),
            whole_wall_s=time.perf_counter()-started,media_grid_duration_s=len(rows)*OBS_DT,
            real_time=False,physical_flight=False,flight_simulated=False,commands_applied=0,
            accuracy_measured=False,readout_refitted=False,
            interpretation='Observation coverage is not tracking accuracy; diagnostic visual guidance only')
        save_json(output/'summary.json',summary)
        save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name!='artifact_hashes.json'})
        print(json.dumps(summary),flush=True)
        return 0 if summary['status']=='completed' else 2
    except Exception as exc:
        if writer is not None: writer.release()
        save_json(output/'failure.json',dict(status='failed',exception=type(exc).__name__,reason=str(exc),
            completed_frames=len(rows),whole_wall_s=time.perf_counter()-started,
            physical_control_authority=False,partial_evidence_preserved=True))
        save_json(output/'artifact_hashes.json',{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
            if p.is_file() and p.name!='artifact_hashes.json'})
        raise


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--provenance',type=Path,required=True)
    parser.add_argument('--start-s',type=float,default=0.)
    parser.add_argument('--duration-s',type=float,default=12.)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--target-point',type=float,nargs=2,metavar=('X','Y'))
    args=parser.parse_args()
    return run(args.video,args.provenance,args.output,args.start_s,args.duration_s,args.target_point)


if __name__ == '__main__':
    raise SystemExit(main())
