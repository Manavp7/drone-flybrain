"""Optional detector CLI. Input images and Gazebo frames never become flight commands."""
from __future__ import annotations
import argparse
from dataclasses import fields
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import time
import zipfile
import numpy as np
from .pipeline import CameraSample, PerceptionPipeline


def json_write(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def package_status():
    packages = {name: importlib.util.find_spec(name) is not None for name in ('numpy','cv2','rclpy','torch','flyvis')}
    from flybrain_sim.px4_sitl import environment_status
    return {'python': platform.python_version(), 'packages': packages,
            'object_detection_runtime_available': packages['cv2'],
            'actual_object_detection_run': False, 'actual_flyvis_run': False,
            'actual_px4_run': False, 'control_authority': False,
            'px4': environment_status(),
            'model_required': 'Official YOLOX ONNX plus local SHA256; weights are not bundled',
            'scope': 'Dependency availability only; runtime/model inference and flight are separate checks'}


def load_sample(path):
    # Bounded uncompressed members before NumPy allocation; no pickled objects.
    import zipfile
    if path.stat().st_size > 100_000_000:
        raise ValueError('camera spool exceeds 100MB')
    with zipfile.ZipFile(path) as z:
        if len(z.infolist()) > 20 or sum(i.file_size for i in z.infolist()) > 100_000_000:
            raise ValueError('camera spool uncompressed size exceeds limit')
    with np.load(path,allow_pickle=False) as data:
        names = {f.name for f in fields(CameraSample)}
        if set(data.files)-names:
            raise ValueError('unexpected camera spool fields')
        values = {}
        for key in data.files:
            a = data[key]
            if key in ('image_rgb','depth_m'):
                values[key] = a.copy()
            elif key == 'camera_intrinsics':
                if a.shape != (4,):
                    raise ValueError('intrinsics must contain four values')
                values[key] = tuple(float(v) for v in a)
            else:
                if a.ndim != 0:
                    raise ValueError('metadata fields must be scalars')
                values[key] = a.item()
    sample = CameraSample(**values)
    sample.validate()
    return sample


def make_detector(args):
    from .detector import YOLOXDetector
    detector = YOLOXDetector(args.model,input_size=args.input_size,expected_sha256=args.sha256,
                             confidence=args.confidence,output_format=args.output_format)
    if getattr(args, 'tile_size', 0):
        from .tiled_detector import TiledDetector
        detector = TiledDetector(detector, tile_size=args.tile_size)
    return detector


def annotate(image_rgb, result):
    import cv2
    bgr = np.ascontiguousarray(image_rgb[:,:,::-1])
    for det in result['detections']:
        x1,y1,x2,y2 = [int(v) for v in det['bbox_xyxy']]
        cv2.rectangle(bgr,(x1,y1),(x2,y2),(20,220,220),2)
        label = f"{det['label']} {det['confidence']:.2f} ID{det['track_id']}"
        cv2.putText(bgr,label,(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,220,220),1,cv2.LINE_AA)
    cv2.putText(bgr,result['status'],(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),2,cv2.LINE_AA)
    return bgr


def offline(args):
    import cv2
    if not args.input.is_file():
        raise ValueError('input must be an existing local image/video')
    detector = make_detector(args)
    appearance_tracking = not getattr(args, 'no_appearance_tracking', False)
    pipeline = PerceptionPipeline(detector,max_frame_age_s=args.max_age_s,
                                  appearance_tracking=appearance_tracking)
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic()
    count = inference_count = detected = 0
    status_counts = {}
    fps = None
    capture = writer = None
    try:
        if args.command == 'image':
            image = cv2.imread(str(args.input),cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError('image could not be decoded')
            iterator = [(0,0.,image)]
        else:
            capture = cv2.VideoCapture(str(args.input))
            if not capture.isOpened():
                raise ValueError('video could not be opened')
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not math.isfinite(fps) or fps <= 0:
                raise ValueError('video requires a finite positive frame rate')
            def frames():
                for index in range(args.max_frames):
                    ok,bgr = capture.read()
                    if not ok:
                        break
                    # Decode index/fps is declared file timing, not live capture time.
                    if index % args.sample_every == 0:
                        yield index,index/fps,bgr
            iterator = frames()
        with (args.output/'detections.jsonl').open('x') as log:
            for index,timestamp,bgr in iterator:
                rgb = np.ascontiguousarray(bgr[:,:,::-1])
                sample = CameraSample(rgb,timestamp,time.monotonic(),index,'offline','file','camera_optical',0.)
                result = pipeline.process(sample)
                result['input_mode'] = 'offline_'+args.command
                log.write(json.dumps(result,allow_nan=False)+'\n')
                log.flush()
                count += 1
                inference_count += int(result['inference_executed'])
                detected += len(result['detections'])
                status_counts[result['status']] = status_counts.get(result['status'], 0) + 1
                rendered = annotate(rgb,result)
                if args.command == 'image':
                    if not cv2.imwrite(str(args.output/'annotated.jpg'),rendered):
                        raise OSError('failed writing annotated image')
                else:
                    if writer is None:
                        writer = cv2.VideoWriter(str(args.output/'annotated.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps/args.sample_every,(rendered.shape[1],rendered.shape[0]))
                        if not writer.isOpened():
                            raise OSError('video encoder unavailable')
                    writer.write(rendered)
        if count == 0:
            raise ValueError('no frames decoded')
        successful = status_counts.get('ok', 0) == count
        summary = {'status':'completed' if successful else 'completed_with_errors','input_mode':'offline_'+args.command,'frames':count,
                   'inference_calls_completed':inference_count,'retained_detections':detected,
                   'frame_status_counts':status_counts,
                   'detector_forward_passes':detector.inference_count,
                   'detector_counter_basis':'Successful detector passes including output validation; failed passes excluded',
                   'confidence_threshold':args.confidence,'model_filename':args.model.name,
                   'input_size':args.input_size,'nominal_source_fps':fps,
                   'tile_size':getattr(args, 'tile_size', 0),
                   'sample_every':args.sample_every if args.command == 'video' else 1,
                   'max_age_s':args.max_age_s,
                   'appearance_tracking':appearance_tracking,
                   'tracker':('Class-aware IoU>=0.3; person clothing RGB histogram similarity>=0.65; score IoU*similarity; '
                              if appearance_tracking else 'Greedy class-aware IoU>=0.3; ')
                              + 'last-observed max age 0.5s; temporary IDs, no guaranteed identity or predicted output boxes',
                   'actual_object_detection_run':inference_count>0,'actual_px4_run':False,
                   'actual_flyvis_run':False,'control_authority':False,
                   'elapsed_wall_s':time.monotonic()-started,'model_sha256':args.sha256.lower(),
                   'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
                   'model_format':args.output_format,'preprocessing':'yolox_bgr_114',
                   'timing_note':'Offline sequential decode; no live-flight FPS or accuracy measurement'}
        json_write(args.output/'summary.json',summary)
        print(json.dumps(summary,indent=2))
        return 0 if successful else 2
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()


def watch(args):
    detector = make_detector(args)
    pipeline = PerceptionPipeline(detector,max_frame_age_s=args.max_age_s,
                                  appearance_tracking=not args.no_appearance_tracking)
    args.output.mkdir(parents=True,exist_ok=False)
    start = time.monotonic()
    latest_identity = None
    last_received = None
    last_notice = -math.inf
    count = 0
    outcome = 'duration_reached'
    with (args.output/'detections.jsonl').open('x') as log:
        try:
            while time.monotonic()-start < args.duration_s:
                path = args.spool/'latest.npz'
                result = None
                if path.is_file():
                    try:
                        st = path.stat()
                        identity = (st.st_ino,st.st_mtime_ns,st.st_size)
                        if identity != latest_identity:
                            sample = load_sample(path)
                            latest_identity = identity
                            result = pipeline.process(sample)
                            last_received = sample.received_monotonic_s
                            count += 1
                    except (ValueError,TypeError,OSError,EOFError,zipfile.BadZipFile) as exc:
                        if time.monotonic()-last_notice > 1:
                            result = {'status':'invalid_camera_packet','reason':str(exc),'control_authority':False}
                            last_notice = time.monotonic()
                now = time.monotonic()
                if result is None and (last_received is None or now-last_received > args.max_age_s) and now-last_notice >= 1:
                    pipeline.tracker.reset()
                    result = {'status':'camera_unavailable' if last_received is None else 'stale_camera_stream',
                              'inference_executed':False,'control_authority':False,
                              'depth_advisory':{'state':'unknown'},'detections':[]}
                    last_notice = now
                if result is not None:
                    result['input_mode'] = 'live_camera_spool'
                    log.write(json.dumps(result,allow_nan=False)+'\n')
                    log.flush()
                time.sleep(.01)
        except KeyboardInterrupt:
            outcome = 'interrupted'
    json_write(args.output/'summary.json',{'status':outcome,'packets_processed':count,
        'model_sha256':args.sha256.lower(),'control_authority':False,
        'scope':'Observation-only camera companion. This process does not verify PX4 flight or publish setpoints.'})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    doctor = sub.add_parser('doctor')
    doctor.add_argument('--output',type=Path)
    for name in ('image','video','watch'):
        p = sub.add_parser(name)
        p.add_argument('--model',type=Path,required=True)
        p.add_argument('--sha256',required=True)
        p.add_argument('--input-size',type=int,choices=[416,640],default=640)
        p.add_argument('--output-format',choices=['raw_yolox','decoded_cxcywh'],default='raw_yolox')
        p.add_argument('--confidence',type=float,default=.5,
                       help='Minimum object/class score (default: 0.5). A score is not a guarantee of correct classification.')
        p.add_argument('--no-appearance-tracking',action='store_true',
                       help='Use legacy geometry-only association. Default also checks person clothing color; IDs remain temporary.')
        p.add_argument('--max-age-s',type=float,default=.3 if name=='watch' else 30.)
        p.add_argument('--output',type=Path,required=True)
        if name == 'watch':
            p.add_argument('--spool',type=Path,required=True)
            p.add_argument('--duration-s',type=float,default=300.)
        else:
            p.add_argument('--input',type=Path,required=True)
            p.add_argument('--tile-size',type=int,default=0,
                           help='Optional crop size in source pixels; 960 retains more detail in 1080p video. 0 disables tiles. Offline only; costs extra inference passes.')
        if name == 'video':
            p.add_argument('--max-frames',type=int,default=3000)
            p.add_argument('--sample-every',type=int,default=1)
    args = parser.parse_args(argv)
    try:
        if args.command == 'doctor':
            status = package_status()
            if args.output:
                args.output.parent.mkdir(parents=True,exist_ok=True)
                json_write(args.output,status)
            print(json.dumps(status,indent=2))
            return 0 if status['object_detection_runtime_available'] else 2
        if args.command == 'watch':
            if not math.isfinite(args.duration_s) or not 0 < args.duration_s <= 86400:
                raise ValueError('duration must be within (0,86400] seconds')
            return watch(args)
        if args.command == 'video' and (not 1 <= args.max_frames <= 1_000_000 or not 1 <= args.sample_every <= args.max_frames):
            raise ValueError('invalid video frame limits')
        return offline(args)
    except (ImportError,OSError,ValueError,RuntimeError) as exc:
        print(f'Perception unavailable/failed: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
