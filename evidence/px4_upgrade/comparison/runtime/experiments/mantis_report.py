"""Export recorded Mantis episodes into an offline, causal replay lab.

This module does no model inference and advances no simulation. Camera frames,
network features and detection labels become visible only at recorded completion
time. Telemetry is sampled independently from the saved physics trace. Scientific
comparison scores and acceptance gates are displayed as supplied, never invented.
"""
from __future__ import annotations

import argparse
from bisect import bisect_right
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

FPS = 10
METHOD_NAMES = {'mantis_neural': 'Mantis Neural', 'direct_yolo': 'Direct YOLO',
                'alpha_beta': 'Alpha-beta'}
ROOT = Path(__file__).resolve().parents[1]

ACTOR_SOURCE = ('https://github.com/KhronosGroup/glTF-Sample-Assets/tree/'
                '90d7ede14c7e280af263824604b427a1ca02cb66/Models/CesiumMan')
ACTOR_LICENSE = 'https://creativecommons.org/licenses/by/4.0/'
ACTOR_CREDIT = f"""Mantis report and video character credit

Adapted Cesium Man, copyright 2017 Cesium.
License: Creative Commons Attribution 4.0 International (CC BY 4.0).
License URL: {ACTOR_LICENSE}
Source: {ACTOR_SOURCE}

Mantis adapts the character at runtime: coordinates are converted to Z-up,
initial-pose height is normalized to 1.8 m, the 19 animated joints are mapped
to MuJoCo mocap bones and the original mesh skin is blended. Fixed skeletal
influences assign plain blue shirt, dark trousers, brown skin and dark shoes.
Plain materials replace the source logo texture. Geometry and the original
two-second walking animation are retained. This is a stylized simulation
actor, not a scan of a real person. No endorsement by Cesium or Khronos is implied.

This credit accompanies every MP4 exported beside this file. Retain it when
sharing images or videos of the adapted character. Other scene elements and
project software have their own notices.
"""


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', value):
        raise ValueError('Case and method names must be plain identifiers')
    return value


def contained_path(folder, relative):
    """Saved frame paths may not escape their own episode, even via symlinks."""
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError('A frame requires an episode-relative path')
    folder = Path(folder).resolve()
    result = (folder / relative).resolve()
    if result == folder or not result.is_relative_to(folder):
        raise ValueError('Saved frame path escapes its episode')
    return result


def script_json(value):
    """Serialize inert application/json without permitting an HTML end tag."""
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False, allow_nan=False).replace(
        '&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e').replace(
        '\u2028', '\\u2028').replace('\u2029', '\\u2029')


def latest_available(observations, time_s):
    eligible = [row for row in observations if not row.get('discarded_after_episode', False)
                and row['completed_time_s'] <= time_s + 1e-9]
    return max(eligible, key=lambda row: row['completed_time_s']) if eligible else None


def validate_records(spec, ticks, observations, episode):
    duration = spec.get('duration_s')
    if not finite(duration) or duration <= 0 or abs(duration*FPS-round(duration*FPS)) > 1e-7:
        raise ValueError('Replay duration must be a positive multiple of 0.1 seconds')
    if not ticks or episode.get('status') != 'completed':
        raise ValueError('Replay requires a completed, nonempty recorded episode')
    if episode.get('physics_ticks') != len(ticks) or episode.get('observations') != len(observations):
        raise ValueError('Episode receipt disagrees with recorded row counts')
    last_time = -1.
    for tick in ticks:
        t = tick.get('time_s')
        state = tick.get('state_before', {})
        after = tick.get('state_after', {})
        if (not finite(t) or t <= last_time or not finite(state.get('time_s'))
                or abs(state['time_s']-t) > 1e-7 or not finite(after.get('time_s'))
                or after['time_s'] <= t):
            raise ValueError('Physics telemetry must have increasing, consistent clocks')
        for item in (state, after):
            for key, shape in (('position', (3,)), ('velocity', (3,)), ('motor_forces', (4,))):
                value = np.asarray(item.get(key), dtype=float)
                if value.shape != shape or not np.isfinite(value).all():
                    raise ValueError('Physics telemetry is missing finite position, velocity or motor forces')
        last_time = t
    if abs(ticks[0]['time_s']) > 1e-7 or abs(ticks[-1]['state_after']['time_s']-duration) > 1e-7:
        raise ValueError('Physics trace does not span the declared duration')
    previous = -1.
    for index, row in enumerate(observations):
        capture, complete = row.get('capture_time_s'), row.get('completed_time_s')
        if (row.get('sequence') != index or not finite(capture) or not finite(complete)
                or capture < 0 or capture >= duration or complete <= capture or complete <= previous):
            raise ValueError('Observation sequence or clocks are invalid')
        if complete > duration + 1e-8 and row.get('discarded_after_episode') is not True:
            raise ValueError('An after-episode observation must be marked discarded')
        overview = row.get('overview_capture_time_s', capture)
        if not finite(overview) or overview < 0 or overview > complete + 1e-8:
            raise ValueError('Overview capture must precede result availability')
        previous = complete


def _features(row, folder=None):
    if row is None:
        return None
    features = row.get('neural_features')
    if features is None and folder is not None:
        with np.load(contained_path(folder, row['frame_file']), allow_pickle=False) as saved:
            features = saved['features'] if 'features' in saved else None
    if features is None:
        return None
    array = np.asarray(features, dtype=float)
    if array.shape != (8,) or not np.isfinite(array).all():
        raise ValueError('Saved neural features must contain exactly eight finite values')
    return array.tolist()


def build_timeline(spec, ticks, observations, folder=None):
    """Match the 10 Hz video clock, preserving pre-completion waiting periods."""
    times = [tick['time_s'] for tick in ticks]
    feature_cache = {}
    timeline = []
    for index in range(round(spec['duration_s']*FPS)+1):
        t = index/FPS
        tick = ticks[max(0, bisect_right(times, t+1e-9)-1)]
        state = tick['state_after'] if t >= tick['state_after']['time_s']-1e-9 else tick['state_before']
        # The video holds its final encoded sample at the end; keep its evidence
        # in sync while telemetry can still show the final physics state.
        frame_time = min(t, max(0., spec['duration_s']-1/FPS))
        row = latest_available(observations, frame_time)
        sequence = row['sequence'] if row else None
        if sequence not in feature_cache:
            feature_cache[sequence] = _features(row, folder)
        request, guardian = tick.get('request', {}), tick.get('guardian', {})
        obs = row.get('observation', {}) if row else {}
        candidate = (row.get('candidate') or {}) if row else {}
        speed = float(np.linalg.norm(state['velocity']))
        timeline.append(dict(time_s=t, state_time_s=state['time_s'], position=state['position'],
            speed_m_s=speed, altitude_m=state['position'][2], motor_forces_n=state['motor_forces'],
            requested_speed_m_s=request.get('forward_speed'), authorized_speed_m_s=guardian.get('forward_speed'),
            guardian_reason=guardian.get('reason'), guidance_reason=request.get('reason'),
            applied_command_sequence=tick.get('applied_command_sequence'),
            candidate_valid=candidate.get('valid'), sequence=sequence,
            capture_time_s=row['capture_time_s'] if row else None,
            completed_time_s=row['completed_time_s'] if row else None,
            overview_capture_time_s=row.get('overview_capture_time_s', row['capture_time_s']) if row else None,
            inference_wall_s=row.get('inference_wall_s') if row else None,
            target_valid=obs.get('valid') is True, target_id=obs.get('track_id'),
            features=feature_cache[sequence]))
    return timeline


def summarize_episode(spec, ticks, observations, episode):
    valid = sum(row.get('observation', {}).get('valid') is True for row in observations)
    start, end = np.asarray(ticks[0]['state_before']['position']), np.asarray(ticks[-1]['state_after']['position'])
    return dict(duration_s=spec['duration_s'], observation_count=len(observations), valid_observation_count=valid,
        valid_observation_fraction=valid/len(observations) if observations else None,
        horizontal_translation_m=float(np.linalg.norm(end[:2]-start[:2])),
        whole_wall_s=episode.get('whole_wall_s'), physics_ticks=len(ticks),
        contact_count=(sum(tick['truth_after']['contact_count'] for tick in ticks) if all(
            isinstance(tick.get('truth_after', {}).get('contact_count'), int)
            and not isinstance(tick['truth_after']['contact_count'], bool)
            and tick['truth_after']['contact_count'] >= 0 for tick in ticks) else None),
        discarded_observation_count=sum(row.get('discarded_after_episode') is True for row in observations))


def normalize_comparisons(entries):
    """Display the scorer's exact matched subset; never pool or invent a win."""
    if not isinstance(entries, list):
        raise ValueError('comparison.json must contain a list of recorded comparisons')
    output = []
    for entry in entries:
        score, result = entry['score'], entry['result']
        count = score['recorded_count']
        rows = result['rows']
        if count != len(rows) or count <= 0:
            raise ValueError('Comparison score disagrees with recorded observation coverage')
        metrics = []
        for method, label in METHOD_NAMES.items():
            saved = score['methods'][method]
            rmse = saved['matched_capture_rmse_rad']
            estimator = result['timing']['per_method_wall_s'][method]
            metrics.append(dict(method=method, label=label,
                rmse_deg=math.degrees(rmse) if rmse is not None else None,
                truth_prediction_count=saved['truth_prediction_count'],
                full_truth_coverage=saved['full_truth_coverage'],
                eligible_truth_coverage=saved['eligible_truth_coverage'],
                estimator_ms_per_observation=estimator*1000/count))
        source = entry.get('source_method', entry.get('source_camera_path_method'))
        output.append(dict(case=entry['case'], variant=entry['variant'], seed=entry['seed'],
            source_method=source, source_label=METHOD_NAMES.get(source, source or 'Not recorded'),
            recorded_count=count, truth_available_count=score['truth_available_count'],
            shared_eligible_count=score['shared_eligible_count'], matched_count=score['matched_count'],
            truth_unavailable_count=score['truth_unavailable_count'],
            shared_missing_reasons=score['shared_missing_reasons'], metrics=metrics,
            detector_ms_per_observation=result['timing']['detector_wall_s']*1000/count,
            neural_reset_ms=result['timing']['neural_reset_wall_s']*1000,
            score_clock=score.get('score_clock', 'capture_time_s'),
            completion_time_accuracy_evaluated=score.get('completion_time_accuracy_evaluated', False)))
    return output


def load_recordings(run):
    root = Path(run).resolve()
    definition = read_json(root/'definition.json')
    methods = definition.get('methods')
    if not isinstance(methods, list) or not methods or len(methods) != len(set(methods)):
        raise ValueError('Definition must list distinct saved methods')
    if any(method not in METHOD_NAMES for method in methods):
        raise ValueError('Unknown recorded controller')
    specs = definition.get('specs')
    if not isinstance(specs, list) or not specs:
        raise ValueError('Definition must list planned cases')
    cases, recordings, seen = [], [], set()
    for planned in specs:
        name = safe_name(planned['name'])
        if name in seen:
            raise ValueError('Duplicate planned case')
        seen.add(name)
        case = dict(name=name, label=name.replace('_', ' ').capitalize(), methods={})
        for method in methods:
            folder = contained_path(root, f'{name}__{method}')
            spec, episode = read_json(folder/'spec.json'), read_json(folder/'episode.json')
            if spec.get('name') != name or spec.get('method') != method:
                raise ValueError('Saved case or controller does not match the frozen plan')
            if any(spec.get(key) != value for key, value in planned.items()):
                raise ValueError('Saved case differs from its frozen specification')
            ticks, observations = read_rows(folder/'ticks.jsonl'), read_rows(folder/'observations.jsonl')
            validate_records(spec, ticks, observations, episode)
            for row in observations:
                contained_path(folder, row['frame_file'])
            video = f'{name}__{method}.mp4'
            timeline = build_timeline(spec, ticks, observations, folder)
            case['methods'][method] = dict(label=METHOD_NAMES[method], video=video, timeline=timeline,
                summary=summarize_episode(spec, ticks, observations, episode))
            recordings.append((folder, spec, ticks, observations, video))
        cases.append(case)
    comparisons = normalize_comparisons(read_json(root/'comparison.json')) if (root/'comparison.json').is_file() else []
    checks = read_json(root/'checks.json') if (root/'checks.json').is_file() else None
    payload = dict(version=1, title='Mantis', fps=FPS, methods=methods, cases=cases,
        comparisons=comparisons, checks=checks, frozen_utc=definition.get('frozen_utc'),
        source_representation=definition.get('source_representation'),
        mode='Recorded offline simulation', real_time=False, physical_flight=False)
    return payload, recordings


def _draw_text(canvas, text, xy, scale=.48, color=(188, 200, 184), weight=1):
    import cv2
    cv2.putText(canvas, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)


def _fit(canvas, rgb, x, y, width, height):
    import cv2
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('Saved camera views must be RGB uint8 arrays')
    h, w = rgb.shape[:2]
    scale = min(width/w, height/h)
    image = cv2.resize(rgb[:, :, ::-1], (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
    dx, dy = x+(width-image.shape[1])//2, y+(height-image.shape[0])//2
    canvas[dy:dy+image.shape[0], dx:dx+image.shape[1]] = image
    return scale, dx, dy


def render_frame(folder, observations, time_s, cache=None):
    """Both recorded views wait until their observation has completed."""
    import cv2
    cache = {} if cache is None else cache
    canvas = np.full((640, 1280, 3), (24, 31, 25), dtype=np.uint8)
    _draw_text(canvas, '01  /  BODY CAMERA', (22, 30), .55)
    _draw_text(canvas, '02  /  RECORDED OVERVIEW', (665, 30), .55)
    cv2.line(canvas, (641, 12), (641, 615), (61, 73, 61), 1)
    _draw_text(canvas, 'Adapted Cesium Man (c) 2017 Cesium | CC BY 4.0 | Full source and changes: CREDITS.txt',
               (22, 634), .34, (151, 167, 146))
    row = latest_available(observations, time_s)
    if row is None:
        _draw_text(canvas, 'AWAITING FIRST COMPLETED OBSERVATION', (104, 303), .57)
        _draw_text(canvas, 'Camera, detection and neural evidence remain unavailable.', (104, 334), .43)
        _draw_text(canvas, 'The physics trace continues during inference.', (688, 322), .52)
        return canvas
    if cache.get('sequence') != row['sequence']:
        with np.load(contained_path(folder, row['frame_file']), allow_pickle=False) as saved:
            cache.clear()
            cache.update(sequence=row['sequence'], rgb=saved['rgb'].copy(), overview=saved['overview'].copy())
    scale, x, y = _fit(canvas, cache['rgb'], 16, 47, 608, 537)
    _fit(canvas, cache['overview'], 660, 47, 604, 537)
    observation = row.get('observation', {})
    box = observation.get('bbox_xyxy')
    if observation.get('valid') is True and box is not None:
        box = np.asarray(box, dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError('Invalid selected detection box')
        left, top, right, bottom = (box*scale+np.array([x, y, x, y])).round().astype(int)
        color = (86, 242, 185)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        _draw_text(canvas, f'PERSON  {observation.get("track_id", "?")}', (left, max(62, top-8)), .47, color)
    capture, completion = row['capture_time_s'], row['completed_time_s']
    overview = row.get('overview_capture_time_s', capture)
    _draw_text(canvas, f'CAPTURE {capture:.3f} s    AVAILABLE {completion:.3f} s', (22, 611), .48)
    _draw_text(canvas, f'OVERVIEW CAPTURE {overview:.3f} s    HELD RECORDED FRAME', (665, 611), .46)
    return canvas


def encode_video(folder, spec, observations, output):
    import cv2
    if shutil.which('ffmpeg') is None:
        raise RuntimeError('ffmpeg is required to export the offline replay')
    cv2.setNumThreads(2)
    args = ['ffmpeg', '-v', 'error', '-nostdin', '-n', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', '1280x640', '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264',
            '-threads', '2', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', '-metadata', 'comment='+ACTOR_CREDIT, str(output)]
    process = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    cache = {}
    try:
        for index in range(round(spec['duration_s']*FPS)):
            process.stdin.write(render_frame(folder, observations, index/FPS, cache).tobytes())
        process.stdin.close()
        process.stdin = None
        _, error = process.communicate(timeout=120)
        if process.returncode:
            raise RuntimeError('Video export failed: '+error.decode(errors='replace'))
    except BaseException:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
            process.stdin = None
        if process.poll() is None:
            process.kill()
        process.communicate()
        raise
    capture = cv2.VideoCapture(str(output))
    decoded = 0
    try:
        while capture.read()[0]:
            decoded += 1
    finally:
        capture.release()
    if decoded != round(spec['duration_s']*FPS):
        raise RuntimeError('Exported video frame count does not match the recorded duration')
    return dict(file=output.name, frames=decoded, fps=FPS, duration_s=decoded/FPS)


def export_report(run):
    root = Path(run).resolve()
    output = root/'report'
    if output.exists() or output.is_symlink():
        raise FileExistsError('A report already exists; export only into a new report directory')
    payload, recordings = load_recordings(root)
    template = (ROOT/'assets/mantis_lab.html').read_text()
    if template.count('__MANTIS_DATA__') != 1:
        raise ValueError('Replay template must have one data placeholder')
    # Validate the complete data before creating any report output.
    serialized = script_json(payload)
    output.mkdir()
    videos = []
    try:
        (output/'CREDITS.txt').write_text(ACTOR_CREDIT)
        for folder, spec, _, observations, video in recordings:
            videos.append(encode_video(folder, spec, observations, output/video))
        (output/'index.html').write_text(template.replace('__MANTIS_DATA__', serialized))
        (output/'export.json').write_text(json.dumps(dict(videos=videos, mode='saved-data-only',
            future_observations_displayed=False, physical_flight=False), indent=2)+'\n')
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps(dict(type=type(exc).__name__, reason=str(exc)), indent=2)+'\n')
        raise
    return output/'index.html'


class ReplayRequestHandler(SimpleHTTPRequestHandler):
    """Serve immutable exported files with the byte ranges required by video."""
    def send_head(self):
        self._remaining = None
        header = self.headers.get('Range')
        path = Path(self.translate_path(self.path))
        if not header or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
        valid = bool(match and (match[1] or match[2]) and size > 0)
        start, end = 0, size-1
        if valid:
            first, last = match.groups()
            if first:
                start = int(first)
                end = min(int(last), size-1) if last else size-1
            else:
                suffix = int(last)
                valid = suffix > 0
                start = max(0, size-suffix)
            valid = valid and 0 <= start <= end < size
        if not valid:
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        source = path.open('rb')
        source.seek(start)
        self._remaining = end-start+1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Length', str(self._remaining))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        return source

    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def copyfile(self, source, outputfile):
        if self._remaining is None:
            return super().copyfile(source, outputfile)
        remaining = self._remaining
        while remaining:
            block = source.read(min(64*1024, remaining))
            if not block:
                break
            outputfile.write(block)
            remaining -= len(block)


def serve_report(report, port=8874):
    report = Path(report).resolve()
    if not (report/'index.html').is_file():
        raise ValueError('Serve an exported report directory containing index.html')
    if not 0 <= port <= 65535:
        raise ValueError('Port must be in 0..65535')
    handler = partial(ReplayRequestHandler, directory=str(report))
    with ThreadingHTTPServer(('127.0.0.1', port), handler) as server:
        print(f'Mantis replay: http://127.0.0.1:{server.server_port}/', flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    parser = argparse.ArgumentParser(__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--run', type=Path, help='Export a saved simulator run once')
    action.add_argument('--serve', type=Path, help='Serve an exported report locally with video seeking')
    parser.add_argument('--port', type=int, default=8874)
    args = parser.parse_args()
    if args.serve is not None:
        serve_report(args.serve, args.port)
    else:
        print(export_report(args.run))


if __name__ == '__main__':
    main()
