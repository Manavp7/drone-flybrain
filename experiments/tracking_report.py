"""Summarize recorded detections; annotate only the source frames actually processed."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import unicodedata

import cv2
import numpy as np


ORIGINAL_VIDEO_SHA256 = "46d165858e41e3a49fcf30262cd4876f5f65f8f9be2c5560f98b74aef602ee03"
ORIGINAL_SUMMARY_SHA256 = "18d9ba5e6c6e3872524a172b2b83ebfd83189e03fe6a2a9de4c4c5a6501d524a"
ORIGINAL_LOG_SHA256 = "1b3bb5c8dccc94a95fd2576db82a1e7c56680f327671b3bdae752d73fd27744b"
ORIGINAL_PROVENANCE = {
    "sha256": ORIGINAL_VIDEO_SHA256,
    "title": "Secuencia 01.webm",
    "author": "Experienciausuario",
    "license": "CC0-1.0",
    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
    "source_url": "https://commons.wikimedia.org/wiki/File:Secuencia_01.webm",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_provenance(path, video_sha256):
    if path is None:
        if video_sha256 != ORIGINAL_VIDEO_SHA256:
            raise ValueError("--provenance is required for a new source video")
        provenance = dict(ORIGINAL_PROVENANCE)
    else:
        provenance = json.loads(Path(path).read_text())
    if provenance.get("sha256") != video_sha256:
        raise ValueError("Provenance hash does not match the source video")
    for key in ("title", "author", "license", "source_url"):
        if not isinstance(provenance.get(key), str) or not provenance[key].strip():
            raise ValueError(f"Provenance requires a nonempty {key}")
    return provenance


def sampling_metadata(rows, source, fallback_fps=None):
    """Frame counts refer to processed frames; sequences refer to source frames."""
    if not rows or len(rows) != source["frames"]:
        raise ValueError("Frame log and run summary disagree")
    sequences = [r["sequence"] for r in rows]
    times = np.asarray([r["capture_time_s"] for r in rows], dtype=float)
    if any(type(s) is not int or s < 0 for s in sequences) or not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("Frame sequences and capture times must be finite and nonnegative")
    declared_step = source.get("sample_every")
    if declared_step is not None and (type(declared_step) is not int or declared_step < 1):
        raise ValueError("Invalid recorded sample_every")
    if len(rows) > 1:
        steps = np.diff(sequences)
        if (steps <= 0).any() or not np.all(steps == steps[0]) or (np.diff(times) <= 0).any():
            raise ValueError("Report requires regularly sampled, ordered frame sequences and increasing times")
        step = int(steps[0])
        source_fps = float((sequences[-1]-sequences[0])/(times[-1]-times[0]))
        fps_basis = "source sequence difference divided by recorded capture-time difference"
    else:
        step = declared_step or 1
        source_fps = source.get("nominal_source_fps", fallback_fps)
        fps_basis = "recorded nominal_source_fps" if source.get("nominal_source_fps") is not None else "video decoder metadata"
    if declared_step is not None and declared_step != step:
        raise ValueError("Recorded sample_every disagrees with logged frame sequences")
    if source_fps is None or not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("A positive source frame rate is required to render the preview")
    return {"sample_every": step, "nominal_source_fps": float(source_fps),
            "preview_fps": float(source_fps)/step, "frame_rate_basis": fps_basis,
            "first_source_frame": sequences[0], "last_source_frame": sequences[-1],
            "frame_count_units": "observed_frames, longest_consecutive_frames and missing_frames_within_span count processed frames; first_frame and last_frame are source sequences"}


def summarize_tracks(rows):
    tracks = defaultdict(list)
    for processed_index, row in enumerate(rows):
        if len({d["track_id"] for d in row["detections"]}) != len(row["detections"]):
            raise ValueError("Duplicate track ID within a frame")
        for det in row["detections"]:
            tracks[det["track_id"]].append((processed_index, row["sequence"], row["capture_time_s"], det))
    track_rows = []
    for tid, observations in sorted(tracks.items()):
        best = current = 1
        best_span = 0.
        current_start = observations[0][2]
        for previous, following in zip(observations, observations[1:]):
            if following[0] == previous[0]+1:
                current += 1
            else:
                current, current_start = 1, following[2]
            span = following[2]-current_start
            if current > best or (current == best and span > best_span):
                best, best_span = current, span
        gaps = [b[0]-a[0]-1 for a,b in zip(observations, observations[1:])]
        labels = {o[3]["label"] for o in observations}
        class_ids = {o[3]["class_id"] for o in observations}
        if len(labels) != 1 or len(class_ids) != 1:
            raise ValueError("A track changed class")
        track_rows.append({"track_id": tid, "label": next(iter(labels)), "observed_frames": len(observations),
                           "first_frame": observations[0][1], "last_frame": observations[-1][1],
                           "first_processed_frame": observations[0][0], "last_processed_frame": observations[-1][0],
                           "first_time_s": observations[0][2], "last_time_s": observations[-1][2],
                           "span_s": observations[-1][2]-observations[0][2],
                           "longest_consecutive_frames": best, "longest_consecutive_span_s": best_span,
                           "gap_events": sum(g>0 for g in gaps), "missing_frames_within_span": sum(gaps),
                           "mean_confidence": float(np.mean([o[3]["confidence"] for o in observations]))})
    return track_rows


def run_metadata(source, summary_sha256, log_sha256, model_label=None):
    legacy = summary_sha256 == ORIGINAL_SUMMARY_SHA256 and log_sha256 == ORIGINAL_LOG_SHA256
    confidence = source.get("confidence_threshold")
    confidence_basis = "run summary" if confidence is not None else "unrecorded"
    if confidence is None and legacy:
        confidence, confidence_basis = .35, "verified original run identified by exact summary and log hashes"
    if confidence is not None and (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                                   or not math.isfinite(confidence) or not 0 <= confidence <= 1):
        raise ValueError("Invalid recorded confidence threshold")
    if model_label is None:
        filename = source.get("model_filename")
        model_label = Path(filename).stem if filename else ("YOLOX-S" if legacy else "Detector (model name unrecorded)")
    return {"model_label": model_label, "model_filename": source.get("model_filename"),
            "confidence_threshold": confidence, "confidence_threshold_basis": confidence_basis,
            "detector_forward_passes": source.get("detector_forward_passes"),
            "detector_counter_basis": source.get("detector_counter_basis", "unrecorded"),
            "tile_size": source.get("tile_size", 0 if legacy else None),
            "appearance_tracking": source.get("appearance_tracking", False if legacy else None),
            "max_age_s": source.get("max_age_s"), "tracker": source.get("tracker", "Tracker configuration unrecorded")}


def logged_source_frames(capture, rows):
    """Sequentially decode skipped frames without ever annotating them."""
    decoded_sequence = -1
    for row in rows:
        target = row["sequence"]
        if target <= decoded_sequence:
            raise ValueError("Logged frame sequences must increase")
        while decoded_sequence < target:
            ok, original = capture.read()
            if not ok:
                raise RuntimeError("Source ended before logged frame sequence")
            decoded_sequence += 1
        yield row, original


def put_line(canvas, text, origin, scale=.51, color=(190,206,211)):
    text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "replace").decode("ascii")
    width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    scale = min(scale, scale*1232/max(width,1))
    cv2.putText(canvas,text,origin,cv2.FONT_HERSHEY_SIMPLEX,scale,color,1,cv2.LINE_AA)


def render_preview(capture, rows, report, out):
    intermediate = out/"tracking_intermediate.mp4"
    writer = cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*"mp4v"),report["preview_fps"],(1280,840))
    if not writer.isOpened():
        raise RuntimeError("Video encoder unavailable")
    snapshots = set(np.linspace(0,len(rows)-1,min(6,len(rows)),dtype=int).tolist())
    stills = []
    colors = [(80,225,145),(90,200,255),(240,180,90),(190,130,245),(220,220,75),(175,220,245)]
    provenance = report["source_provenance"]
    confidence = report["confidence_threshold"]
    threshold_text = "unrecorded" if confidence is None else f"{confidence:g}"
    try:
        for i,(row,original) in enumerate(logged_source_frames(capture, rows)):
            h,w = original.shape[:2]
            frame = cv2.resize(original,(1280,720),interpolation=cv2.INTER_AREA)
            for det in row["detections"]:
                x1,y1,x2,y2 = det["bbox_xyxy"]
                if not 0<=x1<x2<=w or not 0<=y1<y2<=h:
                    raise ValueError("Detection box exceeds image bounds")
                x1,x2 = round(x1*1280/w),round(x2*1280/w)
                y1,y2 = round(y1*720/h),round(y2*720/h)
                color=colors[(det["track_id"]-1)%len(colors)]
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
                label=f"{det['label']} #{det['track_id']}  {det['confidence']:.2f}"
                (tw,th),_=cv2.getTextSize(label,cv2.FONT_HERSHEY_SIMPLEX,.48,1)
                lx=max(0,min(x1,1279-tw-8)); ly=max(th+8,y1-3)
                cv2.rectangle(frame,(lx,ly-th-6),(lx+tw+8,ly+3),color,-1)
                cv2.putText(frame,label,(lx+4,ly),cv2.FONT_HERSHEY_SIMPLEX,.48,(18,26,31),1,cv2.LINE_AA)
            canvas=np.full((840,1280,3),(35,25,16),dtype=np.uint8)
            canvas[72:792]=frame
            put_line(canvas,f"{report['model_label']} / OBJECT TRACKING | confidence threshold {threshold_text}",
                     (24,32),.8,(235,240,240))
            put_line(canvas,f"Video {row['capture_time_s']:.2f}s | Source frame {row['sequence']} | Processed {i+1}/{len(rows)} | Detections: {len(row['detections'])} | {row['status']}",(24,58))
            put_line(canvas,f"Source: {provenance['title']} | {provenance['author']} | {provenance['license']}",(24,812),.44)
            put_line(canvas,f"{provenance['source_url']} | Actual recorded boxes and temporary IDs; offline test",(24,833),.4)
            writer.write(canvas)
            if i in snapshots:
                stills.append(cv2.resize(canvas,(640,420)))
    finally:
        writer.release()
    if len(stills) % 2:
        stills.append(np.zeros_like(stills[-1]))
    contact=np.vstack([np.hstack(stills[i:i+2]) for i in range(0,len(stills),2)])
    if not cv2.imwrite(str(out/"tracking_contact_sheet.jpg"),contact):
        raise RuntimeError("Contact sheet could not be written")
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-i",str(intermediate),"-c:v","libx264",
                    "-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",str(out/"tracking_preview.mp4")],check=True)
    intermediate.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, help="Source/license JSON with the processed video's sha256; required for new videos")
    parser.add_argument("--model-label", help="Display label; defaults to the model filename recorded by the run")
    args = parser.parse_args(argv)
    run, out, video = args.run.resolve(), args.output.resolve(), args.video.resolve()
    source = json.loads((run / "summary.json").read_text())
    video_hash = sha(video)
    if video_hash != source["input_sha256"]:
        raise ValueError("Source video hash changed")
    provenance = load_provenance(args.provenance, video_hash)
    rows = [json.loads(line) for line in (run / "detections.jsonl").read_text().splitlines()]
    summary_hash, log_hash = sha(run/"summary.json"), sha(run/"detections.jsonl")
    metadata = run_metadata(source, summary_hash, log_hash, args.model_label)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError("Video decoder unavailable")
    try:
        sampling = sampling_metadata(rows, source, capture.get(cv2.CAP_PROP_FPS))
        track_rows = summarize_tracks(rows)
        statuses = Counter(r["status"] for r in rows)
        classes = Counter(d["label"] for r in rows for d in r["detections"])
        processing = np.asarray([r["processing_ms"] for r in rows])
        elapsed = source.get("elapsed_wall_s")
        report = {"status": "recorded_tracking_report", "source_run": str(run),
                  "frames": len(rows), "frame_status_counts": dict(statuses),
                  "inference_executed_frames": sum(r["inference_executed"] for r in rows),
                  "frames_with_detections": sum(bool(r["detections"]) for r in rows),
                  "frames_with_person_label": sum(any(d["label"] == "person" for d in r["detections"]) for r in rows),
                  "detection_observations_by_class": dict(classes),
                  "unique_track_ids": len(track_rows), "track_ids_are_not_unique_people": True,
                  "track_fragments_by_class": dict(Counter(r["label"] for r in track_rows)),
                  "max_simultaneous_detections": max(len(r["detections"]) for r in rows),
                  "processing_ms_p50_p95_max": [float(x) for x in np.percentile(processing,[50,95,100])],
                  "run_elapsed_wall_s": elapsed,
                  "effective_offline_processed_fps": len(rows)/elapsed if elapsed is not None and elapsed > 0 else None,
                  **sampling, **metadata, "tracks": track_rows,
                  "input_sha256": video_hash, "model_sha256": source["model_sha256"],
                  "source_log_sha256": log_hash, "source_summary_sha256": summary_hash,
                  "source_provenance": provenance,
                  "source_provenance_sha256": sha(args.provenance) if args.provenance else None,
                  "timing_note": source.get("timing_note", "Timestamp acquisition method unrecorded"),
                  "limitations": ["Track fragments are not a count of distinct people",
                                  "No ground-truth identity, recall or ID-switch benchmark",
                                  "Preview uses only processed source frames at a constant average playback rate",
                                  "Consecutiveness and missed-frame counts refer to processed frames, not skipped source frames",
                                  "Missing detections are not filled with fabricated boxes",
                                  "Detector/tracker outputs; this report does not demonstrate Flyvis or flight control"]}
        out.mkdir(parents=True, exist_ok=False)
        if track_rows:
            with (out/"tracks.csv").open("x", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(track_rows[0]))
                writer.writeheader(); writer.writerows(track_rows)
        render_preview(capture, rows, report, out)
    finally:
        capture.release()
    report["artifact_hashes"]={p.name:sha(p) for p in sorted(out.iterdir())}
    (out/"summary.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:report[k] for k in ["frames","frame_status_counts","unique_track_ids","detection_observations_by_class","effective_offline_processed_fps"]},indent=2))


if __name__=="__main__":
    main()
