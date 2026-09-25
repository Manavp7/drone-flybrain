"""Verify and replay frozen YOLO observations of the previously labelled person.

No detector, Flyvis model, flight controller or depth estimator is executed.
The selected temporary ID is fixed at 296. Evaluation holds only the latest
recorded source frame at or before each label, with a bounded media-time age.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from experiments.hybrid_target import TargetBridge
from experiments.tracking_report import logged_source_frames, load_provenance
from perception.detector import COCO_CLASSES
from perception.pipeline import iou


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACK_ID = 296
IMAGE_HW = (1080, 1920)
FIRST_SEQUENCE, LAST_SEQUENCE = 315, 360


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Artifact hash mismatch: {path}")
    return actual


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def load_inputs(root=PROJECT_ROOT):
    """Validate prior evidence hashes, including the independent annotations."""
    root = Path(root)
    lock = json.loads((root/"experiments/detection_repair_lock.json").read_text())
    annotation_lock = json.loads((root/"experiments/fly_tracking_v2_lock.json").read_text())
    relative = {
        "video": "inputs/sabana_grande/source.webm",
        "model": "models/yolox_s_official/yolox_s.onnx",
        "model_manifest": "models/yolox_s_official/yolox_s.manifest.json",
        "provenance": "inputs/sabana_grande/provenance.json",
        "timestamps": "inputs/sabana_grande/timestamps.json",
        "log": "results/sabana_detail_appearance01/detections.jsonl",
        "run_summary": "results/sabana_detail_appearance01/summary.json",
        "annotations": "results/fly_tracking_v2_annotations01/annotations.json",
    }
    hashes = {}
    for name, path in relative.items():
        if name == "video":
            expected = lock["input_sha256"]
        elif name == "model":
            expected = lock["model_sha256"]
        elif name == "annotations":
            expected = annotation_lock["files"][path]
        else:
            expected = lock["artifacts"][path]
        hashes[path] = require_hash(root/path, expected)
    rows = [json.loads(line) for line in (root/relative["log"]).read_text().splitlines()]
    source = json.loads((root/relative["run_summary"]).read_text())
    annotations = json.loads((root/relative["annotations"]).read_text())
    timestamps = [float(f["best_effort_timestamp_time"]) for f in
                  json.loads((root/relative["timestamps"]).read_text())["frames"]]
    if source["input_sha256"] != hashes[relative["video"]] or source["model_sha256"] != hashes[relative["model"]]:
        raise ValueError("Run summary does not identify the locked model and video")
    if annotations["source_sha256"] != hashes[relative["video"]]:
        raise ValueError("Annotations refer to another video")
    if len(rows) != source["frames"] or any(not np.isfinite(t) or t < 0 for t in timestamps) or any(
            b <= a for a, b in zip(timestamps, timestamps[1:])):
        raise ValueError("Invalid source frame count or presentation timestamps")
    validate_rows(rows, timestamps)
    return {"root": root, "paths": relative, "hashes": hashes, "rows": rows,
            "source": source, "annotations": annotations, "timestamps": timestamps,
            "provenance": load_provenance(root/relative["provenance"], hashes[relative["video"]])}


def validate_rows(rows, timestamps, image_hw=IMAGE_HW):
    """Reject stale ordering, duplicate IDs and label rewriting in replay input."""
    height, width = image_hw
    previous = -1
    for row in rows:
        sequence = row.get("sequence")
        if type(sequence) is not int or not previous < sequence < len(timestamps):
            raise ValueError("Recorded source sequences must increase and have source PTS")
        previous = sequence
        if not isinstance(row.get("detections"), list):
            raise ValueError("Recorded detections must be a list")
        ids = set()
        for detection in row["detections"]:
            tid, cid = detection.get("track_id"), detection.get("class_id")
            if type(tid) is not int or tid < 1 or tid in ids:
                raise ValueError("Recorded track IDs must be unique positive integers")
            ids.add(tid)
            if type(cid) is not int or not 0 <= cid < len(COCO_CLASSES) or detection.get("label") != COCO_CLASSES[cid]:
                raise ValueError("Recorded class labels must match actual COCO class IDs")
            box, score = detection.get("bbox_xyxy"), detection.get("confidence")
            if (not isinstance(box, list) or len(box) != 4
                    or not all(isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x) for x in box)
                    or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height
                    or not isinstance(score, (float, int)) or isinstance(score, bool) or not math.isfinite(score)
                    or not 0 <= score <= 1):
                raise ValueError("Recorded detection has invalid box or confidence")


def evaluate_replay(rows, annotations, timestamps, *, track_id=TRACK_ID, image_hw=IMAGE_HW, max_age_s=.25):
    """Score each noninitial label using the latest source row, never the future.

    Missing/failed latest rows stay missing; this does not reach farther into
    history for an old successful box. Processing latency is not simulated here.
    """
    validate_rows(rows, timestamps, image_hw)
    frames = annotations["frames"]
    initial_sequence = annotations["initial_sequence"]
    checks = []
    prior_annotation = -1
    bridge = TargetBridge(track_id)
    next_row = 0
    latest_row = latest_observation = None
    for annotation in frames:
        sequence = annotation["sequence"]
        if type(sequence) is not int or not prior_annotation < sequence < len(timestamps):
            raise ValueError("Annotations must be ordered source frame indices")
        prior_annotation = sequence
        if abs(timestamps[sequence]-annotation["timestamp_s"]) > 1e-6:
            raise ValueError("Annotation timestamp differs from exact source PTS")
        # Feed every intervening recorded row exactly once. The persistent
        # bridge therefore sees stream resets and losses even between labels.
        while next_row < len(rows) and rows[next_row]["sequence"] <= sequence:
            latest_row = rows[next_row]
            timed_row = {**latest_row, "capture_time_s": timestamps[latest_row["sequence"]]}
            latest_observation = bridge.update(timed_row, timestamps[latest_row["sequence"]],
                                               image_hw, max_age_s=max_age_s)
            next_row += 1
        if sequence <= initial_sequence or not annotation.get("scorable", False):
            continue
        gt = annotation["bbox_xyxy"]
        height, width = image_hw
        if (not isinstance(gt, list) or len(gt) != 4 or not all(math.isfinite(x) for x in gt)
                or not 0 <= gt[0] < gt[2] <= width or not 0 <= gt[1] < gt[3] <= height):
            raise ValueError("Annotation box must be on image")
        base = {"annotation_sequence": sequence, "annotation_pts_s": timestamps[sequence],
                "annotation_bbox_xyxy": gt, "initialization_excluded": True, "track_id": int(track_id)}
        if latest_row is None:
            checks.append({**base, "observed_sequence": None, "observed_pts_s": None,
                           "valid": False, "reason": "no_prior_observation", "bbox_xyxy": None,
                           "iou": 0., "active_iou_ge_0_5": False, "center_error_source_px": None})
            continue
        row = latest_row
        # Replay's explicit media-time copy preserves the original logged time
        # in the evidence row. It never changes the saved detector log.
        observation = latest_observation
        if timestamps[sequence]-timestamps[row["sequence"]] > max_age_s + 1e-9:
            observation = {**observation,"valid":False,"reason":"stale_target_frame","bbox_xyxy":None}
        overlap, error = 0., None
        if observation["valid"]:
            box = observation["bbox_xyxy"]
            overlap = float(iou(box, gt))
            error = float(math.hypot((box[0]+box[2]-gt[0]-gt[2])/2, (box[1]+box[3]-gt[1]-gt[3])/2))
        checks.append({**base, "observed_sequence": row["sequence"],
                       "observed_pts_s": timestamps[row["sequence"]],
                       "original_logged_capture_time_s": row["capture_time_s"],
                       "media_age_s": timestamps[sequence]-timestamps[row["sequence"]],
                       "valid": observation["valid"], "reason": observation["reason"],
                       "bbox_xyxy": observation["bbox_xyxy"], "iou": overlap,
                       "active_iou_ge_0_5": bool(observation["valid"] and overlap >= .5),
                       "center_error_source_px": error})
    errors = [r["center_error_source_px"] for r in checks if r["valid"]]
    stats = {"scored_noninitial_checks": len(checks), "valid_observation_checks": len(errors),
             "active_iou_ge_0_5_hits": sum(r["active_iou_ge_0_5"] for r in checks),
             "mean_iou_all_checks_loss_as_zero": float(np.mean([r["iou"] for r in checks])) if checks else None,
             "mean_center_error_source_px_valid_only": float(np.mean(errors)) if errors else None,
             "mean_center_error_391_retina_px_valid_only": float(np.mean(errors))*391/1080 if errors else None,
             "max_media_observation_age_s": max((r.get("media_age_s",0.) for r in checks), default=None)}
    return checks, stats


def render_replay(video, rows, timestamps, provenance, out, track_id=TRACK_ID):
    """Render only the 16 actually processed source frames315..360, all classes."""
    import cv2
    from experiments.tracking_report import put_line
    shown = [r for r in rows if FIRST_SEQUENCE <= r["sequence"] <= LAST_SEQUENCE]
    if len(shown) < 2:
        raise ValueError("At least two recorded rows are required for the preview")
    fps = (len(shown)-1)/(timestamps[shown[-1]["sequence"]]-timestamps[shown[0]["sequence"]])
    intermediate = out/"replay_intermediate.mp4"
    writer = cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*"mp4v"),fps,(1024,720))
    capture = cv2.VideoCapture(str(video))
    if not writer.isOpened() or not capture.isOpened():
        writer.release(); capture.release()
        raise RuntimeError("Replay video decoder or encoder unavailable")
    count = 0
    try:
        for row, original in logged_source_frames(capture, shown):
            frame = cv2.resize(original,(1024,576),interpolation=cv2.INTER_AREA)
            for detection in row["detections"]:
                selected = detection["track_id"] == track_id
                color = (100,245,90) if selected else (190,150,85)
                x0,y0,x1,y1 = [round(v*1024/1920) for v in detection["bbox_xyxy"]]
                cv2.rectangle(frame,(x0,y0),(x1,y1),color,3 if selected else 1)
                label = f"{detection['label']} #{detection['track_id']} {detection['confidence']:.2f}"
                cv2.putText(frame,label,(max(0,x0),max(13,y0-4)),cv2.FONT_HERSHEY_SIMPLEX,
                            .45 if selected else .25,color,1,cv2.LINE_AA)
            canvas = np.full((720,1024,3),(28,24,20),np.uint8)
            canvas[70:646] = frame
            put_line(canvas,f"SAVED YOLO TRACKING | selected person #{track_id}",(20,29),.7,(240,245,245))
            seen = any(d["track_id"] == track_id and d["class_id"] == 0 for d in row["detections"])
            put_line(canvas,f"Source frame {row['sequence']} | PTS {timestamps[row['sequence']]:.3f}s | selected {'observed' if seen else 'LOST'} | offline replay",(20,55),.47)
            put_line(canvas,"Actual saved boxes and temporary IDs | no new inference or flight execution",(20,670),.45)
            put_line(canvas,f"{provenance['author']} | {provenance['license']} | Sabana Grande Caracas",(20,692),.43)
            put_line(canvas,"Sampled, resized, muted and annotated; source URL and attribution in summary.json",(20,711),.4)
            writer.write(canvas)
            if count == len(shown)//2:
                if not cv2.imwrite(str(out/"replay_preview.jpg"),canvas):
                    raise RuntimeError("Could not save replay preview image")
            count += 1
    finally:
        writer.release(); capture.release()
    subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-i",str(intermediate),
                    "-c:v","libx264","-crf","21","-pix_fmt","yuv420p","-movflags","+faststart",
                    str(out/"tracking_replay.mp4")],check=True)
    intermediate.unlink()
    return {"frames": count, "width": 1024, "height": 720, "fps": fps,
            "first_source_sequence": shown[0]["sequence"], "last_source_sequence": shown[-1]["sequence"],
            "playback_timing": "constant average sampled source PTS rate; not inference throughput"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args(argv)
    out = args.output.resolve()
    if out.exists():
        raise FileExistsError(f"Output already exists: {out}")
    inputs = load_inputs()
    checks, stats = evaluate_replay(inputs["rows"],inputs["annotations"],inputs["timestamps"])
    out.mkdir(parents=True,exist_ok=False)
    write_json(out/"checks.json",checks)
    preview = render_replay(inputs["root"]/inputs["paths"]["video"],inputs["rows"],inputs["timestamps"],inputs["provenance"],out)
    original = inputs["source"]
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "verified_saved_detection_replay",
               "selected_track_id": TRACK_ID, "selection": "Explicit fixed previously identified temporary track; no automatic identity recognition",
               "new_detector_inference_executed": False, "flyvis_inference_executed": False,
               "flight_execution": False, "control_authority": False, "metric_depth_available": False,
               "source_run": inputs["paths"]["run_summary"], "source_provenance": inputs["provenance"],
               "input_sha256": inputs["hashes"][inputs["paths"]["video"]],
               "model_sha256": inputs["hashes"][inputs["paths"]["model"]],
               "verified_input_hashes": inputs["hashes"], "replay_source_sha256": sha256(__file__),
               "checks": stats, "preview": preview,
               "original_actual_detector_run": {"processed_frames": original["frames"],
                   "detector_forward_passes": original["detector_forward_passes"],
                   "elapsed_wall_s": original["elapsed_wall_s"],
                   "processed_fps": original["frames"]/original["elapsed_wall_s"],
                   "confidence_threshold": original["confidence_threshold"], "tile_size": original["tile_size"],
                   "appearance_tracking": original["appearance_tracking"]},
               "limitations": ["Approximate independent agent annotations, not a human benchmark; previously seen scene",
                   "Nine noninitial checks over about 1.5 seconds do not establish long-term identity tracking",
                   "Latest source frame is held for at most two frames here; actual offline detector latency is not replayed",
                   "Mean center error excludes invalid observations; overlap counts every scored check and assigns zero to loss",
                   "Recorded 2D footage cannot demonstrate closed-loop flight; no metric depth is inferred from boxes",
                   "All original recorded classes and boxes are retained in the video; selected person is highlighted"]}
    write_json(out/"summary.json",summary)
    (out/"source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(out/"artifact_hashes.json",{p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps({"output":str(out),"checks":stats,"preview":preview},indent=2,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
