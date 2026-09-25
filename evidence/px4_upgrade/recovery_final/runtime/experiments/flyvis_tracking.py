"""Offline, manually initialized tracking from real frozen Flyvis responses.

Run: .venv/bin/python -m experiments.flyvis_tracking --output results/fly_tracking_run01
No detector, annotation correction, or conventional flow enters the neural arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

import cv2
import numpy as np

from experiments.fly_tracking_core import (FlowBoxTracker, evaluate_tracking, fit_affine_flow,
    pool_flow, raw_flow_mapping, visible_response_indices)
from experiments.fly_tracking_stimuli import (CALIBRATION, TEST_SEEDS, DT, SIDE,
    experiment_definition, real_clip, synthetic_clip, source_to_retina)
from experiments.runtime import VideoFlyvisAdapter
from flybrain_sim.research_model import validate_clip

PROJECT = Path(__file__).resolve().parents[1]


def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def infer(adapter, clip, directory, reuse_source=None):
    directory.mkdir(exist_ok=False)
    sampling = validate_clip(clip["frames"], clip["times"], DT)
    if reuse_source is not None:
        previous = reuse_source / clip["name"]
        summary = json.loads((previous/"neural_receipt.json").read_text())
        if (hashlib.sha256(clip["frames"].tobytes()).hexdigest() != summary["input_sha256"] or
                hashlib.sha256(np.asarray(clip["times"], dtype=np.float64).tobytes()).hexdigest() != summary["timestamps_sha256"]):
            raise ValueError("Replay input differs from recorded neural input")
        flow = np.load(previous/"decoded_flow.npy", allow_pickle=False)
        if flow.shape != (len(sampling.indices),2,721) or not np.isfinite(flow).all():
            raise ValueError("Invalid saved neural flow")
        save_json(directory/"neural_source.json", {"reuse": True, "source": str(previous.resolve()),
                  "decoded_flow_sha256": digest(previous/"decoded_flow.npy"),
                  "neural_receipt_sha256": digest(previous/"neural_receipt.json"),
                  "new_neural_inference_executed": False})
        return flow, sampling
    np.savez_compressed(directory / "input.npz", frames=clip["frames"], timestamps=clip["times"])
    summary, activity = adapter.infer_clip(clip["frames"], clip["times"], dt=DT)
    np.save(directory / "activity.npy", activity)
    flow = np.concatenate([adapter.decode_flow(activity[i:i+32]) for i in range(0, len(activity), 32)])
    np.save(directory / "decoded_flow.npy", flow)
    del activity
    save_json(directory / "neural_receipt.json", summary)
    print(json.dumps({"case": clip["name"], "inference_executed": True,
                      "steps": len(flow), "neural_wall_s": summary["elapsed_wall_s"]}), flush=True)
    return flow, sampling


def track_fields(fields, centers, initial, mapping):
    tracker = FlowBoxTracker(centers, initial, mapping)
    records = [tracker.step(field, DT) for field in fields]
    return np.asarray([r["box_xyxy"] for r in records]), records


def conventional_flow(clip, sampling, initial):
    """Independent Farneback baseline; never supplies input to Flyvis tracker."""
    frames = np.round(clip["frames"] * 255).astype(np.uint8)
    box = np.array(initial, dtype=float)
    boxes, statuses, status = [], [], "tracking"
    prev = frames[sampling.indices[0]]
    for i in sampling.indices:
        current = frames[i]
        if status == "tracking":
            flow = cv2.calcOpticalFlowFarneback(prev, current, None, .5, 3, 21, 3, 5, 1.2, 0)
            x1, y1, x2, y2 = np.rint(box).astype(int)
            area = flow[max(0,y1):min(SIDE,y2), max(0,x1):min(SIDE,x2)]
            if not area.size:
                status = "insufficient_support"
            else:
                movement = np.median(area.reshape(-1, 2), axis=0)
                candidate = box + movement[[0, 1, 0, 1]]
                if np.min(candidate[:2]) < 0 or np.max(candidate[2:]) > SIDE:
                    status = "outside_frame"
                else:
                    box = candidate
        boxes.append(box.copy())
        statuses.append(status)
        prev = current
    return np.asarray(boxes), statuses


def display_boxes(boxes, response_times, display_times, initial):
    indices = visible_response_indices(response_times, display_times)
    return np.asarray([initial if i < 0 else boxes[i] for i in indices])


def display_statuses(statuses, response_times, display_times):
    indices = visible_response_indices(response_times, display_times)
    return np.asarray(["initialized" if i < 0 else statuses[i] for i in indices])


def run_case(adapter, clip, centers, mapping, output, reuse_source=None):
    directory = output / clip["name"]
    fields, sampling = infer(adapter, clip, directory, reuse_source)
    initial = clip.get("initial_box", clip.get("truth_boxes", [None])[0])
    permutation = np.random.default_rng(731).permutation(len(centers))
    arms, traces, statuses = {}, {}, {}
    for name, data, transform in [
        ("flyvis_calibrated", fields, mapping),
        ("flyvis_raw", fields, raw_flow_mapping()),
        ("zero_motion", np.zeros_like(fields), np.zeros((3,2))),
        ("zero_neural", np.zeros_like(fields), mapping),
        ("spatial_shuffle", fields[:, :, permutation], mapping),
    ]:
        boxes, records = track_fields(data, centers, initial, transform)
        arms[name] = display_boxes(boxes, sampling.response_timestamps, clip["times"], initial)
        statuses[name] = display_statuses([r["status"] for r in records], sampling.response_timestamps, clip["times"])
        for i, record in enumerate(records):
            record.update(stimulus_time_s=float(sampling.stimulus_timestamps[i]),
                          response_time_s=float(sampling.response_timestamps[i]),
                          source_input_index=int(sampling.indices[i]))
        traces[name] = records
    conventional, conventional_status = conventional_flow(clip, sampling, initial)
    arms["farneback"] = display_boxes(conventional, sampling.response_timestamps, clip["times"], initial)
    statuses["farneback"] = display_statuses(conventional_status, sampling.response_timestamps, clip["times"])
    np.savez_compressed(directory / "display_boxes.npz", timestamps=clip["times"], **arms)
    save_json(directory / "traces.json", traces)
    save_json(directory / "display_statuses.json", {k:v.tolist() for k,v in statuses.items()})
    if clip["kind"] == "synthetic":
        truth = clip["truth_boxes"]
        mask = clip["times"] >= .8
        metrics = {name: evaluate_tracking(boxes[mask], truth[mask], statuses[name][mask]=="tracking") for name, boxes in arms.items()}
        save_json(directory / "truth.json", {"boxes_xyxy": truth.tolist(), "seed": clip["seed"],
                  "timestamps": clip["times"].tolist(), "score_start_s": .8})
        main, static = metrics["flyvis_calibrated"], metrics["zero_motion"]
        improvement = 1 - main["mean_center_error"] / static["mean_center_error"]
        passed = main["fraction_active_iou_ge_0_5"] >= .75 and improvement >= .20
    else:
        annotations = json.loads((PROJECT / "results/fly_tracking_annotations01/annotations.json").read_text())
        usable = [r for r in annotations["frames"] if r.get("scorable", True) and r["sequence"] > 160]
        indices = np.array([r["sequence"] - 160 for r in usable])
        truth = source_to_retina([r["bbox_xyxy"] for r in usable])
        metrics = {name: evaluate_tracking(boxes[indices], truth, statuses[name][indices]=="tracking") for name, boxes in arms.items()}
        save_json(directory / "annotations_used.json", annotations)
        improvement, passed = None, None
    result = {"case": clip["name"], "kind": clip["kind"], "metrics_retina_pixels": metrics,
              "neural_tracking_acceptance_pass": passed,
              "center_error_reduction_vs_static": improvement,
              "score_policy": "Initialization excluded; lost outputs cannot earn tracking hits. Geometric errors include retained last boxes.",
              "lost_steps": {name: sum(r["status"] != "tracking" for r in rows) for name, rows in traces.items()},
              "no_neural_reference_corrections": True}
    save_json(directory / "metrics.json", result)
    print(json.dumps(result), flush=True)
    return arms, statuses, result


def render_case(writer, clip, arms, statuses, result, stills):
    """Uniform 50FPS presentation; all displayed boxes were causally selected."""
    source_times = clip["times"]
    selected = np.searchsorted(source_times, np.arange(source_times[0], source_times[-1], .02), side="right") - 1
    initial = clip.get("initial_box", clip.get("truth_boxes", [None])[0])
    chosen = set(np.linspace(0, len(selected)-1, 4).round().astype(int))
    for ordinal, i in enumerate(selected):
        source = clip.get("color")
        img = source[i].copy() if source is not None else cv2.cvtColor(np.round(clip["frames"][i]*255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        canvas = np.full((640, 960, 3), (24, 20, 18), np.uint8)
        scale = 520 / SIDE
        visual = cv2.resize(img, (520, 520))
        colors = {"flyvis_calibrated": (255, 224, 70), "farneback": (130, 230, 130), "zero_motion": (110, 110, 240)}
        for name in ("zero_motion", "farneback", "flyvis_calibrated"):
            box = (arms[name][i] * scale).round().astype(int)
            cv2.rectangle(visual, tuple(box[:2]), tuple(box[2:]), colors[name], 2)
        neural_status = statuses["flyvis_calibrated"][i]
        if neural_status not in ("tracking", "initialized"):
            cv2.putText(visual, "LOST - last position", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        .65, colors["flyvis_calibrated"], 2, cv2.LINE_AA)
        if "truth_boxes" in clip:
            box = (clip["truth_boxes"][i] * scale).round().astype(int)
            cv2.rectangle(visual, tuple(box[:2]), tuple(box[2:]), (255,255,255), 1)
        canvas[55:575, 20:540] = visual
        lines = [(clip["name"].replace('_',' ').upper(), (245,245,245)),
                 ("Manually initialized target", (220,220,220)),
                 (f"Video time {source_times[i]-source_times[0]:.2f}s", (190,190,190)),
                 ("FLYVIS: " + neural_status, colors["flyvis_calibrated"]),
                 ("Conventional optical flow", colors["farneback"]),
                 ("Unchanged initial box", colors["zero_motion"]),
                 ("No YOLO corrections", (210,210,210)),
                 ("No person recognition", (210,210,210)),
                 ("45,669 frozen model neurons", (210,210,210))]
        main = result["metrics_retina_pixels"]["flyvis_calibrated"]
        lines.extend([(f"Mean center error: {main['mean_center_error']:.1f}px",(235,235,235)),
                      (f"Active IoU >= 0.5: {100*main['fraction_active_iou_ge_0_5']:.0f}%",(235,235,235))])
        for n, (text, color) in enumerate(lines):
            cv2.putText(canvas, text, (560, 80+n*38), cv2.FONT_HERSHEY_SIMPLEX, .57, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, "Actual saved neural trajectory | offline experiment; playback is not inference speed", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, .52, (220,220,220), 1, cv2.LINE_AA)
        credit = "Synthetic texture and known trajectory; white = ground truth" if clip["kind"]=="synthetic" else "Video: Vicente Quintero / QuinteroP | CC BY 3.0 | cropped, muted and annotated"
        cv2.putText(canvas, credit, (20, 606), cv2.FONT_HERSHEY_SIMPLEX, .5, (180,180,180), 1, cv2.LINE_AA)
        writer.write(canvas)
        if ordinal in chosen:
            stills.append(canvas)


def run(output, reuse_source=None):
    if reuse_source is not None:
        # Verify all immutable original artifacts before reading any saved predictions.
        hashes = json.loads((reuse_source/"artifact_hashes.json").read_text())
        for relative, expected in hashes.items():
            if digest(reuse_source/relative) != expected:
                raise ValueError(f"Prior artifact hash mismatch: {relative}")
    output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(2)
    started = time.perf_counter()
    definition = experiment_definition()
    definition["created_utc"] = datetime.now(timezone.utc).isoformat()
    definition["reuse_neural_run"] = str(reuse_source.resolve()) if reuse_source is not None else None
    definition["report_revision"] = "v2: initialization excluded, loss-aware success and visible lost states"
    definition["source_sha256"] = {str(p.relative_to(PROJECT)): digest(p) for p in [
        Path(__file__), PROJECT/"experiments/fly_tracking_core.py", PROJECT/"experiments/fly_tracking_stimuli.py"]}
    save_json(output/"experiment_definition.json", definition)
    # Select and verify source data before neural evaluation, not based on results.
    real = real_clip(PROJECT)
    adapter = None if reuse_source is not None else VideoFlyvisAdapter(PROJECT/"models/flyvis_0000_000.manifest.json")
    centers = (np.load(reuse_source/"receptor_centers_rc.npy", allow_pickle=False) if reuse_source is not None
               else adapter.eye.receptor_centers.cpu().numpy() + [195,195])
    np.save(output/"receptor_centers_rc.npy", centers)
    pooled, velocities = [], []
    for name in ([] if reuse_source is not None else CALIBRATION):
        clip = synthetic_clip(name)
        flow, sampling = infer(adapter, clip, output/f"calibration_{name}")
        for step, index in enumerate(sampling.indices):
            if sampling.stimulus_timestamps[step] < .4:
                continue
            value, _ = pool_flow(flow[step], centers, clip["truth_boxes"][index])
            pooled.append(value)
            velocities.append(clip["velocities"][index])
        save_json(output/f"calibration_{name}"/"truth.json", {
            "boxes_xyxy": clip["truth_boxes"].tolist(), "velocities_xy": clip["velocities"].tolist(),
            "timestamps": clip["times"].tolist(), "seed": clip["seed"]})
    if reuse_source is None:
        mapping, diagnostics = fit_affine_flow(np.asarray(pooled), np.asarray(velocities))
        np.savez(output/"calibration_samples.npz", pooled_flow=pooled, velocities_xy=velocities)
    else:
        prior_calibration = json.loads((reuse_source/"calibration.json").read_text())
        mapping, diagnostics = np.asarray(prior_calibration["mapping"]), prior_calibration["diagnostics"]
    save_json(output/"calibration.json", {"mapping": mapping.tolist(), "diagnostics": diagnostics,
        "frozen_before_held_out_inference": True, "raw_unit_mapping": raw_flow_mapping().tolist()})
    print(json.dumps({"calibration": diagnostics, "mapping": mapping.tolist()}), flush=True)
    results, stills = [], []
    intermediate = output/"preview_intermediate.avi"
    writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*"MJPG"), 50., (960,640))
    if not writer.isOpened():
        raise ValueError("Could not open preview writer")
    try:
        for clip in [synthetic_clip(name) for name in TEST_SEEDS] + [real]:
            arms, statuses, result = run_case(adapter, clip, centers, mapping, output, reuse_source)
            results.append(result)
            render_case(writer, clip, arms, statuses, result, stills)
    finally:
        writer.release()
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(intermediate), "-an", "-c:v", "libx264",
                    "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output/"tracking_preview.mp4")], check=True)
    intermediate.unlink()
    for i, still in enumerate(stills):
        cv2.imwrite(str(output/f"preview_{i:02d}.jpg"), still)
    contact = np.vstack([np.hstack([cv2.resize(s, (480,320)) for s in stills[i:i+4]]) for i in range(0,len(stills),4)])
    cv2.imwrite(str(output/"contact_sheet.jpg"), contact)
    summary = {"status": "experiment_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "actual_flyvis_inference": True, "semantic_recognition": False, "control_authority": False,
        "new_neural_inference_executed": reuse_source is None,
        "reuse_neural_run": str(reuse_source.resolve()) if reuse_source is not None else None,
        "ablation_notes": {"zero_neural": "Decoded flow is zeroed, not neuron activity; affine bias retained.",
                           "spatial_shuffle": "Receptor positions permuted at the same timestep; no future data."},
        "calibration": diagnostics, "cases": results,
        "all_synthetic_cases_pass": all(r["neural_tracking_acceptance_pass"] for r in results if r["kind"]=="synthetic"),
        "elapsed_wall_s": time.perf_counter()-started,
        "real_video_provenance": real["provenance"]}
    save_json(output/"summary.json", summary)
    save_json(output/"artifact_hashes.json", {str(p.relative_to(output)): digest(p)
        for p in sorted(output.rglob("*")) if p.is_file()})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-neural-run", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory must be new; existing experiment artifacts are immutable.")
    try:
        run(args.output, args.reuse_neural_run)
    except Exception as exc:
        if args.output.is_dir() and not (args.output/"summary.json").exists():
            save_json(args.output/"failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
