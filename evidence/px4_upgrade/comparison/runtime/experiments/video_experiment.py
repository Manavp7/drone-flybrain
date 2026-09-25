"""Bounded, offline, real-checkpoint video inference with reproducible artifacts.

Run from the project root: python -m experiments.video_experiment --help.
No downloads, synthetic inference fallback, or flight-control interface.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np

from flybrain_sim.research_model import file_sha256, validate_clip


def source_timestamps(values) -> np.ndarray:
    """Validate presentation timestamps in display/decode order, in seconds."""
    if not isinstance(values, list) or len(values) < 2:
        raise ValueError("Source timestamps must be a JSON list with at least two values")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in values):
        raise ValueError("Source timestamps must contain numeric seconds")
    pts = np.asarray(values, dtype=np.float64)
    if not np.isfinite(pts).all() or np.any(pts < 0) or np.any(np.diff(pts) <= 0):
        raise ValueError("Source timestamps must be finite, nonnegative, and strictly increasing")
    return pts


def validate_options(start_s, duration_s, sample_fps, max_width, dt):
    for name, value in (("start_s", start_s), ("duration_s", duration_s),
                        ("sample_fps", sample_fps), ("dt", dt)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
    if start_s < 0 or not 0.04 <= duration_s <= 10:
        raise ValueError("start_s must be nonnegative and duration_s must be 0.04–10 seconds")
    if not 4 <= sample_fps <= 120 or not 0.001 <= dt <= 0.02:
        raise ValueError("sample_fps must be 4–120 and dt must be 0.001–0.02 seconds")
    if isinstance(max_width, bool) or not isinstance(max_width, int) or not 2 <= max_width <= 2048:
        raise ValueError("max_width must be an integer between 2 and 2048")


def decode_clip(video: Path, *, timestamps_path=None, start_s=0., duration_s=2.,
                sample_fps=15., max_width=256, dt=.02):
    """Select frames at or after each sample target; retain actual source PTS.

    External PTS must enumerate source frames in display order. Without them,
    OpenCV POS_MSEC must increase on every decoded frame; nominal FPS is never
    substituted for missing timing. The window is inclusive of its endpoint.
    """
    validate_options(start_s, duration_s, sample_fps, max_width, dt)
    import cv2

    pts = None
    if timestamps_path is not None:
        timestamps_path = Path(timestamps_path).resolve()
        pts = source_timestamps(json.loads(timestamps_path.read_text()))
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {video}")
    frames, times, indices = [], [], []
    next_sample = start_s
    previous = -1.
    source_count = 0
    original_hw = None
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    metadata = {"timing_source": "external_source_frame_pts" if pts is not None else "opencv_pos_msec",
                "nominal_source_fps": source_fps if math.isfinite(source_fps) else None,
                "sample_selection": "first_source_frame_at_or_after_each_target_no_duplicates",
                "requested_start_s": start_s, "requested_duration_s": duration_s,
                "requested_sample_fps": sample_fps, "grayscale": "opencv_BGR2GRAY_uint8_div_255",
                "resize": "aspect_preserving_width_cap_INTER_AREA", "max_width": max_width}
    if pts is not None:
        metadata.update(timestamps_path=str(timestamps_path),
                        timestamps_file_sha256=file_sha256(timestamps_path),
                        external_timestamp_count=len(pts))
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if pts is not None and source_count != len(pts):
                    raise ValueError("Decoded source frame count does not match external timestamps")
                break
            index = source_count
            source_count += 1
            if pts is not None and index >= len(pts):
                raise ValueError("External timestamp list is shorter than decoded source video")
            timestamp = float(pts[index]) if pts is not None else float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.
            if not math.isfinite(timestamp) or timestamp < 0 or timestamp <= previous:
                raise ValueError("Video PTS unavailable or non-increasing; provide verified --timestamps")
            previous = timestamp
            if timestamp > start_s + duration_s + 1e-9:
                break
            if timestamp + 1e-9 < next_sample:
                continue
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise ValueError("Expected decoded uint8 BGR frames")
            height, width = frame.shape[:2]
            if original_hw is None:
                original_hw = [height, width]
            if original_hw != [height, width]:
                raise ValueError("Variable frame dimensions are unsupported")
            if width > max_width:
                frame = cv2.resize(frame, (max_width, max(2, round(height * max_width / width))),
                                   interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / np.float32(255.)
            if (len(frames) + 1) * gray.size > 32_000_000:
                raise ValueError("Decoded clip would exceed the 32 million pixel limit")
            frames.append(gray)
            times.append(timestamp)
            indices.append(index)
            target_number = math.floor((timestamp - start_s + 1e-9) * sample_fps) + 1
            next_sample = start_s + target_number / sample_fps
    finally:
        capture.release()
    if len(frames) < 2:
        raise ValueError("Requested video window contains fewer than two selected frames")
    clip = np.stack(frames).astype(np.float32, copy=False)
    times = np.asarray(times, dtype=np.float64)
    validate_clip(clip, times, dt)
    metadata.update(original_hw=original_hw, decoded_source_frames=source_count,
                    selected_source_indices=indices, selected_shape=list(clip.shape),
                    actual_start_s=float(times[0]), actual_end_s=float(times[-1]),
                    actual_span_s=float(times[-1] - times[0]))
    return clip, times, metadata


def aggregate_activity(activity, labels, response_times):
    activity = np.asarray(activity)
    times = np.asarray(response_times, dtype=np.float64)
    labels = np.asarray(labels)
    if activity.ndim != 2 or min(activity.shape) < 1 or not np.isfinite(activity).all():
        raise ValueError("Neural activity must be a finite nonempty [time, neuron] array")
    if labels.ndim != 1 or len(labels) != activity.shape[1] or labels.dtype.kind not in "SU":
        raise ValueError("Connectome labels must be strings matching every activity column")
    if times.shape != (len(activity),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Response timestamps must match activity and strictly increase")
    labels = np.char.decode(labels, "utf-8") if labels.dtype.kind == "S" else labels.astype(str)
    if np.any(labels == ""):
        raise ValueError("Empty cell-type labels are unsupported")
    types = np.unique(labels)
    result = {"cell_types": types, "time_s": times,
              "neuron_count": np.asarray([(labels == name).sum() for name in types], dtype=np.int64)}
    for name, reduction in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max)):
        result[name] = np.stack([reduction(activity[:, labels == label], axis=1) for label in types], axis=1)
    return labels, result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def package_info():
    versions = {}
    for name in ("numpy", "opencv-python", "opencv-python-headless", "torch", "flyvis", "datamate"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "packages": versions}


def run_experiment(args):
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    stage = "input_validation"
    inference_started = inference_completed = False
    try:
        video = Path(args.video).expanduser().resolve()
        manifest = Path(args.manifest).expanduser().resolve()
        source = {"video_path": str(video), "video_sha256": file_sha256(video),
                  "video_bytes": video.stat().st_size, "manifest_path": str(manifest),
                  "manifest_sha256": file_sha256(manifest), "created_utc": datetime.now(timezone.utc).isoformat(),
                  "runtime": package_info()}
        write_json(output / "source.json", source)
        stage = "video_decode"
        frames, timestamps, metadata = decode_clip(
            video, timestamps_path=args.timestamps, start_s=args.start_s, duration_s=args.duration_s,
            sample_fps=args.sample_fps, max_width=args.max_width, dt=args.dt)
        np.savez_compressed(output / "clip.npz", frames=frames, timestamps=timestamps)
        write_json(output / "input.json", metadata)
        stage = "model_initialization"
        from experiments.runtime import VideoFlyvisAdapter
        adapter = VideoFlyvisAdapter(manifest)
        stage = "neural_inference"
        inference_started = True
        summary, activity = adapter.infer_clip(frames, timestamps, dt=args.dt)
        inference_completed = True
        stage = "output_validation"
        labels, grouped = aggregate_activity(activity, adapter.network.connectome.nodes.type[:],
                                            summary["response_timestamps_s"])
        if list(activity.shape) != summary["activity_shape"] or not summary["inference_executed"]:
            raise ValueError("Adapter output contradicts its inference summary")
        np.save(output / "activity.npy", activity, allow_pickle=False)
        np.save(output / "neuron_types.npy", labels, allow_pickle=False)
        np.savez_compressed(output / "cell_type_activity.npz", **grouped)
        with (output / "cell_type_activity.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["response_time_s", "cell_type", "neuron_count", "mean", "std", "min", "max"])
            for t, timestamp in enumerate(grouped["time_s"]):
                for c, label in enumerate(grouped["cell_types"]):
                    writer.writerow([timestamp, label, grouped["neuron_count"][c]] +
                                    [grouped[name][t, c] for name in ("mean", "std", "min", "max")])
        summary.update(source=source, input=metadata, cell_type_count=len(grouped["cell_types"]),
                       experiment_elapsed_wall_s=time.perf_counter() - started,
                       artifact_hashes={path.name: file_sha256(path) for path in sorted(output.iterdir())})
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {
            "status": "experiment_failed", "stage": stage, "error_type": type(exc).__name__,
            "error": str(exc), "inference_started": inference_started, "inference_completed": inference_completed,
            "control_authority": False, "elapsed_wall_s": time.perf_counter() - started,
            "traceback": traceback.format_exc(), "runtime": package_info()})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New output directory; existing directories are refused")
    parser.add_argument("--timestamps", help="JSON list of source-frame presentation timestamps in seconds")
    parser.add_argument("--start-s", type=float, default=0.)
    parser.add_argument("--duration-s", type=float, default=2.)
    parser.add_argument("--sample-fps", type=float, default=15.)
    parser.add_argument("--max-width", type=int, default=256)
    parser.add_argument("--dt", type=float, default=.02)
    args = parser.parse_args(argv)
    try:
        summary = run_experiment(args)
    except Exception as exc:
        print(f"Experiment failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": summary["status"], "output": str(Path(args.output).resolve()),
                      "activity_shape": summary["activity_shape"],
                      "inference_elapsed_wall_s": summary["elapsed_wall_s"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
