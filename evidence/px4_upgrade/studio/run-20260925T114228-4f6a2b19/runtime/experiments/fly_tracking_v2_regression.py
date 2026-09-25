"""Replay the three previous failures with the already-frozen v2 selection.

This is a development regression, not a fresh held-out test. It reuses verified
original neural activity and flow without inference, fitting, or model selection.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from experiments.fly_tracking_core import evaluate_tracking, visible_response_indices
from experiments.fly_motion_readout import LearnedFlowTracker, RidgeReadout
from experiments.fly_neural_template import NeuralTemplateTracker
from flybrain_sim.research_model import validate_clip, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
CASES = ("diagonal", "stop_restart", "reversal")
DT = .02
CELL_TYPES = ["L1", "L2", "L3", "Mi1", "Tm1", "Tm2", "Mi4", "Mi9",
              "T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d"]
BANKS = {"early3": [0, 1, 2], "all16": list(range(16))}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def load_json(path):
    return json.loads(Path(path).read_text())


def checked(root, relative, hashes):
    path = root / relative
    expected = hashes.get(relative)
    if expected is None or sha(path) != expected:
        raise ValueError(f"Original artifact hash mismatch: {path}")
    return path


def aligned(rows, sampling, timestamps, initial):
    """Show only responses whose full model step has already completed."""
    visible = visible_response_indices(sampling.response_timestamps, timestamps)
    boxes = np.asarray([initial if i < 0 else rows[i]["box_xyxy"] for i in visible])
    statuses = np.asarray(["initialized" if i < 0 else rows[i]["status"] for i in visible])
    for i, row in enumerate(rows):
        row.update(stimulus_time_s=float(sampling.stimulus_timestamps[i]),
                   response_time_s=float(sampling.response_timestamps[i]),
                   source_input_index=int(sampling.indices[i]))
    return boxes, statuses


def replay(selected, best_motion, features, flow, centers, initial, scale):
    if selected["method"] == "template":
        config = dict(selected["config"])
        channels = BANKS[config.pop("bank")]
        data = features[:, channels, :]
        tracker = NeuralTemplateTracker(centers, initial, data[0],
                                        channel_scale=scale[channels], **config)
        rows = [{"box_xyxy": initial.tolist(), "status": "initialized", "score": None}]
        rows.extend(tracker.step(field, DT) for field in data[1:])
        return rows
    if selected["method"] != "motion":
        raise ValueError("Unknown frozen selection method")
    if selected["name"] != best_motion["name"] or selected["mode"] != best_motion["mode"]:
        raise ValueError("Frozen motion selection and stored model disagree")
    model = RidgeReadout.from_dict(best_motion["model"])
    tracker = LearnedFlowTracker(centers, initial, model, mode=selected["mode"])
    return [tracker.step(field, DT) for field in flow]


def run(v2_run, output):
    frozen_path = v2_run / "selection_frozen.json"
    if not frozen_path.is_file():
        raise FileNotFoundError("v2 selection_frozen.json must exist before regression evaluation")
    if output.exists():
        raise FileExistsError("Output must be a new directory; preserve previous experiment artifacts")
    started = time.perf_counter()
    cv2.setNumThreads(2)
    frozen = load_json(frozen_path)
    selection_sha = sha(frozen_path)
    definition = load_json(v2_run / "definition.json")
    model = validate_manifest(ROOT / "models/flyvis_0000_000.manifest.json")
    checkpoint_hash = model.manifest["files"][model.manifest["checkpoint"]]
    scale_path = v2_run / "training_channel_scale.npy"
    if sha(scale_path) != frozen["training_scale_sha256"]:
        raise ValueError("Frozen training channel scale changed")
    scale = np.load(scale_path, allow_pickle=False)
    if scale.shape != (16,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Invalid frozen training channel scale")
    binding_path = v2_run / "neural_binding.npz"
    with np.load(binding_path, allow_pickle=False) as binding:
        indices, centers = binding["indices"], binding["centers_rc"]
        cell_types = binding["cell_types"].astype(str).tolist()
    if cell_types != CELL_TYPES or definition["cell_types"] != CELL_TYPES:
        raise ValueError("Neural binding channel order does not match v2")
    if (indices.dtype.kind not in "iu" or indices.shape != (16, 721)
            or np.any(indices < 0) or np.any(indices >= 45669)
            or len(np.unique(indices)) != indices.size):
        raise ValueError("Invalid neural binding indices")
    if centers.shape != (721, 2) or not np.isfinite(centers).all():
        raise ValueError("Invalid receptor coordinates")
    # Tracking code must be the same implementation used for v2 selection.
    for relative in ("experiments/fly_motion_readout.py", "experiments/fly_neural_template.py"):
        if sha(ROOT / relative) != definition["source_hashes"][relative]:
            raise ValueError(f"Tracker source differs from v2 selection source: {relative}")
    original = ROOT / "results/fly_tracking_run01"
    before_report = ROOT / "results/fly_tracking_report02"
    lock = load_json(ROOT / "experiments/flyvis_tracking_lock.json")
    for folder in (original, before_report):
        manifest = folder / "artifact_hashes.json"
        if sha(manifest) != lock["files"][str(manifest.relative_to(ROOT))]:
            raise ValueError("Previous result hash manifest changed")
    old_hashes = load_json(original / "artifact_hashes.json")
    before_hashes = load_json(before_report / "artifact_hashes.json")
    old_centers = np.load(checked(original, "receptor_centers_rc.npy", old_hashes), allow_pickle=False)
    np.testing.assert_array_equal(centers, old_centers)
    output.mkdir(parents=True, exist_ok=False)
    try:
        source_paths = [Path(__file__), ROOT / "experiments/fly_motion_readout.py",
                        ROOT / "experiments/fly_neural_template.py", ROOT / "experiments/fly_tracking_core.py",
                        ROOT / "flybrain_sim/research_model.py"]
        source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
        definition = {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "evaluation_type": "development_regression_on_previous_failures",
                      "new_neural_inference_executed": False, "fitting_or_selection_executed": False,
                      "selected": frozen["selected"], "v2_run": str(v2_run.resolve()),
                      "selection_sha256": selection_sha, "neural_binding_sha256": sha(binding_path),
                      "training_scale_sha256": sha(scale_path),
                      "manifest_sha256": model.manifest_sha256, "checkpoint_sha256": checkpoint_hash,
                      "source_hashes": source_hashes,
                      "score_policy": {"start_time_s": .8, "scored_frames_per_case": 161,
                                       "minimum_fraction_active_iou_ge_0_5": .75,
                                       "minimum_error_reduction_vs_static": .2,
                                       "lost_frames_can_earn_hits": False},
                      "cases_fixed_before_replay": list(CASES)}
        save(output / "definition.json", definition)
        results = []
        for name in CASES:
            folder = output / name
            folder.mkdir()
            required = ("input.npz", "activity.npy", "decoded_flow.npy", "neural_receipt.json", "truth.json")
            paths = {member: checked(original, f"{name}/{member}", old_hashes) for member in required}
            before_path = checked(before_report, f"{name}/metrics.json", before_hashes)
            receipt = load_json(paths["neural_receipt.json"])
            if (receipt["manifest_sha256"] != model.manifest_sha256
                    or receipt["checkpoint_sha256"] != checkpoint_hash
                    or receipt["source_revision"] != model.manifest["source_revision"]
                    or receipt["inference_executed"] is not True):
                raise ValueError(f"Original neural receipt model mismatch: {name}")
            with np.load(paths["input.npz"], allow_pickle=False) as payload:
                frames, times = payload["frames"], payload["timestamps"]
            sampling = validate_clip(frames, times, DT)
            if (hashlib.sha256(frames.tobytes()).hexdigest() != receipt["input_sha256"]
                    or hashlib.sha256(np.asarray(times, dtype=np.float64).tobytes()).hexdigest() != receipt["timestamps_sha256"]):
                raise ValueError(f"Original neural input content mismatch: {name}")
            np.testing.assert_array_equal(sampling.stimulus_timestamps, receipt["stimulus_timestamps_s"])
            np.testing.assert_array_equal(sampling.response_timestamps, receipt["response_timestamps_s"])
            activity = np.load(paths["activity.npy"], allow_pickle=False)
            flow = np.load(paths["decoded_flow.npy"], allow_pickle=False)
            steps = len(sampling.indices)
            if (activity.shape != (steps, 45669) or flow.shape != (steps, 2, 721)
                    or not np.isfinite(activity).all() or not np.isfinite(flow).all()):
                raise ValueError(f"Malformed original neural arrays: {name}")
            features = activity[:, indices]
            truth_payload = load_json(paths["truth.json"])
            truth = np.asarray(truth_payload["boxes_xyxy"], dtype=float)
            np.testing.assert_array_equal(times, truth_payload["timestamps"])
            if truth.shape != (len(times), 4) or not np.isfinite(truth).all():
                raise ValueError("Invalid original truth geometry")
            initial = truth[0].copy()
            rows = replay(frozen["selected"], frozen["best_motion"], features, flow, centers, initial, scale)
            boxes, statuses = aligned(rows, sampling, times, initial)
            score_mask = times >= .8
            if int(score_mask.sum()) != 161:
                raise ValueError("Original regression evaluation frame count changed")
            after = evaluate_tracking(boxes[score_mask], truth[score_mask], statuses[score_mask] == "tracking")
            static_boxes = np.broadcast_to(initial, truth.shape)
            static = evaluate_tracking(static_boxes[score_mask], truth[score_mask], np.ones(score_mask.sum(), bool))
            improvement = 1 - after["mean_center_error"] / static["mean_center_error"]
            passed = bool(after["fraction_active_iou_ge_0_5"] >= .75 and improvement >= .2)
            before = load_json(before_path)
            old_static = before["metrics_retina_pixels"]["zero_motion"]
            for key in static:
                if not np.isclose(static[key], old_static[key], atol=1e-10, rtol=1e-10):
                    raise ValueError(f"Original static score reproduction mismatch: {key}")
            prefix = (times > DT + 1e-9) & (times < .8)
            prefix_error = np.linalg.norm((boxes[prefix, :2] + boxes[prefix, 2:]
                                           - truth[prefix, :2] - truth[prefix, 2:]) / 2, axis=1)
            result = {"case": name, "evaluation_type": "development_regression",
                      "before": before["metrics_retina_pixels"]["flyvis_calibrated"],
                      "before_acceptance_pass": before["neural_tracking_acceptance_pass"],
                      "after": after, "static": static, "after_acceptance_pass": passed,
                      "after_center_error_reduction_vs_static": improvement,
                      "after_stationary_prefix_max_center_error": float(prefix_error.max()),
                      "score_policy": "Original times >= 0.8s; all 161 frames; success requires active status and IoU >= 0.5"}
            np.savez_compressed(folder / "display_boxes.npz", timestamps=times, selected=boxes, static=static_boxes,
                                truth=truth)
            save(folder / "traces.json", {"selected": rows})
            save(folder / "display_statuses.json", {"selected": statuses.tolist()})
            save(folder / "metrics.json", result)
            save(folder / "neural_source.json", {
                "source": str((original / name).resolve()), "new_neural_inference_executed": False,
                "checked_sha256": {member: sha(path) for member, path in paths.items()},
                "before_metrics_sha256": sha(before_path), "neural_binding_sha256": sha(binding_path),
                "input_receipt_match": True, "model_receipt_match": True,
                "activity_channels": CELL_TYPES, "activity_shape": list(activity.shape),
                "selected_feature_shape": list(features.shape)})
            results.append(result)
            print(json.dumps({"case": name, "after_pass": passed,
                              "active_iou_fraction": after["fraction_active_iou_ge_0_5"],
                              "mean_center_error": after["mean_center_error"]}), flush=True)
            del frames, activity, features, flow
        if sha(frozen_path) != selection_sha:
            raise ValueError("Selection changed during regression replay")
        summary = {"status": "regression_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
                   "evaluation_type": "development_regression_not_fresh_heldout",
                   "new_neural_inference_executed": False, "fitting_or_selection_executed": False,
                   "selected": frozen["selected"], "cases": results,
                   "before_passed": sum(r["before_acceptance_pass"] for r in results),
                   "after_passed": sum(r["after_acceptance_pass"] for r in results),
                   "elapsed_wall_s": time.perf_counter() - started,
                   "limitation": "These previously failed clips are development regressions; fresh held-out evidence is reported separately by the v2 run."}
        save(output / "summary.json", summary)
        save(output / "artifact_hashes.json", {str(path.relative_to(output)): sha(path)
             for path in sorted(output.rglob("*")) if path.is_file()})
        print(json.dumps(summary), flush=True)
    except Exception as exc:
        save(output / "failure.json", {"status": "regression_failed", "error_type": type(exc).__name__,
                                       "error": str(exc), "new_neural_inference_executed": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.v2_run.resolve(), args.output.resolve())
