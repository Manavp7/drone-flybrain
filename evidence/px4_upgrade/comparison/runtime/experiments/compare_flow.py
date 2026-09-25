"""Compare a frozen Flyvis decoder with Farneback: agreement, not accuracy.

The input is an existing real neural run. No fitting or neural rerun occurs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments.video_experiment import write_json
from flybrain_sim.research_model import center_crop_square, file_sha256, validate_clip

WARMUP_S = .5
REFERENCE_THRESHOLD = 1e-3
SOURCE_HEIGHT = 436
SOURCE_FPS = 24
FARNEBACK = dict(pyr_scale=.5, levels=3, winsize=21, iterations=3,
                 poly_n=5, poly_sigma=1.2, flags=0)


def align_pairs(frame_times, stimulus_times, response_times, warmup_s=WARMUP_S):
    """Select the first model step containing each pair's current image.

    Pair index i denotes frame i -> i+1. A final image not presented to the
    neural model is excluded. Warmup uses the pair endpoint's source time.
    """
    arrays = [np.asarray(x, dtype=np.float64) for x in
              (frame_times, stimulus_times, response_times)]
    frames, stimulus, response = arrays
    for values in arrays:
        if (values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all()
                or np.any(np.diff(values) <= 0)):
            raise ValueError("Timestamps must be finite and strictly increasing")
    if (len(frames) < 2 or stimulus.shape != response.shape
            or np.any(response <= stimulus) or not np.isfinite(warmup_s) or warmup_s < 0):
        raise ValueError("Invalid response timing or warmup")
    steps = np.searchsorted(stimulus, frames[1:] - 1e-10, side="left")
    keep = ((steps < len(stimulus)) &
            (frames[1:] - frames[0] >= warmup_s - 1e-10))
    pairs = np.flatnonzero(keep)
    return pairs, steps[keep]


def decoder_unit_fields(pixel_flow, delta_s):
    """Map [H,W,(right,down)] displacements to normalized [right,up] fields.

    This transfers the pretrained Sintel target convention to this video;
    it is neither calibrated physical velocity nor a ground-truth target.
    """
    field = np.asarray(pixel_flow, dtype=np.float32)
    if (field.ndim != 3 or field.shape[-1] != 2 or not np.isfinite(field).all()
            or not np.isfinite(delta_s) or delta_s <= 0):
        raise ValueError("Expected finite [H,W,2] pixel flow and positive elapsed seconds")
    result = np.moveaxis(field, -1, 0).copy()
    result[1] *= -1
    result /= np.float32(SOURCE_HEIGHT * SOURCE_FPS * delta_s)
    return result


def agreement(prediction, reference):
    if (prediction.shape != reference.shape or prediction.ndim != 2
            or prediction.shape[0] != 2 or not np.isfinite(prediction).all()
            or not np.isfinite(reference).all()):
        raise ValueError("Expected matching finite [2,receptor] arrays")
    pred_norm = np.linalg.norm(prediction, axis=0)
    ref_norm = np.linalg.norm(reference, axis=0)
    valid = (ref_norm > REFERENCE_THRESHOLD) & (pred_norm > 1e-12)
    cosine = np.sum(prediction[:, valid] * reference[:, valid], axis=0)
    cosine /= pred_norm[valid] * ref_norm[valid]
    return {"mean_cosine_nonzero_reference": float(np.clip(cosine, -1, 1).mean()) if valid.any() else None,
            "cosine_receptor_count": int(valid.sum()),
            "reference_nonzero_count": int((ref_norm > REFERENCE_THRESHOLD).sum()),
            "mean_absolute_vector_difference": float(np.linalg.norm(prediction - reference, axis=0).mean()),
            "mean_component_absolute_difference": float(np.abs(prediction - reference).mean()),
            "prediction_mean_magnitude": float(pred_norm.mean()),
            "reference_mean_magnitude": float(ref_norm.mean())}


def run_comparison(args):
    started = time.perf_counter()
    run = Path(args.run).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    try:
        source = json.loads((run / "summary.json").read_text())
        if not source.get("inference_executed") or source.get("status") != "inference_completed":
            raise ValueError("Requires a completed real neural run")
        hashes = {name: file_sha256(run / name) for name in ("clip.npz", "activity.npy", "summary.json")}
        for name in ("clip.npz", "activity.npy"):
            if hashes[name] != source["artifact_hashes"][name]:
                raise ValueError(f"Neural run artifact integrity mismatch: {name}")
        if file_sha256(manifest) != source["manifest_sha256"]:
            raise ValueError("Comparison manifest differs from neural inference manifest")
        with np.load(run / "clip.npz", allow_pickle=False) as clip:
            frames, timestamps = clip["frames"], clip["timestamps"]
        sampling = validate_clip(frames, timestamps, source["integration_dt_s"])
        stimulus = np.asarray(source["stimulus_timestamps_s"], dtype=np.float64)
        response = np.asarray(source["response_timestamps_s"], dtype=np.float64)
        for observed, expected in ((stimulus, sampling.stimulus_timestamps),
                                   (response, sampling.response_timestamps)):
            if observed.shape != expected.shape or not np.allclose(observed, expected, atol=1e-10, rtol=0):
                raise ValueError("Saved timing differs from the input sampling contract")
        activity = np.load(run / "activity.npy", mmap_mode="r", allow_pickle=False)
        if activity.shape != (len(stimulus), 45669) or not np.isfinite(activity).all():
            raise ValueError("Invalid real neural activity")
        from experiments.runtime import VideoFlyvisAdapter
        import cv2
        import torch
        cv2.setNumThreads(2)
        torch.set_num_threads(2)
        adapter = VideoFlyvisAdapter(manifest)
        # DecoderGAVP is spatial per frame; eval mode freezes batch statistics.
        prediction = np.concatenate([adapter.decode_flow(np.array(activity[i:i + 32]))
                                     for i in range(0, len(activity), 32)], axis=0)
        np.save(output / "decoded_flow.npy", prediction, allow_pickle=False)
        centers = adapter.eye.receptor_centers.detach().cpu().numpy() + 195
        np.save(output / "receptor_centers_row_col.npy", centers, allow_pickle=False)
        square, spatial = center_crop_square(frames)
        def render(index):
            with torch.inference_mode():
                image = torch.as_tensor(square[index:index + 1][None].copy(), device=adapter.device)
                image = torch.nn.functional.interpolate(image, size=(391, 391), mode="bilinear",
                                                       align_corners=False, antialias=True)
            return np.rint(np.clip(image.cpu().numpy()[0, 0], 0, 1) * 255).astype(np.uint8)
        previous = render(0)
        reference = np.empty((len(frames) - 1, 2, 721), dtype=np.float32)
        for i in range(1, len(frames)):
            current = render(i)
            flow = cv2.calcOpticalFlowFarneback(previous, current, None, **FARNEBACK)
            fields = decoder_unit_fields(flow, float(timestamps[i] - timestamps[i - 1]))
            with torch.inference_mode():
                filtered = adapter.eye(torch.as_tensor(fields[None], device=adapter.device), ftype="sum")
            reference[i - 1] = filtered.cpu().numpy()[0, :, 0]
            previous = current
        np.save(output / "reference_flow.npy", reference, allow_pickle=False)
        pairs, steps = align_pairs(timestamps, stimulus, response)
        if not len(pairs):
            raise ValueError("No causally aligned frame pairs remain after fixed 0.5-second warmup")
        aligned_pred, aligned_ref = prediction[steps], reference[pairs]
        np.savez_compressed(output / "aligned_flow.npz", prediction=aligned_pred, reference=aligned_ref,
                            pair_indices=pairs, previous_frame_indices=pairs, current_frame_indices=pairs + 1,
                            neural_step_indices=steps, previous_frame_timestamps_s=timestamps[pairs],
                            current_frame_timestamps_s=timestamps[pairs + 1],
                            stimulus_timestamps_s=stimulus[steps], response_timestamps_s=response[steps])
        rows = [{"pair_index": int(p), "neural_step_index": int(s),
                 "frame_time_s": float(timestamps[p + 1]), "stimulus_time_s": float(stimulus[s]),
                 "response_time_s": float(response[s]), **agreement(prediction[s], reference[p])}
                for p, s in zip(pairs, steps)]
        with (output / "per_frame_metrics.csv").open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        pooled = agreement(aligned_pred.transpose(1, 0, 2).reshape(2, -1),
                           aligned_ref.transpose(1, 0, 2).reshape(2, -1))
        summary = {
            "status": "agreement_comparison_completed", "created_utc": datetime.now(timezone.utc).isoformat(),
            "control_authority": False, "neural_inference_rerun": False, "decoder_fitting": False,
            "decoder": "official_checkpoint_DecoderGAVP_flow_strict_state_load_eval",
            "source_run": str(run), "source_artifact_hashes": hashes,
            "manifest_sha256": source["manifest_sha256"], "checkpoint_sha256": source["checkpoint_sha256"],
            "prediction_shape": list(prediction.shape), "reference_shape": list(reference.shape),
            "aligned_pairs": len(pairs), "total_pairs": len(reference), "warmup_s": WARMUP_S,
            "alignment": "first_stimulus_at_or_after_current_frame_pts; fixed_0.5s_warmup; no_lag_optimization",
            "spatial_transform": spatial, "reference_resize": "391x391_bilinear_align_corners_false_antialias_true",
            "reference_uint8_conversion": "round(clip(luminance,0,1)*255)",
            "reference": "OpenCV_Farneback", "farneback_parameters": FARNEBACK,
            "opencv_version": cv2.__version__, "numpy_version": np.__version__,
            "unit_transfer_assumption": "BoxEye_sum([pixel_dx,-pixel_dy]/(436*24*actual_pair_delta_seconds))",
            "unit_rationale": "Sintel sample_flow divides displacement by original height 436; source 24fps; temporal resampling leaves magnitude unchanged",
            "axes": ["right_positive", "up_positive"], "kernel_size": [13, 13],
            "receptor_centers": "row_col_in_391_square; eye.receptor_centers+195",
            "cosine_reference_magnitude_threshold": REFERENCE_THRESHOLD, "pooled_agreement": pooled,
            "limitations": ["Agreement with an estimated baseline is not ground-truth accuracy",
                            "Target scale is an explicit cross-video transfer assumption, not physical calibration",
                            "Pinhole footage is not a calibrated fly compound-eye recording",
                            "This clip does not establish navigation utility or generalization",
                            "Frame-pair Farneback and causal neural dynamics can have different response delays",
                            "No decoder fitting, lag selection, flight control, or physical flight"],
            "elapsed_wall_s": time.perf_counter() - started,
            "artifact_hashes": {p.name: file_sha256(p) for p in sorted(output.iterdir())}}
        write_json(output / "summary.json", summary)
        return summary
    except Exception as exc:
        write_json(output / "failure.json", {"status": "comparison_failed", "error": str(exc),
                                             "error_type": type(exc).__name__, "control_authority": False})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True, help="New directory; existing outputs are refused")
    args = parser.parse_args(argv)
    try:
        summary = run_comparison(args)
    except Exception as exc:
        print(f"Comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
