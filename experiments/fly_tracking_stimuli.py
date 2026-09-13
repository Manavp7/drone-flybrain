"""Predeclared synthetic motion stimuli and a fixed real-video crop.

Synthetic truth is used for calibration/scoring, never tracker correction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SIDE = 391
DT = .02
CALIBRATION = {"stationary": [0., 0.], "right": [30., 0.],
               "left": [-30., 0.], "down": [0., 30.], "up": [0., -30.]}
TEST_SEEDS = {"diagonal": 201, "stop_restart": 202, "reversal": 203}
INITIAL_REAL_BOX = [530., 474., 751., 904.]


def experiment_definition():
    return {
        "schema": 1, "side": SIDE, "dt_s": DT, "warmup_s": .8,
        "calibration": CALIBRATION, "calibration_texture_seed": 101,
        "calibration_duration_s": 2.4, "calibration_pooling_start_s": .4,
        "held_out_seeds": TEST_SEEDS, "test_duration_s": 4.,
        "synthetic_target_size_px": [96, 96],
        "test_motion": {"diagonal": [24, 18],
                        "stop_restart": "[30,-12] at0.8..1.8, stop1.8..2.5, [-18,24] after2.5",
                        "reversal": "[36,0] at0.8..2.4, [-36,0] after2.4"},
        "real_source": "inputs/sabana_grande/source.webm",
        "real_sequences_inclusive": [160, 205],
        "real_fixed_crop_xywh": [420, 0, 1080, 1080],
        "real_initial_box_source_xyxy": INITIAL_REAL_BOX,
        "real_initialization": "Independent agent visual selection from original frame160 before predictions",
        "mapping": "ridge1e-8 affine2input-to2velocity, calibration data only, no lag fitting",
        "tracker": "median of supported receptors inside previous predicted box; minimum9; permanent loss",
        "arms": ["flyvis_calibrated", "flyvis_raw", "zero_motion", "zero_neural", "spatial_shuffle", "farneback"],
        "spatial_shuffle_seed": 731, "spatial_shuffle": "fixed receptor permutation, same current time only",
        "synthetic_acceptance_each_case": {"fraction_iou_ge_0_5": .75, "center_error_reduction_vs_static": .20},
        "labels": "manually initialized target, no semantic recognition",
        "control_authority": False,
    }


def velocity_at(times, name):
    times = np.asarray(times, dtype=float)
    v = np.zeros((len(times), 2), dtype=float)
    moving = times >= .8 - 1e-9
    if name in CALIBRATION:
        v[moving] = CALIBRATION[name]
    elif name == "diagonal":
        v[moving] = [24., 18.]
    elif name == "stop_restart":
        v[moving & (times < 1.8 - 1e-9)] = [30., -12.]
        v[times >= 2.5 - 1e-9] = [-18., 24.]
    elif name == "reversal":
        v[moving & (times < 2.4 - 1e-9)] = [36., 0.]
        v[times >= 2.4 - 1e-9] = [-36., 0.]
    else:
        raise ValueError(name)
    return v


def synthetic_clip(name):
    is_calibration = name in CALIBRATION
    seed = 101 if is_calibration else TEST_SEEDS[name]
    duration = 2.4 if is_calibration else 4.
    times = np.arange(round(duration / DT) + 1, dtype=np.float64) * DT
    v = velocity_at(times, name)
    offsets = np.vstack([np.zeros(2), np.cumsum(v[:-1] * DT, axis=0)])
    boxes = np.array([147.5, 147.5, 243.5, 243.5]) + offsets[:, [0, 1, 0, 1]]
    rng = np.random.default_rng(seed)
    background = cv2.resize(rng.uniform(.28, .46, (32, 32)).astype(np.float32), (SIDE, SIDE))
    texture = cv2.resize(rng.uniform(.04, .96, (12, 12)).astype(np.float32), (96, 96))
    # Subpixel affine placement avoids integer-position velocity discontinuities.
    patch = np.zeros((SIDE, SIDE), np.float32)
    mask = np.zeros_like(patch)
    patch[:96, :96], mask[:96, :96] = texture, 1.
    frames = []
    for box in boxes:
        transform = np.array([[1, 0, box[0]], [0, 1, box[1]]], np.float32)
        moved = cv2.warpAffine(patch, transform, (SIDE, SIDE), flags=cv2.INTER_LINEAR)
        alpha = cv2.warpAffine(mask, transform, (SIDE, SIDE), flags=cv2.INTER_LINEAR)
        frames.append(background * (1 - alpha) + moved)
    return {"name": name, "frames": np.asarray(frames, np.float32), "times": times,
            "truth_boxes": boxes, "velocities": v, "seed": seed, "kind": "synthetic"}


def source_to_retina(boxes):
    boxes = np.asarray(boxes, dtype=float)
    return (boxes - [420., 0., 420., 0.]) * SIDE / 1080.


def retina_to_source(boxes):
    return np.asarray(boxes, dtype=float) * 1080. / SIDE + [420., 0., 420., 0.]


def real_clip(project: Path):
    source = project / "inputs/sabana_grande/source.webm"
    expected = json.loads((source.parent / "provenance.json").read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected["sha256"]:
        raise ValueError("Source hash mismatch")
    pts_data = json.loads((source.parent / "timestamps.json").read_text())
    pts = np.array([float(f["best_effort_timestamp_time"]) for f in pts_data["frames"]])
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 160)
    frames, color = [], []
    for sequence in range(160, 206):
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Could not decode source frame {sequence}")
        cropped = cv2.resize(frame[:, 420:1500], (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        color.append(cropped)
        frames.append(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.)
    cap.release()
    return {"name": "real_pedestrian", "frames": np.asarray(frames, np.float32),
            "color": np.asarray(color), "times": pts[160:206],
            "initial_box": source_to_retina(INITIAL_REAL_BOX), "kind": "real_video",
            "source_sequences": np.arange(160, 206), "provenance": expected}
