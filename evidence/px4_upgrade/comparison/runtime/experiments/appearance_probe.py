"""Prototype clothing-color gate; replay existing detections, never run a model.

This is a local association experiment, not person identification or production
tracking. Thresholds/feature geometry were chosen before this replay. Appearance
uses a coarse upper-body RGB color histogram, not faces or learned embeddings.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from perception.detector import Detection, validate_rgb_image
from perception.pipeline import ShortTermTracker, iou


SETTINGS = {
    "feature": "normalized joint RGB histogram, 4 bins per channel (64 total)",
    "roi_xyxy_box_fraction": [.2, .25, .8, .6],
    "maximum_feature_pixels": 1024,
    "similarity": "Bhattacharyya coefficient: sum(sqrt(hist_a * hist_b))",
    "minimum_appearance_similarity": .65,
    "minimum_iou": .3,
    "match_score": "IoU * similarity for person; IoU for other classes",
    "appearance_reference": "last observed clothing histogram; no moving-average adaptation",
    "max_age_s": .5,
    "max_tracks": 64,
    "sample_every": 3,
    "neural_inference_executed": False,
    "mode": "prototype replay of saved real detector outputs on source RGB frames",
}


def clothing_histogram(image_rgb, box):
    """Use at most 32x32 original pixels from the central upper-body rectangle."""
    h, w = image_rgb.shape[:2]
    left, top, right, bottom = box
    bw, bh = right - left, bottom - top
    x0, x1 = max(0, math.floor(left + .2 * bw)), min(w, math.ceil(left + .8 * bw))
    y0, y1 = max(0, math.floor(top + .25 * bh)), min(h, math.ceil(top + .6 * bh))
    if x1 <= x0 or y1 <= y0:
        return None
    ys = np.linspace(y0, y1 - 1, min(32, y1 - y0), dtype=int)
    xs = np.linspace(x0, x1 - 1, min(32, x1 - x0), dtype=int)
    pixels = image_rgb[ys[:, None], xs[None, :]].astype(np.int32) // 64
    histogram = np.bincount((pixels[..., 0] * 16 + pixels[..., 1] * 4 + pixels[..., 2]).ravel(), minlength=64).astype(float)
    return histogram / histogram.sum()


class AppearanceProbeTracker(ShortTermTracker):
    """Preserve original greedy geometry/expiry; gate person candidate edges."""
    def __init__(self):
        super().__init__(minimum_iou=.3, max_age_s=.5, max_tracks=64)
        self.rejected_appearance_edges = []
        self.selected_edges = []
        self.created_ids = []

    def update(self, detections, timestamp, image_rgb):
        validate_rgb_image(image_rgb)
        if not math.isfinite(timestamp) or timestamp < 0 or (self.last_time is not None and timestamp <= self.last_time):
            raise ValueError("tracking requires strictly increasing timestamps")
        self.last_time = timestamp
        self.tracks = {k: v for k, v in self.tracks.items() if timestamp - v["time"] <= self.max_age_s}
        indexed = sorted(enumerate(detections), key=lambda p: (-p[1].confidence, p[0]))[:self.max_tracks]
        features = {index: clothing_histogram(image_rgb, det.bbox_xyxy) if det.class_id == 0 else None for index, det in indexed}
        candidates, edge_details = [], {}
        self.rejected_appearance_edges, self.selected_edges = [], []
        for index, det in indexed:
            for tid, old in self.tracks.items():
                overlap = iou(det.bbox_xyxy, old["bbox"]) if det.class_id == old["class_id"] else 0
                if overlap < self.minimum_iou:
                    continue
                similarity = 1.
                if det.class_id == 0:
                    feature, previous = features[index], old["appearance"]
                    similarity = float(np.sqrt(feature * previous).sum()) if feature is not None and previous is not None else 0.
                    if similarity < .65:
                        self.rejected_appearance_edges.append({"detection_index": index, "previous_track_id": tid, "iou": overlap, "similarity": similarity})
                        continue
                candidates.append((-overlap * similarity, index, tid))
                edge_details[index, tid] = {"detection_index": index, "previous_track_id": tid, "iou": overlap, "similarity": similarity}
        assigned, used = {}, set()
        for _, index, tid in sorted(candidates):
            if index not in assigned and tid not in used:
                assigned[index] = tid
                used.add(tid)
                self.selected_edges.append(edge_details[index, tid])
        result = []
        for index, det in indexed:
            tid = assigned.get(index)
            if tid is None:
                if len(self.tracks) >= self.max_tracks:
                    evictable = [(v["time"], k) for k, v in self.tracks.items() if k not in used]
                    if not evictable:
                        continue
                    del self.tracks[min(evictable)[1]]
                tid, self.next_id = self.next_id, self.next_id + 1
                self.created_ids.append(tid)
            used.add(tid)
            self.tracks[tid] = {"bbox": det.bbox_xyxy, "class_id": det.class_id, "time": timestamp, "appearance": features[index]}
            result.append({"track_id": tid, "class_id": det.class_id, "label": det.label,
                           "confidence": float(det.confidence), "bbox_xyxy": [float(x) for x in det.bbox_xyxy]})
        return result


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histories(rows, field):
    found = defaultdict(list)
    for row in rows:
        for item in row["detections"]:
            found[item[field]].append({"sequence": row["sequence"], "time_s": row["capture_time_s"], "label": item["label"],
                                       "bbox_xyxy": item["bbox_xyxy"], "baseline_track_id": item["baseline_track_id"]})
    return {str(key): value for key, value in sorted(found.items())}


def summary_counts(histories_by_id):
    return {"all_track_fragments": len(histories_by_id),
            "track_fragments_by_class": dict(Counter(items[0]["label"] for items in histories_by_id.values())),
            "one_observation_person_fragments": sum(len(items) == 1 and items[0]["label"] == "person" for items in histories_by_id.values())}


def render_examples(output, examples):
    panels = []
    for seq, frame, old, new in examples:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = old["bbox_xyxy"]
        crop = frame[max(0, math.floor(y0) - 10):min(h, math.ceil(y1) + 10), max(0, math.floor(x0) - 10):min(w, math.ceil(x1) + 10)].copy()
        panel = np.full((700, 420, 3), 245, np.uint8)
        scale = min(390 / crop.shape[1], 580 / crop.shape[0])
        resized = cv2.resize(crop, (round(crop.shape[1] * scale), round(crop.shape[0] * scale)))
        left = (420 - resized.shape[1]) // 2
        panel[90:90 + resized.shape[0], left:left + resized.shape[1]] = resized
        cv2.putText(panel, f"Source frame {seq}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (25, 25, 25), 2)
        cv2.putText(panel, f"Old ID108 -> probe ID{new['track_id']}", (15, 62), cv2.FONT_HERSHEY_SIMPLEX, .62, (25, 25, 25), 2)
        panels.append(panel)
    cv2.imwrite(str(output / "known_switch_examples.jpg"), np.concatenate(panels, axis=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/sabana_detail01"))
    parser.add_argument("--video", type=Path, default=Path("inputs/sabana_grande/source.webm"))
    parser.add_argument("--output", type=Path, default=Path("results/appearance_probe01"))
    args = parser.parse_args()
    source_summary = json.loads((args.run / "summary.json").read_text())
    source_log = args.run / "detections.jsonl"
    rows = [json.loads(line) for line in source_log.read_text().splitlines()]
    if [r["sequence"] for r in rows] != list(range(0, len(rows) * 3, 3)) or not all(r["status"] == "ok" for r in rows):
        raise ValueError("Expected complete successful every-third-frame detections")
    video_digest = digest(args.video)
    if source_summary["input_sha256"] != video_digest:
        raise ValueError("Source video hash differs from recorded inference input")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "predeclared_settings.json").write_text(json.dumps(SETTINGS, indent=2) + "\n")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("Video cannot be decoded")
    cv2.setNumThreads(2)
    baseline, probe = ShortTermTracker(), AppearanceProbeTracker()
    saved, examples, edge_rows = [], [], []
    started = time.perf_counter()
    try:
        by_seq = {r["sequence"]: r for r in rows}
        for seq in range(rows[-1]["sequence"] + 1):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Missing source frame {seq}")
            if seq not in by_seq:
                continue
            row = by_seq[seq]
            detections = [Detection(tuple(d["bbox_xyxy"]), d["confidence"], d["class_id"], d["label"]) for d in row["detections"]]
            baseline_result = baseline.update(detections, row["capture_time_s"])
            if [d["track_id"] for d in baseline_result] != [d["track_id"] for d in row["detections"]]:
                raise ValueError(f"Baseline replay does not reproduce recorded IDs at frame {seq}")
            result = probe.update(detections, row["capture_time_s"], cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if len(result) != len(row["detections"]):
                raise ValueError("Replay dropped observations")
            for old, new in zip(row["detections"], result):
                for field in ("class_id", "label", "confidence", "bbox_xyxy"):
                    if old[field] != new[field]:
                        raise ValueError(f"Replay changed detector evidence: {field}")
                new["baseline_track_id"] = old["track_id"]
                if seq in (120, 240, 360) and old["track_id"] == 108:
                    examples.append((seq, frame.copy(), old, new))
            saved.append({"sequence": seq, "capture_time_s": row["capture_time_s"], "mode": "saved_detection_replay", "neural_inference_executed": False,
                          "control_authority": False, "detections": result})
            edge_rows.append({"sequence": seq, "rejected_appearance_edges": probe.rejected_appearance_edges, "selected_edges": probe.selected_edges})
    finally:
        capture.release()
    old_history, new_history = histories(saved, "baseline_track_id"), histories(saved, "track_id")
    assert probe.created_ids == list(range(1, len(probe.created_ids) + 1))
    assert len(probe.created_ids) == len(new_history)
    assert all(b["time_s"] - a["time_s"] <= .5 for track in new_history.values() for a, b in zip(track, track[1:]))
    chain = [{"sequence": row["sequence"], "probe_track_id": det["track_id"], "bbox_xyxy": det["bbox_xyxy"]}
             for row in saved for det in row["detections"] if det["baseline_track_id"] == 108]
    report = {"settings": SETTINGS, "source_video_sha256": video_digest, "saved_detection_log_sha256": digest(source_log),
              "prototype_source_sha256": digest(Path(__file__)), "frames_replayed": len(rows), "observations_preserved": sum(len(r["detections"]) for r in rows),
              "replay_wall_s": time.perf_counter() - started, "baseline": summary_counts(old_history), "appearance_probe": summary_counts(new_history),
              "baseline_reproduced_all_recorded_ids": True, "detection_boxes_scores_classes_unchanged": True,
              "created_ids_strictly_increasing_never_recycled": True, "matched_observation_gaps_all_within_expiry": True,
              "known_switch_examples": [{"sequence": seq, "old_track_id": 108, "probe_track_id": new["track_id"], "bbox_xyxy": new["bbox_xyxy"]}
                                        for seq, _, _, new in examples],
              "old_id108_assignment_sequence": chain,
              "rejected_appearance_candidate_edges": sum(len(r["rejected_appearance_edges"]) for r in edge_rows),
              "limits": ["Prototype only; no production tracker edited", "Color histograms cannot establish identity, distinguish similar clothing, or guarantee continuity",
                         "Settings fixed before replay, but the clip and three failure examples are known development data, not a held-out tracking benchmark",
                         "Track fragment counts are not accuracy or distinct-person counts", "This reuses saved neural detections; no new model inference or flight ran"]}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    for filename, records in (("replayed_detections.jsonl", saved), ("association_edges.jsonl", edge_rows)):
        (args.output / filename).write_text("".join(json.dumps(r) + "\n" for r in records))
    (args.output / "track_histories.json").write_text(json.dumps({"baseline": old_history, "appearance_probe": new_history}, indent=2) + "\n")
    render_examples(args.output, examples)
    print(json.dumps({k: v for k, v in report.items() if k != "old_id108_assignment_sequence"}, indent=2))


if __name__ == "__main__":
    main()
