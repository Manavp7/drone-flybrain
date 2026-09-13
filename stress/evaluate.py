"""Reproducible fixed-controller campaign; every trial retains its entire trace."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import statistics
import time

from .course import PROFILES, course_manifest, make_trial, validate_course
from .runner import run_trial

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONTROLLER = "14d36d261083b17b9e5b67e063b29532dcf76e73b6823867fd1f31f36edafcba"
LIMITATIONS = [
    "Reduced-order geometric simulator; no PX4, Gazebo, real flight, or hardware validation.",
    "Conventional V3 controller. Actual Flyvis/research-model inference is NOT RUN; no camera images are used.",
    "The hard suite repeats one fixed course; the separate holdout suite varies layouts. Neither establishes real-world reliability.",
    "All static obstacles are known in advance; dynamic objects use 6 m range and centerline occlusion.",
    "Compute-stall profile invalidates observations; it does not actually suspend the controller process.",
    "Collision uses a conservative radius-expanded motion AABB; this can flag corner proximity before spherical contact.",
    "Timed faults may not be reached before an early abort. Exposure is reported separately.",
    "Host wall timings are not evidence of onboard real-time capability. Trace floats are rounded to six decimals after scoring.",
    "Static witness proves geometric passability, not dynamic feasibility, battery sufficiency, or controller competence.",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def package_hash(directory):
    h = hashlib.sha256()
    for path in sorted((ROOT / directory).glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _one(job):
    trial_id, seed, profile, directory, controller_hash, suite = job
    if suite == "holdout":
        from validation.holdout import make_trial as factory, obstacles_at_trial
        scenario = factory(seed, profile)
        run = run_trial(scenario, profile, record=True, obstacle_function=obstacles_at_trial)
    else:
        scenario = make_trial(seed, profile)
        run = run_trial(scenario, profile, record=True)
    trace = run["trace"]
    timed_observations = [s for s in trace if s["observation"] is not None]
    window_samples = sum(bool(s["observation"]["visibility_and_fault_metadata"]["in_fault_window"])
                         for s in timed_observations)
    dynamic_samples = sum(bool(s["observation"]["observed_dynamic_obstacles"]) for s in timed_observations)
    metadata = {"trial_id": trial_id, "profile": profile, "seed": seed,
                "scenario": asdict(scenario), "controller_hash": controller_hash, "suite": suite}
    full = {**metadata, **run}
    relative = f"traces/trial_{trial_id:04d}_seed_{seed}.json.gz"
    path = Path(directory) / relative
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as stream:
            stream.write(json.dumps(full, separators=(",", ":"), allow_nan=False).encode())
        raw.flush()
        os.fsync(raw.fileno())
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    d = run["diagnostics"]
    row = {"trial_id": trial_id, "profile": profile, "suite": suite, **run["result"],
            "final_battery_wh": d["remaining_battery_wh"], "max_speed_mps": d["peak_speed_m_s"],
            "max_acceleration_mps2": d["peak_acceleration_m_s2"],
            "abort_reason": d["controller_abort_reason"], "terminal_stage": d["terminal_stage"],
            "fault_window_exposed": window_samples > 0, "fault_window_observation_samples": window_samples,
            "timed_fault_exposed": bool(d["fault_seconds"]),
            "dynamic_observed": dynamic_samples > 0, "dynamic_observation_samples": dynamic_samples,
            "trace_file": relative, "trace_sha256": digest, "trace_bytes": path.stat().st_size,
            "diagnostics": d}
    dump(Path(directory) / "receipts" / f"trial_{trial_id:04d}.json", row)
    return row


def quantile(values, q):
    a = sorted(values)
    index = q * (len(a) - 1)
    lo, hi = math.floor(index), math.ceil(index)
    return a[lo] + (a[hi] - a[lo]) * (index - lo)


def describe(values):
    return {"mean": statistics.mean(values), "median": statistics.median(values),
            "p05": quantile(values, .05), "p95": quantile(values, .95), "min": min(values), "max": max(values)}


def aggregate(rows):
    aliases = {"duration_s": "simulated_seconds", "energy_used_wh": "energy_wh",
               "min_clearance_m": "minimum_clearance_m", "path_length_m": "distance_m",
               "max_speed_mps": "max_speed_mps", "interventions": "interventions", "wall_time_s": "wall_seconds"}
    return {"runs": len(rows), "successes": sum(r["mission_complete"] for r in rows),
            "collisions": sum(r["collision"] for r in rows),
            "geofence": sum(r["geofence_violation"] for r in rows),
            "returned_home": sum(r["returned_home"] for r in rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows)),
            "abort_reasons": dict(Counter(r["abort_reason"] or "none" for r in rows)),
            "waypoints_completed": dict(Counter(str(r["waypoints_completed"]) for r in rows)),
            "fault_window_exposed": sum(r["fault_window_exposed"] for r in rows),
            "timed_fault_exposed": sum(r["timed_fault_exposed"] for r in rows),
            "dynamic_observed": sum(r["dynamic_observed"] for r in rows),
            "trace_samples": sum(r["diagnostics"]["trace_samples"] for r in rows),
            "total_simulated_seconds": sum(r["simulated_seconds"] for r in rows),
            "contact_obstacles": dict(Counter(x for r in rows for x in r["diagnostics"]["contact_obstacle_ids"])),
            "metrics": {name: describe([r[key] for r in rows]) for name, key in aliases.items()}}


def write_tables(directory, rows):
    with (directory / "all_trials.jsonl").open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    csv_rows = []
    for row in rows:
        item = {k: v for k, v in row.items() if k != "diagnostics"}
        for key, value in row["diagnostics"].items():
            item[f"diagnostics.{key}"] = (json.dumps(value, separators=(",", ":"))
                                                  if isinstance(value, (dict, list, tuple)) else value)
        csv_rows.append(item)
    with (directory / "all_trials.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)


def journal_matches(journal_rows, rows):
    """Compare complete records after the same JSON tuple/list normalization."""
    normalized = json.loads(json.dumps(rows, allow_nan=False))
    ids = [r["trial_id"] for r in journal_rows]
    if len(ids) != len(set(ids)):
        return False
    return sorted(journal_rows, key=lambda row: row["trial_id"]) == sorted(normalized, key=lambda row: row["trial_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed-start", type=int, default=1200000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path("results/v31_hard_2000"))
    parser.add_argument("--suite", choices=("hard", "holdout"), default="hard")
    parser.add_argument("--development", action="store_true", help="Run a labeled development campaign without final controller lock")
    args = parser.parse_args()
    if args.trials < 1 or args.workers < 1:
        parser.error("positive trials and workers required")
    controller_hash = package_hash("flybrain_sim")
    if not args.development:
        lock = json.loads((ROOT / "controller_lock.json").read_text())
        if controller_hash != lock["sha256"]:
            raise RuntimeError("Frozen controller mismatch")
    if args.suite == "holdout":
        from validation.holdout import validation_manifest, course_certificate
        manifest = validation_manifest(seed_start=args.seed_start, trials=args.trials)
        certificates = [course_certificate(args.seed_start + i) for i in range(args.trials)]
        certificate = {"valid": all(c["all_segments_clear"] for c in certificates),
                       "static_geometry_sha256": hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                       "course_certificates": certificates}
    else:
        certificate, manifest = validate_course(), course_manifest()
    if not certificate["valid"]:
        raise RuntimeError("Course witness failed")
    directory = args.output.resolve()
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite a campaign: {directory}")
    directory.mkdir(parents=True)
    (directory / "traces").mkdir()
    (directory / "receipts").mkdir()
    jobs = [(i + 1, args.seed_start + i, PROFILES[i % len(PROFILES)], str(directory), controller_hash, args.suite) for i in range(args.trials)]
    protocol = {"title": "Navigation V3 — " + args.suite + " evaluation", "created_utc": utc_now(),
                "suite": args.suite, "development": args.development,
                "recording_policy": "immutable_per_trial_receipts",
                "trials": args.trials, "seed_start": args.seed_start, "seed_end": args.seed_start + args.trials - 1,
                "assignment": "round-robin profile by zero-based trial index modulo 10", "profiles": list(PROFILES),
                "controller_hash": controller_hash, "harness_hash": package_hash("stress"),
                "validation_hash": package_hash("validation") if args.suite == "holdout" else None,
                "baseline_controller_hash": BASELINE_CONTROLLER,
                "hash_method": "SHA256 of sorted top-level package .py filename bytes followed by file bytes",
                "course_hash": certificate["static_geometry_sha256"],
                "controller_frozen": not args.development, "controller_variant": "baseline (V3 conventional controller)",
                "tuning_during_campaign": False, "pilots_included": False,
                "python": platform.python_version(), "platform": platform.platform(), "workers": args.workers,
                "simulation_dt_s": .2, "deadline_s": 300, "body_radius_m": .45,
                "inspection_rule": "Truth within 1 m and speed <=1.2 m/s for >=0.6 s; unordered original score preserved",
                "home_rule": "Truth within 1 m and speed <=0.6 m/s, no collision/geofence",
                "full_success_rule": "All three independent truth inspections plus controller completion and verified return",
                "trace_policy": "Every control sample and final integration endpoint; six decimal places after scoring",
                "failure_policy": "All completed trials retained; harness exceptions invalidate the campaign and are recorded",
                "limitations": LIMITATIONS, "trial_plan": [{"trial_id": j[0], "seed": j[1], "profile": j[2]} for j in jobs]}
    dump(directory / "protocol.json", protocol)
    dump(directory / "course.json", manifest)
    dump(directory / "course_certificate.json", certificate)
    started = time.perf_counter()
    rows = []
    print(f"START {args.trials} trials; {args.workers} workers; output={directory}", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            iterator = iter(jobs)
            pending = {}
            for _ in range(min(args.workers * 2, len(jobs))):
                job = next(iterator)
                pending[pool.submit(_one, job)] = job
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future)
                    row = future.result()
                    rows.append(row)
                    if len(rows) % 50 == 0 or len(rows) == len(jobs):
                        print(json.dumps({"completed": len(rows), "planned": len(jobs),
                                          "elapsed_s": round(time.perf_counter() - started, 1),
                                          "outcomes": dict(Counter(r["outcome"] for r in rows))}), flush=True)
                    job = next(iterator, None)
                    if job is not None:
                        pending[pool.submit(_one, job)] = job
    except BaseException as exc:
        dump(directory / "CAMPAIGN_INVALID.json", {"error": repr(exc), "completed": len(rows), "utc": utc_now()})
        raise
    if (package_hash("flybrain_sim") != controller_hash or package_hash("stress") != protocol["harness_hash"]
            or (args.suite == "holdout" and package_hash("validation") != protocol["validation_hash"])):
        dump(directory / "CAMPAIGN_INVALID.json", {"error": "source changed during campaign", "utc": utc_now()})
        raise RuntimeError("Source changed during campaign")
    receipt_paths = sorted((directory / "receipts").glob("trial_*.json"))
    receipt_rows = [json.loads(path.read_text()) for path in receipt_paths]
    if not journal_matches(receipt_rows, rows) or len(receipt_rows) != args.trials:
        dump(directory / "CAMPAIGN_INVALID.json", {"error": "immutable receipts do not match all results", "utc": utc_now()})
        raise RuntimeError("Immutable receipts do not match all results")
    # Final tables are derived from independent closed-file receipts, not a
    # long-lived append stream. Every receipt's paired trace is checked here.
    for row in receipt_rows:
        path = directory / row["trace_file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["trace_sha256"]:
            dump(directory / "CAMPAIGN_INVALID.json", {"error": "trace receipt hash mismatch", "trial_id": row["trial_id"]})
            raise RuntimeError("Trace receipt hash mismatch")
    rows = receipt_rows
    rows.sort(key=lambda row: row["trial_id"])
    write_tables(directory, rows)
    summary = aggregate(rows)
    summary.update({"campaign_wall_seconds": time.perf_counter() - started, "completed_utc": utc_now(),
                    "trace_bytes": sum(r["trace_bytes"] for r in rows)})
    by_profile = [{"profile": profile, **aggregate([r for r in rows if r["profile"] == profile])}
                  for profile in PROFILES if any(r["profile"] == profile for r in rows)]
    dump(directory / "summary.json", summary)
    dump(directory / "by_profile.json", by_profile)
    dump(directory / "COMPLETE.json", {"rows": len(rows), "completed_utc": utc_now(),
                                      "controller_hash": controller_hash, "harness_hash": protocol["harness_hash"],
                                      "validation_hash": protocol["validation_hash"],
                                      "recording_policy": protocol["recording_policy"], "receipts": len(receipt_rows)})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
