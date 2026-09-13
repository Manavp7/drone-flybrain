"""Paired seeded evaluations with raw episode records and honest denominators."""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import time

from . import __version__
from .brain import make_brain
from .runner import run_episode
from .scenarios import CATEGORIES, generate_scenario

LIMITATIONS = [
    "This navigation benchmark does not execute MaleCNS or pretrained flyvis. Optional visual research inference is evaluated separately; bio_proxy remains an engineered heuristic.",
    "This benchmark uses reduced-order 3D translational dynamics, without six-DOF aerodynamics or PX4. Separate PX4 integration results must not be inferred from these episodes.",
    "Navigation uses geometric position/obstacle observations with modeled errors. The separate pixel renderer and research adapter do not supply navigation estimates; no visual-inertial SLAM is implemented.",
    "Static geometry is a declared prior map; moving obstacles are observed only within the modeled sensor range.",
    "Mission completion means visiting three points with geometric dwell and returning to a hover-height home, not takeoff/landing or image-quality verification.",
    "Dwell is at least 0.6 s elapsed between qualifying truth samples at 0.2 s spacing; motion between samples is not continuously verified.",
    "compute_stall injects unavailable geometric observations while the control process runs; real scheduler stalls and hardware watchdogs are not tested.",
    "Energy is a synthetic analytic surrogate; host wall time is not Jetson performance or real-time flight latency.",
    "The paired variants share scenario seeds and exogenous disturbance schedules; two controller runs are not two independent worlds.",
    "Results apply only to the declared generator and fault distributions; successful simulation does not establish flightworthiness."
]


def code_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total))/denominator
    radius = z*math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denominator
    return [max(0.0, center-radius), min(1.0, center+radius)]


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    mean = lambda key: statistics.fmean(r[key] for r in rows) if n else 0.0
    successful = [r for r in rows if r["mission_complete"]]
    successes = len(successful)
    return {"variant": rows[0]["variant"] if rows else "", "runs": n,
            "mission_successes": successes, "mission_success_rate": successes/n if n else 0.0,
            "mission_success_wilson95": wilson(successes, n),
            "collision_count": sum(bool(r["collision"]) for r in rows),
            "geofence_count": sum(bool(r["geofence_violation"]) for r in rows),
            "returned_home_count": sum(bool(r["returned_home"]) for r in rows),
            "mean_energy_wh": mean("energy_wh"), "mean_duration_s": mean("simulated_seconds"),
            "mean_clearance_m": mean("minimum_clearance_m"), "mean_wall_seconds": mean("wall_seconds"),
            "mean_completed_mission_energy_wh": statistics.fmean(r["energy_wh"] for r in successful) if successful else None,
            "interventions": sum(r["interventions"] for r in rows),
            "outcomes": dict(Counter(r["outcome"] for r in rows))}


def summarize(rows: list[dict], variants: list[str]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["variant"], row["category"])].append(row)
    overall = [aggregate([r for r in rows if r["variant"] == v]) for v in variants]
    category = [{**aggregate(grouped[(v, c)]), "category": c}
                for c in CATEGORIES for v in variants if grouped[(v, c)]]
    paired = {"baseline_only_success": 0, "proxy_only_success": 0, "both_success": 0, "both_failed": 0,
              "success_difference_percentage_points": None, "paired_scenario_count": 0,
              "mean_energy_difference_on_joint_success_wh": None}
    by_seed = defaultdict(dict)
    for r in rows:
        by_seed[r["seed"]][r["variant"]] = r
    energy_diff = []
    for variants_for_seed in by_seed.values():
        if not {"baseline", "bio_proxy"}.issubset(variants_for_seed):
            continue
        base, proxy = variants_for_seed["baseline"], variants_for_seed["bio_proxy"]
        b, p = base["mission_complete"], proxy["mission_complete"]
        key = "both_success" if b and p else "baseline_only_success" if b else "proxy_only_success" if p else "both_failed"
        paired[key] += 1
        paired["paired_scenario_count"] += 1
        if b and p:
            energy_diff.append(proxy["energy_wh"] - base["energy_wh"])
    n = paired["paired_scenario_count"]
    if n:
        paired["success_difference_percentage_points"] = 100*(paired["proxy_only_success"]-paired["baseline_only_success"])/n
    if energy_diff:
        paired["mean_energy_difference_on_joint_success_wh"] = statistics.fmean(energy_diff)
    return {"overall": overall, "by_category": category, "paired": paired}


def _run_seed(task: tuple[int, tuple[str, ...]]) -> list[dict]:
    seed, variants = task
    scenario = generate_scenario(seed)
    rows = []
    for variant in variants:
        row = asdict(run_episode(scenario, variant, record=False))
        row.pop("trajectory")
        row.pop("events")
        rows.append(row)
    return rows


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate(out: Path, scenarios: int, seed_start: int, variants: list[str], workers: int = 1,
             replay_limit: int = 32) -> dict:
    if scenarios < 1 or workers < 1:
        raise ValueError("scenarios and workers must be positive")
    if len(set(variants)) != len(variants) or not variants:
        raise ValueError("variants must be nonempty and unique")
    model_status = {v: make_brain(v).metadata() for v in variants}
    out.mkdir(parents=True, exist_ok=True)
    for name in ("summary.json", "episodes.csv", "episodes.jsonl"):
        if (out / name).exists():
            raise FileExistsError(f"Refusing to overwrite prior results: {out/name}; choose a new --output directory")
    started = datetime.now(timezone.utc).isoformat()
    original_hash = code_hash()
    wall_start = time.perf_counter()
    tasks = ((seed, tuple(variants)) for seed in range(seed_start, seed_start + scenarios))
    rows = []
    pool = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    iterator = pool.map(_run_seed, tasks, chunksize=8) if pool else map(_run_seed, tasks)
    try:
        with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
            for index, batch in enumerate(iterator, 1):
                for row in batch:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                    rows.append(row)
                if index % max(1, scenarios // 20) == 0 or index == scenarios:
                    f.flush()
                    print(f"Scenarios {index}/{scenarios}; episodes {len(rows)}; elapsed {time.perf_counter()-wall_start:.1f}s", flush=True)
    finally:
        if pool:
            pool.shutdown()
    if code_hash() != original_hash:
        raise RuntimeError("Simulation code changed during evaluation; raw records retained, run cannot be certified reproducible")
    with (out / "episodes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": 1, "software_version": __version__, "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_hash": original_hash, "model_status": model_status, "limitations": LIMITATIONS,
        "benchmark": {"scenario_count": scenarios, "episode_count": len(rows), "seed_start": seed_start,
                      "seed_end_inclusive": seed_start+scenarios-1, "paired": len(variants)>1,
                      "variants": variants, "categories": list(CATEGORIES), "dt_seconds": 0.2,
                      "maximum_simulated_seconds": 180, "workers": workers, "started_at": started,
                      "wall_seconds": time.perf_counter()-wall_start, "host_python": platform.python_version(),
                      "host_system": platform.platform(), "sampling": "deterministic balanced category assignment by seed modulo 10"},
        **summarize(rows, variants),
    }
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {"summary": "summary.json", "raw": ["episodes.csv", "episodes.jsonl"],
                                      "source_hash": original_hash, "benchmark": summary["benchmark"],
                                      "scoring": "truth-position 1m inspection dwell>=0.6s at speed<=1.2m/s, home<=1m at speed<=0.6m/s; conservative swept AABB collision radius0.45m"})
    # Select contrasting records deterministically. These are illustrative, not a representative random sample.
    selected = {}
    for c in CATEGORIES:
        for v in variants:
            subset = [r for r in rows if r["category"] == c and r["variant"] == v]
            for kind in (True, False):
                candidate = next((r for r in subset if r["mission_complete"] == kind), None)
                if candidate:
                    selected[(candidate["seed"], v)] = candidate
    for r in rows:
        if r["collision"]:
            selected.setdefault((r["seed"], r["variant"]), r)
            if len(selected) >= replay_limit:
                break
    replays = []
    for (seed, variant), row in list(selected.items())[:replay_limit]:
        scenario = generate_scenario(seed)
        replay_result = asdict(run_episode(scenario, variant, record=True))
        # Rerun must reproduce all functional metrics exactly; wall time is intentionally excluded.
        for key, value in row.items():
            if key != "wall_seconds" and replay_result[key] != value:
                raise RuntimeError(f"Nonreproducible replay {seed}/{variant}: {key}")
        replays.append({"id": f"{seed}-{variant}", "scenario": asdict(scenario), "result": replay_result})
    write_json(out / "replays.json", replays)
    summary["benchmark"]["sampled_replay_count"] = len(replays)
    summary["benchmark"]["replay_selection"] = "first success and failure by category/variant, then collision examples; diagnostic sample"
    write_json(out / "summary.json", summary)
    return summary


def build_report(results: Path, template: Path, destination: Path) -> None:
    summary = json.loads((results/"summary.json").read_text())
    replays = json.loads((results/"replays.json").read_text())
    payload = "window.FLYBRAIN_DATA=" + json.dumps(summary, allow_nan=False) + ";window.FLYBRAIN_REPLAYS=" + json.dumps(replays, allow_nan=False) + ";"
    # Script-closing sequences in any future external string must not terminate embedding.
    payload = payload.replace("</", "<\\/")
    text = template.read_text(encoding="utf-8")
    if text.count("/*__EMBEDDED_DATA__*/") != 1:
        raise ValueError("Dashboard template needs exactly one data marker")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text.replace("/*__EMBEDDED_DATA__*/", payload), encoding="utf-8")
