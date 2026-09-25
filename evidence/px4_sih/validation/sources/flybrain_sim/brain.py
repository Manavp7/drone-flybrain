"""Simulation-only brain boundary, with honest model provenance.

``bio_proxy`` is an engineered recurrent geometry heuristic. It contains no
Google/Janelia connectome, neural checkpoint, image processing, or fly neurons.
The benchmark factory deliberately excludes the separate Flyvis shadow adapter.
See research_model.py for optional pixel-clip inference with a real checkpoint.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.parse import urlparse

from .contracts import BrainOutput, Observation


VARIANT_METADATA = {
    "baseline": {
        "variant": "baseline",
        "version": "1.0.0",
        "provenance": "engineered_conventional_baseline",
        "description": "No optional brain modulation; common planner remains active.",
        "uses_connectome": False,
        "uses_camera_pixels": False,
        "flight_deployable": False,
    },
    "bio_proxy": {
        "variant": "bio_proxy",
        "version": "1.0.0",
        "provenance": "engineered_geometry_proxy",
        "description": "Three bounded leaky signals from geometric clearance and relative motion.",
        "uses_connectome": False,
        "uses_camera_pixels": False,
        "flight_deployable": False,
    },
}


def _unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _finite_number(value: Any, limit: float = 1e9) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and abs(value) <= limit and math.isfinite(value))


class _BrainBase:
    max_observation_age = 0.8
    variant = "baseline"

    def __init__(self) -> None:
        self._last_now: float | None = None
        self._last_capture: float | None = None
        self._activity = [0.0, 0.0, 0.0]

    def metadata(self) -> dict:
        return deepcopy(VARIANT_METADATA[self.variant])

    def reset(self) -> None:
        """Explicitly start a new stream/episode, including its timestamp epoch."""
        self._activity[:] = [0.0, 0.0, 0.0]
        self._last_now = None
        self._last_capture = None

    def _observation_ok(self, obs: Observation, now: float) -> bool:
        if not obs.valid:
            return False
        if not all(_finite_number(v) for v in (now, obs.capture_time, obs.receive_time)):
            return False
        if not (0 <= obs.capture_time <= obs.receive_time + 1e-8 <= now + 2e-8):
            return False
        if now - obs.capture_time > self.max_observation_age + 1e-8:
            return False
        if self._last_now is not None and now <= self._last_now:
            return False
        if self._last_capture is not None and obs.capture_time <= self._last_capture:
            return False
        if not _finite_number(obs.battery_wh) or obs.battery_wh < 0:
            return False
        for vector in (obs.position, obs.velocity):
            if len(vector) != 3 or not all(_finite_number(v, 1e6) for v in vector):
                return False
        for box in obs.obstacles:
            if any(len(vector) != 3 or not all(_finite_number(v, 1e6) for v in vector)
                   for vector in (box.low, box.high, box.velocity)):
                return False
            if any(low > high for low, high in zip(box.low, box.high)):
                return False
        return True

    def _accept(self, obs: Observation, now: float) -> float | None:
        if not self._observation_ok(obs, now):
            # Reset recurrent activity while retaining the timestamp watermark:
            # replaying a rejected packet must not become valid on the next call.
            self._activity[:] = [0.0, 0.0, 0.0]
            return None
        dt = 0.2 if self._last_now is None else min(now - self._last_now, 2.0)
        self._last_now, self._last_capture = now, obs.capture_time
        return dt

    @staticmethod
    def _rejected() -> BrainOutput:
        return BrainOutput(0.0, 1.0, False, {"input_rejected": 1.0, "state_reset": 1.0})


class BaselineBrain(_BrainBase):
    def update(self, observation: Observation, now: float) -> BrainOutput:
        if self._accept(observation, now) is None:
            return self._rejected()
        return BrainOutput(1.0, 0.0, True, {"optional_modulation": 0.0})


class GeometryProxyBrain(_BrainBase):
    """Recurrent caution modulation for an inspection simulator, not biology.

    The known AABBs and their velocities are privileged geometric inputs.
    They are not an optical-flow estimate obtained from a camera.
    """

    variant = "bio_proxy"

    def update(self, observation: Observation, now: float) -> BrainOutput:
        dt = self._accept(observation, now)
        if dt is None:
            return self._rejected()
        proximity = approach = 0.0
        clearance = 1e6
        for box in observation.obstacles:
            closest = tuple(max(lo, min(p, hi)) for p, lo, hi in
                            zip(observation.position, box.low, box.high))
            delta = tuple(q - p for q, p in zip(closest, observation.position))
            distance = math.hypot(*delta)
            clearance = min(clearance, distance)
            proximity = max(proximity, _unit((3.0 - distance) / 3.0))
            if distance < 1e-9:
                approach = 1.0
                continue
            closing_speed = max(0.0, sum((v - w) * d / distance for v, w, d in
                                        zip(observation.velocity, box.velocity, delta)))
            approach = max(approach, _unit(closing_speed / (distance + 0.25))
                           * _unit((8.0 - distance) / 8.0))
        old_near, old_approach, old_context = self._activity
        targets = (
            _unit(proximity + 0.10 * old_context),
            _unit(approach + 0.10 * old_context),
            max(old_near, old_approach),
        )
        for i, (target, tau) in enumerate(zip(targets, (0.25, 0.20, 0.9))):
            decay = math.exp(-dt / tau)
            self._activity[i] = _unit(decay * self._activity[i] + (1.0 - decay) * target)
        near, moving, context = self._activity
        caution = _unit(max(0.55 * near + 0.45 * moving, 0.70 * context))
        return BrainOutput(
            speed_scale=max(0.35, 1.0 - 0.65 * caution),
            caution=caution,
            healthy=True,
            diagnostics={"geometric_clearance_m": clearance, "proximity_signal": near,
                         "approach_signal": moving, "recurrent_context": context,
                         "geometric_input": 1.0, "actual_connectome": 0.0},
        )


class ExternalModelUnavailable(RuntimeError):
    pass


def make_brain(variant: str) -> _BrainBase:
    if variant == "baseline":
        return BaselineBrain()
    if variant == "bio_proxy":
        return GeometryProxyBrain()
    if variant in {"connectome", "flyvis", "malecns", "external"}:
        raise ExternalModelUnavailable(
            "Flyvis is not an active navigation controller in this benchmark. "
            "Use scripts/research_model_tool.py for the separate real-checkpoint shadow pathway; see docs/RESEARCH_MODEL.md. "
            "Select bio_proxy explicitly for the engineered geometry experiment."
        )
    raise ValueError(f"Unknown brain variant: {variant!r}")


class ManifestValidationError(ValueError):
    """A local research artifact failed integrity or contract checks."""


def _mapping(value: Any, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ManifestValidationError(f"{label} must contain exactly {sorted(keys)}")
    return value


def _string(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 2048
            or value.lower().strip() in {"unknown", "pending", "todo", "tbd", "latest"}):
        raise ManifestValidationError(f"{label} requires a concrete nonempty value")
    return value


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ManifestValidationError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _date(value: Any, label: str) -> date:
    try:
        if not isinstance(value, str) or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is None:
            raise ValueError
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ManifestValidationError(f"{label} must be an ISO date") from exc


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ManifestValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json(raw: bytes, label: str) -> dict:
    def reject_constant(value: str) -> None:
        raise ManifestValidationError(f"Nonfinite JSON number: {value}")
    try:
        result = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ManifestValidationError(f"{label} is not valid JSON") from exc
    if not isinstance(result, dict):
        raise ManifestValidationError(f"{label} must be a JSON object")
    return result


def _verified_artifact(base: Path, value: Any, label: str, limit: int) -> bytes:
    item = _mapping(value, {"path", "sha256"}, label)
    relative = Path(_string(item["path"], f"{label}.path"))
    expected = _digest(item["sha256"], f"{label}.sha256")
    resolved = (base / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not resolved.is_relative_to(base):
        raise ManifestValidationError(f"{label} must be inside the manifest directory")
    try:
        if not resolved.is_file() or resolved.stat().st_size > limit:
            raise ManifestValidationError(f"{label} missing, not regular, or exceeds {limit} bytes")
        with resolved.open("rb") as stream:
            raw = stream.read(limit + 1)
    except OSError as exc:
        raise ManifestValidationError(f"Cannot read {label}") from exc
    if len(raw) > limit or sha256(raw).hexdigest() != expected:
        raise ManifestValidationError(f"{label} checksum/size mismatch")
    return raw


def validate_external_manifest(
    path: str | Path, *, expected_sha256: str, as_of: date | None = None,
) -> dict:
    """Validate a local circuit package without executing or downloading it.

    ``expected_sha256`` must come from an independently reviewed/pinned record.
    A digest calculated from untrusted bytes does not establish authenticity.
    Passing validation establishes structural consistency and declared provenance,
    not biological truth, license permission, model behavior, or flight fitness.
    Artifact limit is 64 MiB each; this is a circuit-package interface and does
    not claim to load an entire connectome with hundreds of millions of edges.
    """
    expected_sha256 = _digest(expected_sha256, "expected_sha256")
    path = Path(path).resolve()
    try:
        if not path.is_file() or path.stat().st_size > 1_048_576:
            raise ManifestValidationError("Manifest missing or exceeds 1 MiB")
        with path.open("rb") as stream:
            raw = stream.read(1_048_577)
    except OSError as exc:
        raise ManifestValidationError("Cannot read manifest") from exc
    if len(raw) > 1_048_576 or sha256(raw).hexdigest() != expected_sha256:
        raise ManifestValidationError("Manifest checksum/size mismatch")
    manifest = _mapping(_read_json(raw, "manifest"), {
        "schema_version", "model_id", "model_version", "provenance", "licenses",
        "review", "graph", "dynamics", "weights", "runtime",
    }, "manifest")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1:
        raise ManifestValidationError("Unsupported manifest schema version")
    for field in ("model_id", "model_version"):
        _string(manifest[field], field)
    provenance = _mapping(manifest["provenance"], {
        "dataset", "dataset_release", "source_url", "source_revision", "retrieved_on",
    }, "provenance")
    for key, value in provenance.items():
        _string(value, f"provenance.{key}")
    try:
        url = urlparse(provenance["source_url"])
    except ValueError as exc:
        raise ManifestValidationError("Malformed provenance URL") from exc
    if url.scheme != "https" or not url.hostname or url.username or url.password:
        raise ManifestValidationError("Provenance requires an HTTPS source reference")
    licenses = _mapping(manifest["licenses"], {"code", "data", "weights"}, "licenses")
    for key, value in licenses.items():
        _string(value, f"licenses.{key}")
    review = _mapping(manifest["review"], {"reviewer", "reviewed_on", "valid_until"}, "review")
    _string(review["reviewer"], "review.reviewer")
    retrieved = _date(provenance["retrieved_on"], "provenance.retrieved_on")
    reviewed = _date(review["reviewed_on"], "review.reviewed_on")
    valid_until = _date(review["valid_until"], "review.valid_until")
    today = date.today() if as_of is None else as_of
    if type(today) is not date:
        raise ManifestValidationError("as_of must be a date")
    if not retrieved <= reviewed <= today <= valid_until:
        raise ManifestValidationError("Review is expired, future-dated, or predates retrieval")
    runtime = _mapping(manifest["runtime"], {
        "framework", "framework_version", "adapter_api", "input_contract", "output_contract",
    }, "runtime")
    for key, value in runtime.items():
        _string(value, f"runtime.{key}")
    if runtime["adapter_api"] != "flybrain-sim-external-v1":
        raise ManifestValidationError("Unsupported adapter contract version")
    graph = _read_json(_verified_artifact(path.parent, manifest["graph"], "graph", 67_108_864), "graph")
    _mapping(graph, {"schema_version", "neuron_ids", "edges", "weight_semantics"}, "graph")
    if type(graph["schema_version"]) is not int or graph["schema_version"] != 1:
        raise ManifestValidationError("Unsupported graph schema version")
    if graph["weight_semantics"] not in {"synapse_count", "signed_effective_weight"}:
        raise ManifestValidationError("Weight semantics must distinguish anatomy from dynamics")
    ids = graph["neuron_ids"]
    if not isinstance(ids, list) or not ids:
        raise ManifestValidationError("neuron_ids must be a nonempty list")
    for neuron_id in ids:
        _string(neuron_id, "neuron_id")
    id_set = set(ids)
    if len(id_set) != len(ids):
        raise ManifestValidationError("Duplicate neuron IDs")
    edges = graph["edges"]
    if not isinstance(edges, list) or not edges:
        raise ManifestValidationError("edges must be a nonempty list")
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        _mapping(edge, {"source", "target", "weight"}, "edge")
        _string(edge["source"], "edge.source")
        _string(edge["target"], "edge.target")
        pair = edge["source"], edge["target"]
        if pair[0] not in id_set or pair[1] not in id_set or pair in seen:
            raise ManifestValidationError("Edge references missing neurons or duplicates an edge")
        seen.add(pair)
        weight = edge["weight"]
        if not _finite_number(weight):
            raise ManifestValidationError("Edge weight must be a finite number")
        if graph["weight_semantics"] == "synapse_count" and (weight < 0 or int(weight) != weight):
            raise ManifestValidationError("Synapse counts must be nonnegative integers")
    # Read only; no pickle, eval, checkpoint loader, dynamic import, or runtime execution.
    for field in ("dynamics", "weights"):
        data = _verified_artifact(path.parent, manifest[field], field, 67_108_864)
        if not data:
            raise ManifestValidationError(f"{field} artifact is empty")
    return {
        "model_id": manifest["model_id"], "model_version": manifest["model_version"],
        "manifest_sha256": expected_sha256,
        "neuron_count": len(ids), "edge_count": len(edges),
        "weight_semantics": graph["weight_semantics"],
        "provenance": deepcopy(provenance), "licenses": deepcopy(licenses),
        "integrity_validated": True, "biological_authenticity_verified": False,
        "runtime_integrated": False, "flight_deployable": False,
    }
