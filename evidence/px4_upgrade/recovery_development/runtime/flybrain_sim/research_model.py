"""Optional, offline Flyvis visual-model adapter. It never emits flight commands.

Imports of torch and flyvis occur only when real checkpoint inference is requested.
There is no synthetic-weight or geometry-proxy fallback. See docs/RESEARCH_MODEL.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import time
from typing import Any, Mapping
from urllib.parse import urlparse


FLYVIS_VERSION = "1.2.0"
FLYVIS_REVISION = "92b3845cc426dd309a1a0e1b3890156c42e14021"
REPOSITORY = "https://github.com/TuragaLab/flyvis"
PUBLICATION = "https://doi.org/10.1038/s41586-024-07939-3"
SCHEMA_VERSION = 1
MAX_CLIP_SECONDS = 10.0
MAX_INPUT_PIXELS = 32_000_000
MAX_MODEL_STEPS = 2000
MAX_FRAME_GAP = 0.25


class ResearchModelError(RuntimeError):
    """Explicit failure that callers must not replace with pretend neural output."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict:
        return {"status": self.code, "reason": str(self), "inference_executed": False,
                "mode": "shadow_only", "control_authority": False}


def _np():
    try:
        return importlib.import_module("numpy")
    except ImportError as exc:
        raise ResearchModelError("dependency_unavailable", "numpy is required for image clips") from exc


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ResearchModelError("invalid_manifest", f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _concrete(value: Any, label: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 2048
            or value.strip().lower() in {"unknown", "pending", "todo", "tbd", "latest"}):
        raise ResearchModelError("invalid_manifest", f"{label} must be a concrete value")
    return value


def _https(value: Any, label: str) -> str:
    value = _concrete(value, label)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ResearchModelError("invalid_manifest", f"{label} requires an HTTPS source URL")
    return value


def _local_member(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ResearchModelError("invalid_manifest", "Model members require POSIX relative paths")
    name = PurePosixPath(relative)
    if name.is_absolute() or any(part in {".", ".."} for part in name.parts):
        raise ResearchModelError("invalid_manifest", "Model members must remain within model_dir")
    # Reject symlinks before resolving; hashing a changing outside target is not a lock.
    candidate = root
    for part in name.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ResearchModelError("invalid_manifest", "Symlinked model members are unsupported")
    resolved = candidate.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ResearchModelError("artifact_unavailable", f"Missing model member: {relative}")
    return resolved


@dataclass(frozen=True)
class LockedModel:
    manifest_path: Path
    model_dir: Path
    checkpoint: Path
    manifest: dict
    manifest_sha256: str


def lock_model(model_dir: str | Path, checkpoint: str, output: str | Path, *,
               weights_source: str, weights_license: str, license_evidence: str) -> LockedModel:
    """Record a user-acquired model's bytes and declarations; this is not authentication.

    Requires an already downloaded official model directory. Does not download,
    infer a weight license from the code license, or claim publisher verification.
    """
    root = Path(model_dir).expanduser().resolve()
    if not root.is_dir():
        raise ResearchModelError("artifact_unavailable", "An existing Flyvis model directory is required")
    checkpoint_path = _local_member(root, checkpoint)
    output_path = Path(output).expanduser().resolve()
    if output_path.is_relative_to(root):
        raise ResearchModelError("invalid_manifest", "Write the manifest outside the model directory")
    _https(weights_source, "weights_source")
    _concrete(weights_license, "weights_license")
    _https(license_evidence, "weights_license_evidence")
    members = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ResearchModelError("invalid_manifest", "Symlinked model members are unsupported")
        if path.is_file():
            members[path.relative_to(root).as_posix()] = file_sha256(path)
    if len(members) > 10_000:
        raise ResearchModelError("invalid_manifest", "Select one model directory, not the complete dataset")
    data = {
        "schema_version": SCHEMA_VERSION, "model_family": "flyvis_visual_dmn",
        "mode": "shadow_only", "source_repository": REPOSITORY,
        "source_revision": FLYVIS_REVISION, "flyvis_version": FLYVIS_VERSION,
        "publication": PUBLICATION, "model_dir": str(root),
        "checkpoint": checkpoint_path.relative_to(root).as_posix(), "files": members,
        "weights_source": weights_source, "weights_license": weights_license,
        "weights_license_evidence": license_evidence,
        "provenance_level": "local_integrity_lock_with_declared_upstream_source",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous locks unless the caller chooses a new path.
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return validate_manifest(output_path)


def validate_manifest(path: str | Path) -> LockedModel:
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise ResearchModelError("artifact_unavailable", "A local model integrity manifest is required")
    if manifest_path.stat().st_size > 4_000_000:
        raise ResearchModelError("invalid_manifest", "Manifest exceeds 4 MB")
    try:
        raw = manifest_path.read_text(encoding="utf-8")
        data = json.loads(raw, object_pairs_hook=_json_object)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ResearchModelError("invalid_manifest", f"Cannot read model manifest: {exc}") from exc
    required = {"schema_version", "model_family", "mode", "source_repository", "source_revision",
                "flyvis_version", "publication", "model_dir", "checkpoint", "files",
                "weights_source", "weights_license", "weights_license_evidence",
                "provenance_level", "created_utc"}
    if not isinstance(data, dict) or set(data) != required:
        raise ResearchModelError("invalid_manifest", "Unexpected or missing manifest fields")
    exact = {"schema_version": SCHEMA_VERSION, "model_family": "flyvis_visual_dmn",
             "mode": "shadow_only", "source_repository": REPOSITORY,
             "source_revision": FLYVIS_REVISION, "flyvis_version": FLYVIS_VERSION,
             "publication": PUBLICATION,
             "provenance_level": "local_integrity_lock_with_declared_upstream_source"}
    if any(type(data[key]) is not type(value) or data[key] != value for key, value in exact.items()):
        raise ResearchModelError("invalid_manifest", "Unsupported model, revision, mode, or schema")
    for field in ("weights_source", "weights_license_evidence"):
        _https(data[field], field)
    _concrete(data["weights_license"], "weights_license")
    try:
        created = datetime.fromisoformat(data["created_utc"])
        if created.tzinfo is None:
            raise ValueError("Timezone required")
    except (TypeError, ValueError) as exc:
        raise ResearchModelError("invalid_manifest", "created_utc requires an ISO timestamp with timezone") from exc
    root_value = _concrete(data["model_dir"], "model_dir")
    root = Path(root_value).expanduser()
    root = (manifest_path.parent / root).resolve() if not root.is_absolute() else root.resolve()
    if not root.is_dir():
        raise ResearchModelError("artifact_unavailable", "model_dir is absent")
    members = data["files"]
    if not isinstance(members, dict) or not 1 <= len(members) <= 10_000:
        raise ResearchModelError("invalid_manifest", "files must lock a nonempty model directory")
    for relative, expected in members.items():
        if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
            raise ResearchModelError("invalid_manifest", "Every model member requires a SHA-256 digest")
        member = _local_member(root, relative)
        if file_sha256(member) != expected:
            raise ResearchModelError("integrity_mismatch", f"Model member changed: {relative}")
    _concrete(data["checkpoint"], "checkpoint")
    if data["checkpoint"] not in members:
        raise ResearchModelError("invalid_manifest", "checkpoint must be included in files")
    # Metadata is part of the model. Reject unrecorded files before NetworkView opens it.
    actual = set()
    for member in root.rglob("*"):
        if member.is_symlink():
            raise ResearchModelError("invalid_manifest", "Symlinked model members are unsupported")
        if member.is_file():
            actual.add(member.relative_to(root).as_posix())
    if actual != set(members):
        raise ResearchModelError("integrity_mismatch", "Unrecorded model files were added")
    return LockedModel(manifest_path, root, _local_member(root, data["checkpoint"]), data,
                       hashlib.sha256(raw.encode()).hexdigest())


def dependency_status() -> dict:
    packages = {}
    for name in ("numpy", "torch", "torchvision", "flyvis", "datamate"):
        try:
            available = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            available = False
        try:
            version = importlib.metadata.version(name) if available else None
        except importlib.metadata.PackageNotFoundError:
            version = None
        packages[name] = {"available": available, "version": version}
    missing = [name for name, item in packages.items() if not item["available"]]
    result = {"mode": "shadow_only", "control_authority": False, "packages": packages,
              "required_flyvis_version": FLYVIS_VERSION, "inference_executed": False,
              "artifact_checked": False, "status": "dependency_unavailable" if missing else "artifact_required",
              "missing": missing}
    if packages["flyvis"]["available"] and packages["flyvis"]["version"] != FLYVIS_VERSION:
        result["status"] = "version_mismatch"
    return result


@dataclass(frozen=True)
class ClipSampling:
    indices: Any
    stimulus_timestamps: Any
    response_timestamps: Any
    dt: float
    input_frame_count: int
    native_median_fps: float


def validate_clip(frames: Any, timestamps: Any, dt: float = 0.01) -> ClipSampling:
    """Validate photometric/timing contracts; zero-order hold uses no future image."""
    np = _np()
    if not isinstance(frames, np.ndarray) or frames.dtype != np.float32 or frames.ndim != 3:
        raise ResearchModelError("invalid_input", "frames must be a numpy float32 array [T,H,W]")
    if min(frames.shape) < 2 or frames.size > MAX_INPUT_PIXELS:
        raise ResearchModelError("invalid_input", "Clip dimensions are empty, too small, or exceed 32M pixels")
    if not np.isfinite(frames).all() or np.min(frames) < 0 or np.max(frames) > 1:
        raise ResearchModelError("invalid_input", "Luminance values must be finite and within [0,1]")
    times = np.asarray(timestamps)
    if times.ndim != 1 or len(times) != len(frames) or times.dtype.kind != "f":
        raise ResearchModelError("invalid_input", "timestamps must be floating-point seconds [T]")
    if not np.isfinite(times).all() or np.any(times < 0):
        raise ResearchModelError("invalid_input", "Timestamps must be finite and nonnegative")
    times = times.astype(np.float64)
    gaps = np.diff(times)
    if np.any(gaps <= 0) or np.max(gaps) > MAX_FRAME_GAP + 1e-9:
        raise ResearchModelError("invalid_input", "Timestamps must increase; frame gaps may not exceed 0.25 s")
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(dt) or not 0.001 <= dt <= 0.02:
        raise ResearchModelError("invalid_input", "Integration dt must be 0.001–0.02 seconds")
    duration = float(times[-1] - times[0])
    steps = int(math.floor(duration / dt + 1e-8))
    if duration > MAX_CLIP_SECONDS + 1e-9 or not 1 <= steps <= MAX_MODEL_STEPS:
        raise ResearchModelError("invalid_input", "Clip exceeds 10 s / 2000 model steps or has no complete step")
    stimulus_times = times[0] + np.arange(steps, dtype=np.float64) * dt
    # The tiny comparison tolerance addresses binary representation of matching timestamps.
    indices = np.searchsorted(times, stimulus_times + 1e-10, side="right") - 1
    return ClipSampling(indices, stimulus_times, stimulus_times + dt, float(dt), len(frames),
                        float(1.0 / np.median(gaps)))


def center_crop_square(frames: Any) -> tuple[Any, dict]:
    """Preserve pixel aspect before BoxEye's square lattice resampling.

    Full photometric/timing validation remains validate_clip's responsibility.
    Cropped peripheral pixels are reported, not silently treated as full FOV.
    """
    np = _np()
    if not isinstance(frames, np.ndarray) or frames.ndim != 3:
        raise ResearchModelError("invalid_input", "Square crop requires [T,H,W] frames")
    height, width = frames.shape[1:]
    side = min(height, width)
    if side < 2:
        raise ResearchModelError("invalid_input", "Image dimensions must be at least 2")
    left, top = (width - side) // 2, (height - side) // 2
    square = np.ascontiguousarray(frames[:, top:top + side, left:left + side])
    return square, {"operation": "center_square_crop_then_isotropic_resize",
                    "source_hw": [height, width], "crop_xywh": [left, top, side, side],
                    "discarded_periphery": height != width, "pixel_aspect_preserved": True}


class FlyvisShadowAdapter:
    """Real-checkpoint pathway; intentionally no update/command/velocity interface.

    Each infer_clip call resets neural state with a one-second contrast fade-in.
    This is offline clip analysis, not a bounded-latency onboard controller.
    """

    def __init__(self, manifest_path: str | Path):
        self.locked = validate_manifest(manifest_path)
        status = dependency_status()
        if status["status"] == "dependency_unavailable":
            raise ResearchModelError("dependency_unavailable", "Missing: " + ", ".join(status["missing"]))
        if status["status"] == "version_mismatch":
            raise ResearchModelError("version_mismatch", f"This adapter targets flyvis=={FLYVIS_VERSION}")
        try:
            self.torch = importlib.import_module("torch")
            self.flyvis = importlib.import_module("flyvis")
            self.BoxEye = importlib.import_module("flyvis.datasets.rendering").BoxEye
            self.device = self.flyvis.device
            # Load only tensor-safe state. We never retry with unrestricted pickle.
            payload = self.torch.load(self.locked.checkpoint, map_location=self.device, weights_only=True)
            if not isinstance(payload, Mapping) or not isinstance(payload.get("network"), Mapping) or not payload["network"]:
                raise ResearchModelError("invalid_checkpoint", "Checkpoint lacks a nonempty 'network' state mapping")
            weights = payload["network"]
            for key, value in weights.items():
                if not isinstance(key, str) or not self.torch.is_tensor(value) or not self.torch.isfinite(value).all().item():
                    raise ResearchModelError("invalid_checkpoint", "Network state contains invalid/nonfinite tensors")
            view = self.flyvis.NetworkView(self.locked.model_dir)
            # Avoid recover_network's warning-only path when weights are absent.
            network = self.flyvis.Network(**view.dir.config.network.to_dict())
            network.load_state_dict(weights, strict=True)
            self.network = network.eval().requires_grad_(False)
            if not 1 <= int(self.network.n_nodes) <= 100_000:
                raise ResearchModelError("unsupported_model", "Unexpected neuron count for the visual DMN")
            self.eye = self.BoxEye(extent=15, kernel_size=13)
            self.eye.conv.to(self.device)
        except ResearchModelError:
            raise
        except Exception as exc:
            raise ResearchModelError("model_initialization_failed", f"Restricted checkpoint initialization failed: {type(exc).__name__}: {exc}") from exc
        self.runtime = status["packages"]

    def infer_clip(self, frames: Any, timestamps: Any, *, dt: float = 0.01) -> tuple[dict, Any]:
        sampling = validate_clip(frames, timestamps, dt)
        np = _np()
        square, spatial_transform = center_crop_square(frames)
        started = time.perf_counter()
        try:
            with self.torch.inference_mode():
                native = self.torch.as_tensor(square[None].copy(), dtype=self.torch.float32, device=self.device)
                target_hw = tuple(int(v) for v in self.eye.min_frame_size.tolist())
                if target_hw != (391, 391):
                    raise ResearchModelError("unsupported_model", "Unexpected BoxEye lattice extent")
                native = self.torch.nn.functional.interpolate(
                    native, size=target_hw, mode="bilinear", align_corners=False, antialias=True)
                spatial_transform["resized_hw"] = list(target_hw)
                spatial_transform["resize"] = "bilinear_align_corners_false_antialias_true"
                rendered = self.eye(native)
                if tuple(rendered.shape) != (1, len(frames), 1, 721):
                    raise ResearchModelError("invalid_model_output", "BoxEye did not produce [1,T,1,721]")
                index = self.torch.as_tensor(sampling.indices, dtype=self.torch.long, device=self.device)
                model_input = rendered.index_select(1, index)
                initial = self.network.fade_in_state(1.0, dt, model_input[:, 0])
                responses = self.network.simulate(model_input, dt, initial_state=initial)
                activity = responses.detach().cpu().numpy()
            expected = (1, len(sampling.indices), int(self.network.n_nodes))
            if activity.shape != expected or not np.isfinite(activity).all():
                raise ResearchModelError("invalid_model_output", "Unexpected or nonfinite neural activity")
        except ResearchModelError:
            raise
        except Exception as exc:
            raise ResearchModelError("inference_failed", f"Flyvis execution failed: {type(exc).__name__}: {exc}") from exc
        elapsed = time.perf_counter() - started
        activities = np.asarray(activity[0], dtype=np.float32)
        summary = {
            "status": "inference_completed", "inference_executed": True, "mode": "shadow_only",
            "control_authority": False, "model_family": "flyvis_visual_dmn", "google_whole_brain": False,
            "manifest_sha256": self.locked.manifest_sha256,
            "checkpoint_sha256": self.locked.manifest["files"][self.locked.manifest["checkpoint"]],
            "provenance_level": self.locked.manifest["provenance_level"],
            "source_revision": FLYVIS_REVISION, "packages": self.runtime, "device": str(self.device),
            "input_shape": list(frames.shape), "input_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "timestamps_sha256": hashlib.sha256(np.asarray(timestamps, dtype=np.float64).tobytes()).hexdigest(),
            "spatial_transform": spatial_transform,
            "native_median_fps": sampling.native_median_fps, "integration_dt_s": dt,
            "temporal_resampling": "causal_zero_order_hold", "fade_in_s": 1.0,
            "state_reset": "each_clip", "photoreceptors": 721,
            "activity_shape": list(activities.shape), "activity_min": float(activities.min()),
            "activity_max": float(activities.max()), "activity_mean": float(activities.mean()),
            "activity_std": float(activities.std()), "elapsed_wall_s": elapsed,
            "simulated_clip_s": float(sampling.response_timestamps[-1] - sampling.stimulus_timestamps[0]),
            "stimulus_timestamps_s": sampling.stimulus_timestamps.tolist(),
            "response_timestamps_s": sampling.response_timestamps.tolist(),
            "limitations": ["No calibrated neural-to-navigation decoder", "No control commands",
                            "Pinhole camera geometry is not a calibrated fly compound eye",
                            "Wall time is a host clip measurement, not a real-time deadline proof",
                            "Local hashes lock declared artifacts; they do not authenticate their publisher"],
        }
        return summary, activities
