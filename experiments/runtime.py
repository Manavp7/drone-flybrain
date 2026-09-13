"""Real Flyvis inference with immutable model artifacts and local runtime caches.

The V3.1 research adapter is retained unchanged. This experimental loader uses
the same inference method but reads plain YAML instead of NetworkView, whose
joblib cache would modify the locked pretrained directory.
"""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from collections.abc import Mapping

from flybrain_sim.research_model import (
    FLYVIS_VERSION, FlyvisShadowAdapter, ResearchModelError,
    dependency_status, validate_manifest,
)


class VideoFlyvisAdapter(FlyvisShadowAdapter):
    def __init__(self, manifest_path: str | Path):
        project = Path(__file__).resolve().parents[1]
        for variable, relative in {
            "MPLCONFIGDIR": ".cache/matplotlib",
            "NUMBA_CACHE_DIR": ".cache/numba",
            "FLYVIS_ROOT_DIR": ".cache/flyvis",
        }.items():
            os.environ.setdefault(variable, str(project / relative))
            Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
        self.locked = validate_manifest(manifest_path)
        status = dependency_status()
        if status["status"] == "dependency_unavailable":
            raise ResearchModelError("dependency_unavailable", str(status["missing"]))
        if status["status"] == "version_mismatch":
            raise ResearchModelError("version_mismatch", f"Expected Flyvis {FLYVIS_VERSION}")

        import torch
        import yaml
        import flyvis
        from flyvis.datasets.rendering import BoxEye

        torch.set_num_threads(2)
        self.torch, self.flyvis, self.BoxEye = torch, flyvis, BoxEye
        self.device = flyvis.device
        # Hash the same immutable bytes that restricted torch.load consumes.
        checkpoint_bytes = self.locked.checkpoint.read_bytes()
        expected = self.locked.manifest["files"][self.locked.manifest["checkpoint"]]
        if hashlib.sha256(checkpoint_bytes).hexdigest() != expected:
            raise ResearchModelError("integrity_mismatch", "Checkpoint changed before loading")
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location=self.device, weights_only=True)
        if not isinstance(payload, Mapping):
            raise ResearchModelError("invalid_checkpoint", "Expected a state mapping")
        weights = self._tensor_state(payload.get("network"), "network")
        metadata_path = self.locked.model_dir / "_meta.yaml"
        metadata_bytes = metadata_path.read_bytes()
        if hashlib.sha256(metadata_bytes).hexdigest() != self.locked.manifest["files"].get("_meta.yaml"):
            raise ResearchModelError("integrity_mismatch", "Model configuration changed")
        metadata = yaml.safe_load(metadata_bytes)
        self.config = metadata["config"]
        self.network = flyvis.Network(**self.config["network"])
        self.network.load_state_dict(weights, strict=True)
        self.network.eval().requires_grad_(False)
        if int(self.network.n_nodes) != 45669:
            raise ResearchModelError("unsupported_model", "Expected the 45,669-neuron pretrained visual network")
        self.eye = BoxEye(extent=15, kernel_size=13)
        self.eye.conv.to(self.device)
        self.runtime = status["packages"]
        self.flow_weights = self._tensor_state(payload.get("decoder", {}).get("flow"), "flow decoder")
        self.decoder = None
        # Initialization must not invalidate the source artifact lock.
        validate_manifest(manifest_path)

    def _tensor_state(self, state, label):
        if not isinstance(state, Mapping) or not state:
            raise ResearchModelError("invalid_checkpoint", f"Missing {label} state")
        for name, value in state.items():
            if (not isinstance(name, str) or not self.torch.is_tensor(value)
                    or not self.torch.isfinite(value).all().item()):
                raise ResearchModelError("invalid_checkpoint", f"Invalid {label} tensor")
        return state

    def decode_flow(self, activity):
        """Return official pretrained optic-flow outputs, not flight commands."""
        import numpy as np
        from flyvis.task.decoder import DecoderGAVP

        if self.decoder is None:
            config = dict(self.config["task"]["decoder"]["flow"])
            if config.pop("type") != "DecoderGAVP":
                raise ResearchModelError("unsupported_model", "Unexpected decoder class")
            self.decoder = DecoderGAVP(self.network.connectome, **config)
            self.decoder.load_state_dict(self.flow_weights, strict=True)
            self.decoder.eval().requires_grad_(False)
        if (not isinstance(activity, np.ndarray) or activity.ndim != 2
                or activity.shape[1] != self.network.n_nodes or not np.isfinite(activity).all()):
            raise ResearchModelError("invalid_input", "Expected finite [time,neurons] activity")
        with self.torch.inference_mode():
            result = self.decoder(self.torch.as_tensor(activity[None], device=self.device)).cpu().numpy()[0]
        if result.shape != (len(activity), 2, 721) or not np.isfinite(result).all():
            raise ResearchModelError("invalid_model_output", "Invalid pretrained flow output")
        return result
