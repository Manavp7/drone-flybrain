# Local model setup

The public repository contains source, a portable integrity manifest and the small calibration/photo fixtures. YOLOX and Flyvis pretrained weights are acquired separately and stay outside Git. Unit tests do not need either model.

Use Python 3.12 in an activated virtual environment and run the following commands from the repository root. See the [README](../README.md) for platform, graphics and `ffmpeg` prerequisites.

```bash
python -m pip install -r requirements-research.txt
```

For tests without learned inference, use `requirements-test.txt` instead. Installing Python packages does not download the model artifacts or run a simulation.

## YOLOX-tiny

The motor-flight experiment uses the official YOLOX-tiny ONNX model with 416-pixel input. Acquire it explicitly:

```bash
python scripts/fetch_yolox.py --variant tiny --output-dir models/yolox_tiny_official
```

The helper downloads from the [official YOLOX release](https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0) and writes a local provenance manifest. The flight runtime requires SHA256 `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7`; a different file is rejected before inference. Do not substitute S/M/Nano weights under the Tiny filename.

## Flyvis

Flyvis source is distributed upstream under MIT. A separate redistribution grant for the pretrained archive has not been established by this project; its public availability is not treated as a grant. The repository's Apache-2.0 code license does not license those weights. Review the [upstream model workflow](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/) and applicable terms when acquiring and using them. The weights and archive are not included in this repository.

Acquire `results_pretrained_models.zip` from the upstream source listed in the [pinned official downloader](https://github.com/TuragaLab/flyvis/blob/92b3845cc426dd309a1a0e1b3890156c42e14021/flyvis_cli/download_pretrained_models.py): [official archive download](https://drive.google.com/uc?export=download&id=13cJr2nMn89j-jBAd5RduYRJpBcXwoNrC). The expected archive is 3,417,042 bytes, SHA256:

```text
71c78d4070556a536b13b23ee3139cd2788aa2a9d07d430a223b4edead281db1
```

Keep that ZIP outside the checkout, then provide its local path explicitly:

```bash
python scripts/prepare_flyvis.py --archive /path/to/results_pretrained_models.zip
```

This standard-library helper makes no network requests and performs no inference. It verifies the full archive digest, checks ZIP names/types and bounded sizes, selects only `results/flow/0000/000`, and checks every selected byte against the portable manifest. The official archive contains regular files; symlinks, special files, duplicate paths and traversal paths are rejected. It does not call `extractall`.

The output is exactly `models/flyvis_0000_000/` with these five regular files:

```text
_meta.yaml
best_chkpt
chkpts/chkpt_00000
validation/loss.h5
validation_loss.h5
```

The manifest remains unchanged. Input is fully verified and staged before the destination is claimed exclusively; an installation failure cleans up the newly created destination. An existing target, including an empty directory or dangling symlink, is never overwritten. If setup has already succeeded, skip the installation command. A checksum mismatch requires obtaining the pinned archive again; do not weaken the integrity check or edit model hashes to accommodate another archive.

Verify an existing installation without loading the network:

```bash
python - <<'PY'
from flybrain_sim.research_model import validate_manifest
locked = validate_manifest('models/flyvis_0000_000.manifest.json')
print('Verified', len(locked.manifest['files']), 'model files')
PY
```

## Run the connected experiment

After the local dependencies and both models are ready, a short smoke run is:

```bash
python -m experiments.flight_demo --development --case approach_left --duration 4 --output results/flight_smoke
```

For the full nine-case simulation and preview:

```bash
python -m experiments.flight_demo --output results/flight_new
python -m experiments.flight_report --run results/flight_new
```

Every output directory must be new. Rendering requires a working native OpenGL context; preview export requires `ffmpeg`. The research runtime uses CPU threads explicitly, and measured computation time affects the simulated observation delay, so another machine may produce different observation counts and outcomes. This is an offline simulation with a photograph fixture and ideal sensors; it does not establish real-time or physical flight readiness.
