# Drone FlyBrain

[![Tests](https://github.com/Manavp7/drone-flybrain/actions/workflows/tests.yml/badge.svg)](https://github.com/Manavp7/drone-flybrain/actions/workflows/tests.yml)

An experimental drone simulator combining **YOLO person tracking, a recurrent Flyvis visual neural network, depth-based stopping and four-motor flight physics**.

YOLO locates the target. A target cue drives the actual Flyvis network, whose neural activity feeds a frozen learned readout for visual guidance. A conventional autopilot stabilizes the simulated aircraft. Movement changes the next camera image, closing the loop.

[![Simulation preview](assets/flight-demo.png)](assets/flight-demo.mp4)

[Watch the 80-second demo](assets/flight-demo.mp4) · [Measured results](docs/MOTOR_FLIGHT_TEST.md) · [Model setup](docs/MODEL_SETUP.md) · [Media attribution](assets/ATTRIBUTION.md)

## What works

| Component | Implementation |
|---|---|
| Recognition and temporary target tracking | Official YOLOX-tiny through OpenCV, geometric/clothing association, camera-motion compensation |
| Visual neural processing | Actual frozen Flyvis model: 45,669 neurons; eight readout features from 721 L2 cells |
| Guidance | Neural bearing plus registered target-surface depth for standoff |
| Safety | Wide-depth stopping gate; missing, stale or unsupported observations prevent forward movement |
| Stabilization and physics | Conventional autopilot, four rotor forces, motor lag and MuJoCo six-DOF dynamics |
| Timing | Capture-anchored deadlines; earlier commands remain active while simulated time advances through measured inference delay |

This is a **research simulation**, using a photographed target and ideal depth/state sensors. Flyvis is a visual-network component, not a whole fly motor brain. Physical aircraft, real-time onboard execution, PX4 integration, general aerial person recognition and route planning are not demonstrated.

## Recorded results

The predeclared nine-scenario run passed all gates: left/right approach, moving target, target loss, obstacle stop, missing depth, stale depth, delayed inference and zero-neural-feature ablation.

- **312/312** normal-case observations retained the selected person and passed the approximate photograph-overlap check.
- **732 actual YOLO calls and 732 Flyvis observations**, with 16,000 motor-physics steps.
- All injected failure scenarios stopped; the obstacle case retained at least **0.286 m** conservative planar clearance.
- Zeroing the supplied neural features prevented pursuit while YOLO continued tracking.
- **498 tests passed** in the original native-rendering environment. Independent saved-array reconstruction checked tracking and neural readouts; a separate replay reproduced every motor-physics step.

Eighty simulated seconds took **140.25 seconds** of summed episode wall time. These controlled results establish a connected prototype, not broad tracking accuracy or biological superiority. Earlier pure-Flyvis real-person tracking and overhead detector failures remain unresolved; see [research history](docs/RESEARCH_HISTORY.md).

The public [evidence folder](evidence/README.md) contains a compact results subset. Multi-gigabyte raw captures, checkpoints and original videos are not bundled.

## Install and run tests

Use **Python 3.12**. The development machine was Apple ARM64; CI exercises Linux. The unit suite needs no model weights and performs no model downloads.

```bash
git clone https://github.com/Manavp7/drone-flybrain.git
cd drone-flybrain
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-test.txt
python -m unittest discover -s tests -v
```

The public release adds 17 model-setup checks; all **515 tests passed** in a fresh native-rendering environment. Three camera-rendering tests are skipped by default. With a working native OpenGL context, enable them:

```bash
FLIGHT_RENDER_TESTS=1 python -m unittest discover -s tests -v
```

On a headless Linux machine, native graphics need a suitable display or EGL setup. CI uses a virtual X display. The hardware-free tests check contracts and generated fixtures; they do not repeat trained-model inference.

## Run actual YOLO + Flyvis flight

Install `ffmpeg` separately for the exported video (`ffprobe` is also required by recorded-video tools). Install the optional neural runtime, acquire the upstream model artifacts, and verify them as described in [model setup](docs/MODEL_SETUP.md):

```bash
python -m pip install -r requirements-research.txt
python scripts/fetch_yolox.py --variant tiny --output-dir models/yolox_tiny_official
python scripts/prepare_flyvis.py --archive /path/to/results_pretrained_models.zip
```

The Flyvis checkpoint is not distributed here. Its upstream acquisition and weight terms are separate from this repository's software license. The helper verifies the pinned archive and five selected model files and refuses mismatches or overwrites.

Start with one short development case:

```bash
python -u -m experiments.flight_demo --development --case approach_left --duration 4 --output results/my_smoke
python -m experiments.flight_report --run results/my_smoke
```

A shortened development run is not expected to satisfy the full-duration acceptance gates; inspect its capture, inference and motor outputs. For the complete nine-case suite:

```bash
python -u -m experiments.flight_demo --output results/my_flight_run
python -m experiments.flight_report --run results/my_flight_run
```

Every output directory must be new. Export can resume from completed saved observations without rerunning the models. Processing speed and resulting observation counts depend on your machine because measured inference delay is included in the simulation.

## Repository map

| Path | Purpose |
|---|---|
| `experiments/flight_*.py` | Connected motor-flight fixture, tracking, guidance, safety and reporting |
| `experiments/hybrid_*.py` | Neural readout, target cue and earlier hybrid/video experiments |
| `perception/` | YOLOX, short-term tracking, registered depth and video/image interfaces |
| `flybrain_sim/`, `stress/`, `validation/` | Preserved earlier navigation and evaluation software |
| `integrations/gazebo/` | Observation-only ROS/Gazebo camera bridge; local flight integration remains unverified |
| `tests/` | Unit, numerical, physics and optional rendering checks |
| `results/hybrid_flight_run01/` | Four small, versioned calibration/photo fixtures required by the unchanged experiment |
| `evidence/` | Selected recorded scores and verification receipts |

The earlier experiment tools are retained for research continuity. Some require source clips or raw outputs that are intentionally not included. The quickstart above targets the connected motor-flight experiment.

## License and credits

Project software is licensed under **Apache-2.0**; see [LICENSE](LICENSE) and [NOTICE](NOTICE). The photograph, demo screenshot and video contain **CC BY 3.0** media by Vicente Quintero / QuinteroP, with modifications and attribution documented in [assets/ATTRIBUTION.md](assets/ATTRIBUTION.md).

YOLOX, Flyvis and MuJoCo are upstream projects with their own notices and artifact terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No model-weight redistribution license is implied by this repository's code license.
