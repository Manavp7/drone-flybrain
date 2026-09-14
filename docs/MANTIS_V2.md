# Mantis V2

Mantis connects actual YOLO person detections, a frozen visual neural model,
registered depth, a conventional stabilizer and MuJoCo's four-motor aircraft.
The underlying visual model is **Flyvis**, credited under its original name.
Mantis is the application/experiment name, not a renamed upstream model.

## New behavior

An animated Cesium Man mesh replaces V1's flat photograph. Its 19 bones drive
real skinned geometry, contributing both RGB and optical depth. The character
has plain clothing, a prescribed gait and an approximate invisible capsule for
contacts. It is stylized and kinematic, not a photorealistic human or human
physics model. See [source and media license](../assets/mantis_actor/ATTRIBUTION.md).

The aircraft remains a free six-DOF body, moved only by four motor forces.
Aircraft physics runs at 200 Hz; actor/depth updates at 20 Hz, with actor pose
refreshed exactly at each primary camera capture. Inference delay advances the
physics clock with the previous unexpired command. A late result cannot act
before completion or extend its capture-anchored expiry.

V1's central-box depth sampler often saw the distant wall through the new
actor's limbs. V2 measures a sufficiently supported, connected foreground depth
layer in the upper central box. It retains the 80% finite-depth and 0.35 m spread
limits and rejects weak, ambiguous, stale or unregistered measurements. A nearer
occluder can be measured; this measurement never establishes person identity.
The independent wide-depth stopping gate remains unchanged.

## What the comparison means

- **Mantis Neural:** YOLO's selected rectangle becomes a visual cue. The actual
  frozen Flyvis network produces neural activity; its existing eight-feature
  readout estimates bearing. No target truth enters this path.
- **Direct YOLO:** calibrated bearing from that same selected box.
- **Alpha-beta:** a fixed causal angular filter on the direct bearing, with
  alpha 0.65 and beta 0.10. It suppresses output during missing observations and
  resets after a measurement gap greater than 0.5 seconds.

Every flight method keeps the same target selection, cue envelope, depth limits,
autopilot, speed limits and stopping gate. Every flight also executes the full
YOLO/neural pipeline, giving all methods the same kind of measured processing
delay. Each controller produces a different camera path, so these flights are
not a matched-image accuracy comparison or an efficiency comparison.

The separate benchmark replays each normal Mantis Neural camera path into all
three estimators. Each sees exactly the same capture timestamps, camera
calibration, detections and imposed corruption. Conditions are clean input,
4-pixel Gaussian box-center jitter with three fixed seeds, and two four-frame
measurement gaps. Off-image jitter is unavailable input for every method.
The frozen model integrates 0.1 seconds per observation on its own neural clock;
the filter uses the actual recorded capture intervals. These clocks are reported
separately and are not presented as matched physical neural dynamics.

Evaluator-only truth is the projected full animated mesh box center. It is
geometric fixture truth, not visible segmentation or a real-human annotation.
Primary accuracy is wrapped capture-time bearing RMSE on the identical subset
where truth and all three predictions exist. Full/eligible coverage and missing
counts remain visible. Completion-time accuracy, statistical superiority and
generalization to new people are not established by these measurements.

## Run and inspect

Install the existing [research runtime and weights](MODEL_SETUP.md). No new
Python dependencies are required. The licensed actor is bundled; checkpoints
remain separate.

```bash
python -u -m experiments.mantis_flight --output results/mantis_run
python -m experiments.mantis_report --run results/mantis_run
python -m experiments.mantis_report --serve results/mantis_run/report --port 8874
```

Open `http://127.0.0.1:8874`. The lab is recorded replay, not a live flight
controller. Its case/controller selectors load separately recorded flights.
Scrubbing shows the latest result available at that simulation time; camera
capture time and current vehicle telemetry are distinct. Neural features are
actual recorded readout features, not a complete brain activity visualization.

For a quick shifted development trajectory:

```bash
python -u -m experiments.mantis_flight --development --case walk \
  --method mantis_neural --duration 4 --skip-benchmark --output results/mantis_probe
```

A single non-neural method requires `--skip-benchmark`, because the benchmark
source is explicitly the Mantis Neural camera path. Outputs refuse overwrites.
The HTTP server supports byte ranges for reliable video seeking and binds only
to localhost. Report export runs no detector or neural inference.

Each run freezes source/asset/readout hashes, scenarios, comparison parameters
and acceptance limits before inference. It saves full motor traces, RGB-D,
neural activity/bindings, controller decisions and scores. `execution_complete`
means every planned computation finished; inspect `checks.json` for operational
passes/failures. A failed experimental gate is retained, not silently relabeled
as a success.

## Scope

The four scenarios are diagonal walking, a turning trajectory, brief target
disappearance/recovery and an obstacle that should cause a stop. Short natural
misses are allowed only within predeclared coverage/gap limits; changing target
ID, long tail loss, missing truth, stale command use or insufficient motion fails.

Obstacle detours, multi-person identity recovery, raw-pixel neural reflexes,
PX4/Gazebo flight control, real cameras and physical aircraft remain future work.
No whole-fly motor brain, YOLO replacement, real-time execution or neural
advantage is claimed.

## Recorded V2 results

The frozen `mantis_run02` completed all twelve scenario/controller combinations
and ten matched-input benchmarks. **Eight of twelve** combinations passed every
operational gate. The complete compact scores are in
[the V2 evidence summary](../evidence/mantis-v2-summary.json).

Mantis Neural passed diagonal walking (**53/57** observed captures, **1.68 m**
travel), turning (**40/42**, **1.41 m**) and brief disappearance/recovery
(**49/52**, **1.66 m**, the same selected ID recovered). All twelve flights had
zero aircraft contact and stayed within their speed/altitude/command limits.

Four failures remain explicit:

- Alpha-beta's walking flight lost the selected target during the final two
  seconds, failing tail tracking/centering coverage.
- All three obstacle flights stopped and retained at least **0.362 m** of
  conservative planar clearance, but failed to demonstrate independent depth
  braking. Target loss suppressed guidance before a positive request was
  blocked by depth. A stop alone does not prove the depth channel caused it.

The flight runs used **720 actual YOLO calls**, **720 neural observations** and
**22,800 motor-physics steps**. The benchmarks made **495 additional neural
observations** using the saved detections. The **114 simulated flight seconds**
took **272.85 seconds** of summed episode wall time, excluding benchmark time.

Mantis Neural did not beat the stronger conventional baseline in any of the
ten matched comparisons. For clean walking, bearing RMSE was **0.312° neural,
0.307° direct and 0.289° filtered**. For clean turning it was **0.482°, 0.275°
and 0.436°**, respectively. Alpha-beta had lower error than neural guidance in
all six jitter runs. This supports keeping the neural path experimental; it
does not establish general superiority of any controller.

The final combined test suite passed **614 tests**, including native rendering.
All twelve exported videos were fully decoded (**1,140 frames**), and their
1,152 timeline samples and ten benchmark displays were checked against saved
records. The frozen source and failed initial batch remain preserved locally.

A separate verifier reconstructed the saved neural readouts and benchmark math,
replayed all **22,800 physics steps** and depth decisions, and checked **834 raw
artifact hashes** with no discrepancies. It did not rerun YOLO or recurrent
Flyvis dynamics, and its native rendering/physics replay reused the frozen
implementations. See the [verification receipt](../evidence/mantis-v2-verification.json)
for exact scope. All twelve flight selectors and ten benchmark selections were
also inspected in the browser. A later narrow-screen layout and waiting-text
fix changed only presentation; the receipt distinguishes it from frozen HTML.

`mantis_run01` stopped after its first successful flight because a NumPy boolean
could not be printed as JSON. The score boundary was fixed and tested; model,
scenario and acceptance parameters were unchanged for `mantis_run02`. Earlier
actor-texture and depth-sampling development failures are retained as well.
