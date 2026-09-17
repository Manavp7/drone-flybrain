# Studio detour and raw-motion remediation

These local checks address the two unfinished Studio additions. The final frozen
batch passed **five of five controlled cases** with actual YOLO/Flyvis inference
and motor-driven MuJoCo simulation. The earlier failures remain unchanged in
[initial Studio evidence](../mantis_studio/README.md) and this bundle's
[`integration/studio_remediation_probe01`](integration/studio_remediation_probe01/summary.json).
No earlier benchmark is regraded.

## Changes and causal limits

The detour planner previously stopped to inspect a target corridor outside the
current camera view, rediscovered the same obstruction and exhausted its time
budget. It now continues along its independently depth-certified detour corridor
until rejoining is observed clear or the existing early view limit requires
inspection. The 14-second/2-metre attempt bounds, target freshness, command
expiry and final depth guardian are unchanged.

A subsequent probe genuinely rejoined but stopped after one zero-person YOLO
frame. A Studio-only wrapper now permits one extra horizontal-flip inference on
that same frame when primary detection finds no person. It retains confidence,
identity and total deadline checks and never fabricates or relabels detections.
None of the five final trials triggered this fallback, so those passes do not
establish recovery of the original missed frame. A lossy-video reconstruction
also failed to reproduce that miss; compressed video cannot supply the original
pixels. The wrapper's behavior is covered by eighteen focused tests.

Raw motion had two runtime bottlenecks: lazy decoder initialization/repeated
blank-state resets, and rebuilding fixed neural parameter views while retaining
unneeded intermediate states. Both optimizations preserve the official update
at every 0.02-second step. Actual full neural state and decoded flow comparisons
against the original implementation were **bit-exact**. Initialization moves
into model loading, resets clone the verified initial state, and fixed parameter
views are reused. Model weights, decoder, warmup and deadlines are unchanged.

## Frozen final integration results

The cases, thresholds and twenty Mantis source digests were written before any
final trial ran. Selection used one current detection nearest the image centre;
there was no automatic reselection. All five cases had zero contacts and zero
wrong-person evaluation samples. All twenty source hashes still matched after
completion.

| Case | Result | Measured evidence |
| --- | --- | --- |
| Neural detour A | Pass | 24 resumed fresh observations; 0.731 m resumed translation; 0.286 m minimum hull clearance |
| Neural detour B | Pass | 30 resumed fresh observations; 0.909 m resumed translation; 0.259 m minimum hull clearance |
| Direct-YOLO detour | Pass | 30 resumed fresh observations; 0.674 m resumed translation; 0.308 m minimum hull clearance |
| Raw motion: observe | Pass | 27/28 fresh raw observations; 100% steady freshness; zero gap resets |
| Raw motion: brake | Pass | 30/31 fresh raw observations; 100% steady freshness; zero gap resets; 160 same-tick depth-approved reductions |

The detour gate requires the vehicle centre to pass the obstacle's back face
after recorded rejoining, at least three fresh observed following samples and
0.15 m of resumed travel, and at least 0.2 m conservative planar hull clearance.
The source command's capture, sequence, selected ID and release time are checked;
held or predicted selection cannot satisfy the gate. Whole-hull passage of the
obstacle's longitudinal plane is not required; actual hull separation is checked
independently. These are three repeats of one offset fixture, not varied-route
validation.

The motion gate requires at least ten fresh outputs and 80% freshness starting
one second after the first raw capture. The first output in each trial is
warming up. In brake mode, recorded same-tick comparisons show an allowed
0.45 m/s forward request reduced to 0.09 m/s with the unchanged depth gate.
Counterfactual depth evaluation does not control the motors. The bundle keeps
the first 32 timestamped pairs and the total count; it is not a full physics
trace. A brake effect does not demonstrate improved collision avoidance or
tracking accuracy.

## Preserved failed probe

The first new probe failed all three cases. The detour rejoined but lost target
detection before adequate forward resumption. Observe mode produced 5/18 fresh
outputs and 31.25% steady freshness; brake mode produced 14/22 and 70%.
Those original definitions, summaries and scores are copied byte-for-byte.
The final batch uses stronger causal checks and is stored separately.

## Artifacts and verification

- [`integration/`](integration/manifest.json): 29 hashed compact artifacts,
  original failed/final results, model/source provenance and observation
  telemetry. Large neural/depth arrays and videos are omitted.
- [`motion_optimization/`](motion_optimization/README.md): first-call/reset
  equivalence profile across fifteen captures and four resets.
- [`motion_step_optimization/`](motion_step_optimization/README.md): original,
  first and final implementations compared across thirteen captures, including
  longer intervals and gap resets, with portable frozen sources and runner.
- [`video_verification.json`](video_verification.json): all sixteen new camera
  and overview videos decoded fully, covering failed and final trials.
- [`verification.json`](verification.json): software, source-integrity,
  independent review and browser playback checks with scope limits.

From the repository root, these checks require no learned-model execution:

```sh
python -B evidence/mantis_studio_remediation/integration/verify.py
python -B evidence/mantis_studio_remediation/motion_step_optimization/verify.py
```

The integration verifier checks artifact hashes and recomputes all five scores.
It reuses the frozen scorer and recorded clearance/physics summaries; it is not
an independent replay of all physics or learned inference. The native software
suite passed **821 tests** in 58.985 seconds. Separate review checked the
combined code and final recorded evidence. See the
[Studio guide](../../docs/MANTIS_STUDIO.md#reproducible-studio-checks) for a new
actual-model run; elapsed inference times and results can vary by machine.

These optimizations preserve the earlier **negative motion-accuracy benchmark**.
Neither neural guidance nor the raw-camera decoder has established superiority
over the strongest conventional baseline. The world uses stylized actors,
ideal depth/state sensors and a conventional flight controller. General person
identity, general obstacle navigation, guaranteed real-time processing and
physical aircraft readiness remain unestablished.
