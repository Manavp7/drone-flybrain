# Actual PX4 control in local simulation

Mantis can run the actual PX4 autopilot and its official Simulator in Hardware
(SIH) **as software-only SITL**. No flight controller or camera is needed. This
is available from the browser Studio and as bounded command-line experiments.
Studio keeps its original MuJoCo backend as the default. The PX4 backend is
simulation-only and owns the local autopilot process it starts.

The loop is:

```text
PX4 controllers → SIH dynamics → simulated camera pixels
   ↑                                  ↓
bounded velocity/yaw ← depth gate ← YOLO → Flyvis cue → neural readout
```

YOLO identifies people and maintains a temporary visual track. An explicit
browser selection or `--select-track` chooses an observed track. With neural
guidance, its target cue goes through the real frozen 45,669-neuron Flyvis visual
network and existing Mantis readout. CLI missions can instead use direct or
filtered guidance through the same tracking and depth-safety path.
PX4 handles estimation, attitude/rate control, actuator mixing, altitude and
landing. Flyvis is not a complete biological motor brain.

## Run

Tested on an Apple Silicon Mac with Python 3.12, Apple Command Line Tools and
ffmpeg. Install the [existing model artifacts](MODEL_SETUP.md) first. From the
repository directory:

```bash
python3.12 -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-research.txt \
  -r integrations/px4/requirements-sih.txt
.venv-test/bin/python -B scripts/build_px4_sih.py

# Explicit reduced-noise takeoff/position-hold/landing, without model inference:
.venv-test/bin/python -B -m experiments.px4_follow --smoke --low-noise-sensors \
  --hover-seconds 20 --output results/px4_smoke_new

# Fixed two-person fixture; explicit 1% noise; select observed Track 1 once:
.venv-test/bin/python -B -m experiments.px4_follow --low-noise-sensors \
  --select-track 1 --output results/px4_low_noise_neural_new
```

Every output directory must be new. Each command stops its owned simulator at
the end. Model loading and graphics warmup happen before arming. A first cold
model load took approximately two minutes; subsequent startup was much shorter.
These commands run acceptance checks; they do not imply a successful result.
The current strict validation status is described below.

## Browser controls

```bash
.venv-test/bin/python -B -m experiments.mantis_studio
```

Open [Mantis Studio](http://127.0.0.1:8875), choose **PX4 / SIH autopilot**, and
start a run. By default this uses the experimental `sih_velocity_settled`
estimator profile with full sensor noise; reliable full-noise flight is still
unresolved. The **1% sensor-noise diagnostic** checkbox explicitly selects the
reduced-noise diagnostic with the stock estimator profile instead. The active
backend and sensor mode remain visible.

After takeoff, click a detected person in the camera or its track button. The
aircraft holds until selection is accepted. The interface shows pending,
accepted or rejected selection status, current guidance/depth decisions, the
camera, overview, estimated speed and neural features. **Land & save** requests
LAND and lets the owned runner confirm landing/disarm and save its recording.
Stopping the Studio server uses the same cooperative cleanup path.

PX4 Studio currently fixes the scene to stationary people, neural guidance, a
3.5 m standoff and compact recording. Duration and recording budget can be set
before starting. Pause, live follow-distance/motion edits, detours, raw-camera
motion experiments and alternative guidance methods are disabled for this
backend; unsupported API settings are also rejected. CLI mission experiments
provide the additional trajectories and methods below. The original MuJoCo
controls remain available on their own backend.

A click refers to the exact displayed frame. The server admits intent only from
its last eight published frames, each at most three seconds old. The perception
worker must then find that same track in a new observation within the unchanged
0.65-second freshness limit, with uninterrupted visibility, no ambiguity and
the original appearance match. A cached click supplies identity intent, never
old geometry or command authority. New selections revoke the previous command
immediately; superseded selection receipts cannot authorize motion. Studio
results describe an operator demonstration, not a fixed-identity benchmark:
explicit switches and an operator's stop are allowed, and no tracking-success
score is inferred from them.

## CLI missions and comparison

`--mission` runs a declared 2–120-second following window. Choose
`--method neural`, `direct` or `filtered`, and `--trajectory stationary`, `walk`,
`crossing` or `occlusion`. Scenario phase begins after stable takeoff, using SIH
source time rather than model-loading time. The selected observed track remains
fixed; a loss or ambiguous association does not trigger automatic replacement.

```bash
# Diagnostic walking mission with one real 1-second inference delay:
.venv-test/bin/python -B -m experiments.px4_follow --mission --low-noise-sensors \
  --method neural --trajectory walk --target-speed .08 --duration 60 --faults \
  --select-track 1 --output results/px4_walk_new

# Stationary association recovery during a small commanded yaw turn:
.venv-test/bin/python -B -m experiments.px4_follow --mission --low-noise-sensors \
  --method neural --trajectory stationary --target-speed 0 --duration 30 --recovery \
  --select-track 1 --output results/px4_recovery_new

# Experimental full-noise estimator profile; no sensor-noise reduction:
.venv-test/bin/python -B -m experiments.px4_follow --smoke \
  --estimator-profile sih_velocity_settled --hover-seconds 20 \
  --output results/px4_full_noise_hover_new

# Serial comparison: three scenes × three methods × two repeats:
.venv-test/bin/python -B scripts/run_px4_missions.py --low-noise-sensors \
  --estimator-profile stock --duration 60 --repeats 2 \
  --output results/px4_comparison_new
```

The batch saves its source hashes, parameters, order and stopping rules before
launching anything. Every method uses the same declared scene, sensor profile,
speed, duration and delayed-inference fault. Each controller creates its own
closed-loop observations; equal settings do not mean identical camera paths or
noise samples. The batch preserves failed and skipped cases, reverses method
order on alternating repeats, and produces descriptive paired comparisons.
It never automatically declares neural superiority.

Runs are serialized. The batch stops after two consecutive failed takeoffs,
two consecutive source-clock freshness failures, unconfirmed landing, a cleanup
error or a source/configuration mismatch. Its
512 MiB storage policy reserves 64 MiB before admitting each new run and never
deletes evidence. Videos are capped per run; logs, JSON and source copies are
not hard-capped within an admitted run, so this is an admission budget rather
than a guaranteed whole-directory size ceiling. Existing output folders are
refused; there is no implicit retry or resume.

## Pinned build

The build downloads PX4 v1.16.2 at
`54f0455ffcd755534539a7cf33a09a20bf71d29d` and only required submodules into
ignored `.cache/`. The board overlay disables external Gazebo/ROS/DDS dependencies
and uses SIH's stock lockstep loop, paced toward real time. A recorded
compatibility patch supplies the oldest constituent sample timestamp in
`HIL_STATE_QUATERNION`, which this PX4 release otherwise leaves zero. Four
additional, default-off patches provide the sensor diagnostic described below.
The GPS patch also supports explicitly selected accuracy-reporting calibration;
that option changes reported accuracy fields without changing sensor samples.
Build-local Apple SDK headers and an explicit C++ extension-warning compatibility
flag handle the tested compiler. No global compiler configuration is changed.
The binary, startup tree, patches and profile are hashed; unexpected tracked
source or submodule edits stop the build.

## Sensor and estimator profiles

**The default uses stock PX4 sensor noise.** `--low-noise-sensors` explicitly
enables a diagnostic that scales generated measurement noise to 1% of its stock
amplitude in four pinned source files:

| Source module | Noise scaled to 1% in the diagnostic |
| --- | --- |
| `sensor_gps_sim/SensorGpsSim.cpp` | GPS position, altitude and velocity noise |
| `sensor_baro_sim/SensorBaroSim.cpp` | Barometric measurement noise |
| `sensor_mag_sim/SensorMagSim.cpp` | Magnetometer measurement noise |
| `simulator_sih/sih.cpp` | Accelerometer and gyroscope measurement noise |

The patches retain the random draws and multiply only their measurement-noise
contributions by 0.01 when the mode is enabled; otherwise the multiplier is one.
PX4's EKF, controllers, actuator handling and SIH vehicle dynamics remain in use
and their algorithms are unchanged. PX4's freshness and constant-value sensor
checks also remain active. This is not estimator ground-truth injection, and it
does not remove other sensor timing, quantization or modelling limits.
Other generated measurements, such as SIH airspeed, are outside these patches.

The launcher clears inherited `MANTIS_SIH_*` values, including
`MANTIS_SIH_LOW_NOISE_SENSORS`, along with inherited `PX4_*` overrides before
creating its child. Only the explicit `--low-noise-sensors` option sets that
switch for the run. Its sensor profile is
recorded in provenance and the simulator-verification event; the build receipt
records the original and patched hashes of all four files. The diagnostic is
available with `--smoke` as well as with person following.

RGB and depth remain ideal rendered camera measurements in **both** profiles.
Camera pose metadata and control state still come from PX4 estimates. A success
with reduced measurement noise would establish only that declared diagnostic
configuration, not robustness with realistic sensor noise.

The CLI defaults to `--estimator-profile stock`. The named
`sih_velocity_settled` experiment keeps stock measurement-noise amplitudes but
adjusts the estimator configuration for this SIH model: barometric height
reference, 0.10 m barometer noise, no modelled ground effect, accelerometer and
gyro noise settings of 1.0 and 0.10, and GPS position/velocity settings of 0.20 m
and 0.11 m/s. GPS accuracy metadata is reported as 0.20 m horizontal, 0.50 m
vertical and 0.11 m/s velocity, rather than the upstream defaults. The velocity
setting accounts for the EKF's vertical scaling; it is a research calibration,
not a claim that all GPS axes have the same generated noise. The profile
requires ten seconds of GPS readiness and an additional ten-second healthy
settling interval before takeoff. Exact parameters and the selected sensor
mode are recorded independently in provenance.

Intermediate `baro`, `baro_settled`, `sih_covariance`, `sih_calibrated` and
`sih_calibrated_settled` profiles remain available to reproduce the investigation.
These options change neither SIH dynamics nor the generated measurement samples
unless the separate `--low-noise-sensors` flag is supplied. They do not bypass
PX4's estimator, arming, freshness or sensor-validator checks.

The SIH fixture also has a narrowly scoped stationary-person association helper.
If a currently matched person lacks coherent depth for one frame, it can retain
the previous observed world corners for association at the next camera pose.
Their original capture timestamp is preserved, with a 1.2-second maximum
capture separation. A second depth gap, absent detection, registration loss,
clock regression/reset or expiry clears them. New detections still require the
existing overlap and clothing checks. Current missing-depth guidance stays
invalid, and selection never switches automatically. The shared Studio/base
tracker keeps its original missing-depth policy. This helper makes no claim
about recovering moving people; packets record its retention/reprojection use.

## Current upgrade evidence

Full-noise reliability remains unresolved. One experimental calibrated run
completed a 20-second zero-velocity hold with residual drift; a later normal-noise
mission failed the stable-takeoff gate and landed. That earlier velocity-hold
result is not a position-hold qualification. The current smoke runner now holds
the takeoff position, and reports estimated position error and estimated/truth
speed separately from its takeoff/landing lifecycle gates.

The reduced-noise development run `px4_recovery_diagnostic01` completed its
30-second mission with 107 observations, **93.6517% correct-identity time
coverage** and **zero wrong-person observations**. Its original summary passed.
A separately labelled strict retrospective audit of the unchanged saved
observations also passed: the original retained anchor was 0.496530 seconds old
at the restored capture, the measured turn was 0.032072 rad, and the fresh
same-person recovery command was released 0.549347 seconds after the depth-gap
capture. All nine gap sends requested and sent zero forward speed. That audit
is not a rerun or a replacement of the original
summary; it checks retained-anchor reprojection and freshness as well as the
earlier recovery gates.

The later frozen `px4_recovery_final01` trial failed a source-clock freshness
check 3.5605 seconds into its mission. Three native Studio trials successfully
displayed live camera images, but each aborted on source-clock freshness before
person selection. Browser selection, following and operator-requested landing
therefore do **not** have complete native browser acceptance. The implemented
integration and focused tests do not turn those failures into passes.

The frozen comparison declared 18 cases, each with a 60-second following
window. It attempted two walking cases: neural guidance failed on source-clock
freshness after 17.0386 seconds, and direct guidance after 4.5748 seconds. Both
landed/disarmed. The predeclared two-consecutive-clock-failures rule then marked
the other 16 cases skipped. Neither attempted run reached its planned midpoint
inference delay; crossings, occlusion and repeats were not exercised. There are
zero complete three-method pairs and no valid performance comparison from these
unequal, truncated windows.

The final combined native software suite passed **966 tests in 89.835 seconds**
with `FLIGHT_RENDER_TESTS=1`. Those contract, numerical, physics and camera checks
are separate from actual learned-guidance flight acceptance; they do not qualify
the failed recovery, browser or comparison trials above.

The [upgrade evidence bundle](../evidence/px4_upgrade/README.md) preserves the
development result, its retrospective audit, final-run failures and comparison
outcomes. A controlled diagnostic recovery result does not establish general
recovery, full-noise robustness or a neural advantage. The earlier acceptance
bundle below remains unchanged.

## Preserved earlier validation

An early development build without stock lockstep completed takeoff and landing.
Its learned following run recorded 34 fresh neural commands, 109 depth-approved
forward ticks and approximately 0.91 m of estimated horizontal movement, rejected
a delayed result and confirmed landing/disarm. That run used weaker acceptance:
it did not require the injected delay to interrupt an actively advancing command
or require a reduction in the selected person's measured depth. It is preserved
as development evidence, not a pass of the strict test below.

The first strict validation failed before stable takeoff because a fresh source
clock receipt was unavailable. After switching to stock SIH lockstep, strict
validation and a separate hover diagnostic failed the stable-takeoff gate;
landing/disarm was confirmed. These runs provide no accepted strict following
result. Sensor noise is a hypothesis under investigation, not an established
cause of those failures.

An intervening zero-noise probe failed before arming. Repeated identical
magnetometer measurements triggered PX4's constant-value stale-data check; its
log reported a stale `MAG #0`, and the run never reached estimator readiness.
That failed probe is preserved. The replacement diagnostic retains 1% of stock
noise while leaving those validator checks intact; zero-noise mode is no longer
offered by the runner.

The earlier `px4_sih_low_noise_validation04` run passed all 11 gates with the
declared 1%-noise profile: 11 fresh neural guidance results, 59 advancing follow
ticks, 0.3293 m following displacement, and 0.2903 m final selected-depth
reduction. A 4.213-second delayed result was rejected. The first send after
command expiry requested zero while depth was still clear; all 188 subsequent
sends requested zero. Late horizontal speed stayed below 0.022 m/s, and PX4
confirmed landing/disarm.

This is one controlled case. The stationary association-retention branch was
used zero times in that passing flight; its focused tests do not establish
actual recovery from the preceding flight's identity loss. Stock-noise hover
and general tracking robustness remain unresolved. Earlier failures, including
the short-follow protocol and identity-loss cases, are retained unchanged in
the [compact evidence bundle](../evidence/px4_sih/README.md).

Verify the frozen receipt (consistency and gate recomputation, not physics replay):

```bash
.venv-test/bin/python -B scripts/verify_px4_sih.py evidence/px4_sih/validation
```

## Acceptance policies

Without `--mission` or `--smoke`, the preserved short following protocol requires:

- A verified, newly launched local PX4 child, healthy estimator and stable takeoff.
- At least five fresh selected-person neural results and ten depth-approved
  forward commands; at least 0.30 m horizontal movement and a 0.20 m reduction
  in the selected person's measured surface depth.
- Complete the following milestones first, then inject a four-second worker
  delay while a recent forward command exceeds 0.10 m/s. Stall initiation must
  fall between 4 and 18 seconds, leaving its full six-second observation period
  inside the unchanged 24-second mission cap. If this cannot occur, fail.
- Preserve the active command's original capture/expiry. Reject the delayed
  result; send zero forward speed after expiry; slow below 0.10 m/s.
- Request LAND and confirm both fresh landed state and disarmed heartbeat.

Mission scoring is separate. It requires healthy takeoff, the complete declared
window, valid capture-anchored commands and fresh depth/state for every positive
send, no wrong-person observations, and confirmed landing/disarm. Tracking
qualification additionally requires fresh guidance and forward following, with
at least **80% of the declared window** covered by fresh selected observations
and a uniquely confirmed original actor. Startup, loss and expiry intervals
count against coverage. Missing or ambiguous identity evidence never counts as
correct identity. Fault-enabled runs also require a submitted delay and a
stale rejected output.

`--recovery` adds independent checks: one genuinely missing front-depth frame,
an older retained association anchor, at least 0.02 rad of measured yaw between
anchor and gap, zero requested/sent forward motion during the gap, and a fresh
same-person/current-depth recovery with a valid positive send within 1.2 seconds.
Retention alone cannot pass. Rendered actor annotations are matched to the
exact observation sequence and capture timestamp in the offline evaluator;
they do not enter selection, depth gating or control.

Smoke scores only the autopilot lifecycle and reports hover metrics separately.
It does not load YOLO/Flyvis or demonstrate following. Studio uses the distinct
operator-demonstration policy described above.

`summary.json` contains all gates; `observations.json`, `control.json` and
`events.json` preserve the inputs needed to audit them. `provenance.json` pins
runtime sources and each new run copies them into `runtime/`.
`recording/camera.mp4` and `recording/overview.mp4` show the
following/hold interval. Takeoff and landing have telemetry evidence, but are
outside these videos. Studio stores these flight receipts below `flight/` and
its browser-accessible recordings below `capture/`.

## Timing and ownership

Inference runs in a separate process with one in-flight image. The control loop
never waits for model completion. Results older than 0.65 seconds cannot be
released; commands expire at capture + 0.90 seconds. The independent depth and
estimated state must be no older than 0.10 seconds when sending forward motion.
The forward speed cap remains 0.45 m/s. No timeout was enlarged to make the run pass.

Video encoding has a separate recording thread with at most two queued samples.
If it falls behind, samples are dropped and counted; encoder startup and pipe
writes cannot block flight polling. Camera/overview rendering stays on the
graphics thread and its cost is included in the final command deadline check.
Videos use elapsed host time and causally hold the latest admitted image;
telemetry also records PX4 source time. Finalization follows simulator shutdown.
Studio preview JPEG encoding has a separate single-slot queue. OpenCV and the
JPEG path are initialized before starting the preview thread or flight runner,
so their first-use startup cost does not occur during flight. Preview drops
are counted and never expand the flight freshness limits.

TIMESYNC echoes establish conservative acquisition-time lower bounds. Buffered
old source samples cannot become fresh because a new packet arrived. Source
time need not advance at exactly host-clock rate. Pauses, stale depth, failed
selection and expired guidance remove forward authority. PX4's own offboard-loss
action is configured to LAND after a lost setpoint stream; the
vision-stall test keeps that stream alive with zero forward commands.

This adapter has no serial port, remote address or attach option. It uses fixed
connected loopback sockets, exclusive receiving ports, an OS lifetime lock,
child-process liveness and source-port ownership checks. An occupied port causes
refusal. It never stops another PX4 instance or force-disarms an airborne vehicle.

## Limits

MuJoCo renders RGB/depth at SIH truth pose but never advances vehicle dynamics.
PX4 estimates supply camera metadata and control state. Depth is ideal simulated
optical depth, with the existing native Mac depth-precision warning. SIH does not
collide with obstacles in the rendered room, so this experiment cannot validate
physical obstacle clearance or contact response. The scenes use stylized
actors and temporary clothing/geometric association; CLI missions add bounded
walking, crossing and occlusion to the stationary fixture.

The frozen Flyvis readout integrates 0.1 model seconds per sampled target cue;
that is not a continuously clocked biological response to raw camera motion.
The experiment tests this connected control path. It does not establish
superiority over conventional guidance, physical aircraft readiness, general
navigation or onboard performance.

Default recordings are capped at 32 MiB, omit neural arrays, and disable PX4's
full ULog capture at startup. Build tools/model dependencies remain reinstallable
local files; they are not included in Git. No automatic deletion of older runs occurs.

Upstream: [PX4 SIH guide](https://docs.px4.io/v1.16/en/sim_sih/) and
[pinned PX4 source](https://github.com/PX4/PX4-Autopilot/tree/v1.16.2).
