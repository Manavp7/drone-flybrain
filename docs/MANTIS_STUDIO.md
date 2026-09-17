# Mantis Flight Studio

Mantis Studio is a local browser interface for running new motor-driven MuJoCo
simulations, selecting a detected person, changing the scene and saving bounded
recordings. It complements the existing recorded-run viewer on port 8874.
There is no aircraft connection. A responsive browser does not mean the
simulation or learned perception runs in real time.

## Start the studio

Use the project's existing Python 3.12 environment, dependencies and pinned
YOLOX-tiny/Flyvis model installation. Follow [model setup](MODEL_SETUP.md) if the
weights are not installed; they are not bundled in the public repository.
Native MuJoCo rendering needs a working OpenGL context. Recording requires
`ffmpeg` with `libx264` on `PATH`.

From the repository root, with that environment activated:

```bash
python -m experiments.mantis_studio
```

Open [Mantis Studio](http://127.0.0.1:8875). The server binds to `127.0.0.1` and
starts one simulation child at a time. It stays responsive while the child
loads models. No Node installation or frontend build is required.

Optional server settings:

```bash
python -m experiments.mantis_studio --port 8875 --runs results/studio --max-total-mb 256
```

`--port` accepts 1024–65535. `--max-total-mb` accepts 32–2048 and uses binary
MiB units. The default run directory is `results/studio`. Existing recordings
are retained across server restarts; a previous child is not automatically
resumed.

## Browser workflow

1. Choose the scene, guidance method, recording profile and run budget, then
   press **Launch simulation**.
2. Wait for the camera preview. The initial simulation waits for an explicit
   selection when a fresh selectable person is available.
3. Click one detected person in the aircraft camera, or its **Track #** button.
   A click includes the displayed frame sequence. If the frame has changed or
   become stale, select from the new view.
4. Watch camera/overview images, selected track, vehicle speed, depth decisions,
   eight target-cue neural features and recorded size. **Pause** freezes the
   simulation clock; **Resume** continues it. This is not a physical hover test.
5. **Stop & save** requests simulated braking and finalizes a partial run.
   Stop is terminal for that run; later Resume requests cannot cancel it.
   Completed recordings appear in the notebook with camera and overview replays
   and a JSON results link.

The overview is an evaluation view. It does not provide target identity or
navigation input. On narrow displays the overview is hidden to preserve room
for the selectable camera.

Normal duration completion means the simulation ended; it does not by itself
demonstrate a final stop. Pause before changing the selected person if advancing
frames make repeated clicks stale.

### Controls and bounds

| Setting | Values | Change during a run? |
| --- | --- | --- |
| Scene | `walk`, `crossing`, `occlusion`, `stationary`, `detour` | Yes |
| Follow distance | 2.5–5.0 m, default 3.5 m | Yes |
| Target speed | 0–0.2 m/s, default 0.08 m/s | Yes |
| Short detours | Off/on, default off | Yes |
| Guidance | `mantis_neural`, `direct_yolo`, `alpha_beta` | Before launch |
| Duration | API: 2–90 simulation seconds; browser: 15/30/60/90 | Before launch |
| Raw-image motion | `off`, `observe`, `brake` | Before launch |
| Recording | `compact`, `full-research` | Before launch |
| Per-run budget | Integer 16–128 MiB, default 64 MiB | Before launch |

Stationary and detour scenes keep their ground anchors fixed; the speed slider
does not move those anchors. Scene changes may cause tracking loss and require
explicit reselection. Inference completion and the stop/braking interval can
extend the final recorded simulation duration beyond its requested boundary.

## Two people and explicit selection

The arena contains two copies of the pinned skinned actor, with independent
animation bones and distinct clothing. Crossing and occlusion are prescribed
test motions. The aircraft remains the original free-body, four-motor fixture.

`SelectionGuard` consumes current detector boxes and camera appearance. It
never automatically selects the highest-confidence person or substitutes a
different track after loss. It keeps a fixed, softly binned upper-body colour descriptor from the
selection frame, with separate treatment of neutral clothing. Overlapping people, a disappearing person near the selected
box, changed clothing, missing appearance, stale/reordered frames and missing
selected detections stop valid target observations. Ambiguity, missing-person and changed-appearance holds remain latched until
explicit reselection. A separately verified computation-deadline hold may recover
within the existing three-second association-memory limit, measured from the
last accepted observation. Recovery requires a fresh same-ID detection, the
fixed appearance anchor, and current/pre-gap competitor checks. Old boxes are
not selectable during the gap. Repeated timeouts cannot renew the window;
observation freshness and command expiry do not change.

This is temporary visual tracking, not identity recognition. Similar clothing,
missed detections and an unobserved identity swap can remain unresolved. Wrong-
person statistics compare valid selected observations against independent
projected actor geometry. Ambiguous geometric matches are unscorable, not
successful matches. The displayed count measures observations, not a validated
real-world identity-switch rate. Evaluator truth never enters selection or
guidance.

The `mantis_neural` path uses YOLO recognition to form the existing target cue
for Flyvis, then uses its calibrated bearing readout. `direct_yolo` and
`alpha_beta` select conventional bearing estimates with the same supported
target/cue and depth conditions. The shared pipeline still executes the neural
path for measurement, including in these baseline modes.

## Bounded recordings

`compact` is the default. It retains browser-playable H.264 camera and overview
video, observation telemetry, source/model provenance, result summary and actor
credits. It omits the large per-observation neural/depth arrays.

`full-research` adds compressed numeric arrays supplied by the session: depth
with its validity mask, neural activity, retinal input and readout features.
These are additional diagnostic records. They do not promise exact
reconstruction of every physical state, original RGB pixel or recurrent state.
Videos remain compressed, and the telemetry is observation-based rather than
a dump of every 200 Hz physics step.

Each recorder limits the logical bytes of the files it writes, including
reserved final summary space. It admits complete MP4 fragments so a budget hit
can retain a playable prefix. Exhaustion detected during an active run stops recording and requests braking.
Exhaustion discovered during final encoder flush occurs after simulation end.
Both mark the saved run `budget-exhausted`; such evidence is partial. Errors and
user-stopped runs are also distinguished from completed recordings.

The studio defaults to a **256 MiB total budget** across its run directory. A
new run must reserve its entire requested recording budget plus **4 MiB** for
bounded status, preview and log overhead. Admission is refused if that reserve
does not fit. Neither the server nor recorder automatically deletes old runs.
Move a finished run out of the studio directory, choose a smaller next-run
budget, or deliberately change the total budget. Filesystem allocation overhead
can differ from logical byte counts.

Typical output:

```text
results/studio/run-.../
  config.json
  control.json
  state.json
  runtime.log
  camera-<sequence>.jpg
  overview-<sequence>.jpg
  capture/
    camera.mp4
    overview.mp4
    telemetry.jsonl
    provenance.json
    summary.json
    CREDITS.txt
    raw/                    # full-research only
```

Only a bounded recent preview set is retained. Saved video uses simulation time
and holds the latest image only after that image is available; it must not be
interpreted as camera-rate or real-time performance evidence.

## Short obstacle detours

`DetourNavigator` is a conventional local planner. Its inputs are the current
registered depth, vehicle state and a fresh selected-target command. It does
not receive actor positions, obstacle coordinates or an occupancy map.

It first brakes, tests small alternate headings against a longer depth horizon,
then turns in place. Every moving step still depends on the unchanged final
`DepthGuardian` check along the actual aircraft heading. Rejoining also brakes
before turning. If the old camera view cannot certify the target route, the
aircraft can inspect it while stationary; translation requires a new clear
depth view. Brief braking drift receives zero movement authority until the
existing gate accepts the measured velocity again.

Attempts are bounded to 14 seconds and 2 metres, with small heading changes that
must keep current target guidance supported. There is no side/reverse command
authority. Missing target/depth, unsupported visibility, a time/distance limit
or no observed corridor causes a stop. A centred obstruction that cannot be
passed inside these constraints correctly produces a hold.

For the deliberately offset development fixture, choose **Offset obstacle**,
enable detours, select the person nearest the camera centre, and use a **2.5 m** follow distance. The browser applies those
settings when selecting that scene. The isolated depth planner completes this controlled fixture. The final
actual-YOLO/Flyvis development trial retained 113 scored observations without
wrong-person matches but hit its 14-second attempt limit: zero completed
detours. Reliable integrated detour completion remains unresolved; this is not
general obstacle navigation. See the [development evidence](../evidence/mantis_studio/README.md).

## Raw-camera Flyvis motion experiment

The optional motion arm owns an independent recurrent Flyvis state and receives
camera luminance rather than YOLO boxes or the target mask. It records the
upstream optic-flow decoder response, T4/T5 population summaries and a matched
Farneback optical-flow comparison. Decoder-to-pixel velocity transfer remains
uncalibrated for this scene. Image motion includes camera motion.

`observe` reports measurements without altering commands. `brake` is explicitly
experimental: a fixed neural-motion magnitude rule can reduce forward speed,
never add speed or steering. Invalid, warming-up or stale motion information
holds forward motion; the depth stop gate remains final. The extra model work
can increase inference latency and reduce available guidance. On the development
Mac, live raw-motion processing exceeded the freshness window and repeatedly
reset warm-up; these short runs did not establish usable motion control. Keep
this arm off for ordinary interactive flights. The first selection preview
is presented before the optional research computation starts.

The local frozen raw-motion benchmark did **not** demonstrate an advantage over
its conventional baseline. The [portable benchmark evidence](../evidence/mantis_studio/motion/README.md) includes independent score verification. There is no demonstrated flight-control benefit from
this optional brake. Existing Mantis neural guidance also has not established an
advantage over the strongest conventional baseline. Diagnostic activity is not
evidence of a complete fly brain controlling an aircraft.

To run a fresh, compact motion benchmark in a new directory:

```bash
python -m experiments.mantis_motion \
  --manifest models/flyvis_0000_000.manifest.json \
  --output results/mantis_motion_new
```

## Local API and integration points

The browser uses JSON requests with its server-provided `X-Mantis-Token` and
same-origin/loopback checks. The token is not a public remote API credential.
Only one child session can run at a time.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/state` | Current state, up to 12 recent runs, defaults and storage |
| `POST /api/start` | Validated configuration object; launches a new session |
| `POST /api/control` | Active `run_id` with operation, live settings or selection |
| `GET /api/frame?id=...&sequence=...&view=camera` | Sequence-bound camera JPEG; `overview` also supported |
| `GET /media/<run_id>/<artifact>` | Allowlisted completed/partial recording files with byte-range support |

A selection control has the following body; use actual values from the
displayed current state rather than copying these example identifiers:

```json
{"run_id":"run-20260917T120000-01234567","selection":{"track_id":1,"sequence":3}}
```

Other controls use `operation: "run" | "pause" | "stop"`, a `settings` object
containing only live keys, or `selection: {"clear": true}`. Both parent and child
check the selection frame. Unsupported fields and stale run IDs are rejected.

The implementation separates responsibilities:

- `mantis_studio.py`: loopback HTTP server, admission budget and child lifecycle.
- `mantis_session.py`: model/simulator loop, controls, final depth gate and telemetry.
- `mantis_arena.py`: two-actor fixture and evaluator-only geometry.
- `mantis_selection.py`: observation-only explicit selection and latched holds.
- `mantis_navigation.py`: bounded depth-based detour proposals.
- `mantis_recording.py`: streaming recorder with a byte budget.
- `mantis_motion.py`: independent raw-camera neural motion diagnostic and benchmark.

## Evidence boundary

Component tests and separate native rendering/motor checks support the new
software pieces. A local navigation probe completed a short detour using actual
motors and rendered depth with an observed colour cue; that isolated check did
not run the YOLO/Flyvis target pipeline. It cannot establish full studio
integration, general person tracking, neural advantage, real-time performance
or physical flight readiness. Use the actual saved session receipts and current
validation results for any stronger claim.
