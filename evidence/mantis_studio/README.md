# Mantis Studio development evidence

These are compact records from local implementation checks on 2026-09-17.
They are **not a predeclared acceptance suite**, and `completed` means the
simulation ended, not that tracking or detouring succeeded. Failed development
runs remain included. Source hashes in each provenance file identify the code
used for that run; some earlier snapshots differ from the final code.

## Outcomes

- Browser launch, explicit person selection, pause/resume, stop/save, rapid live
  edits, reload persistence and a 390 px layout were exercised. The selected
  camera and world view updated. An accessible replay button visibly played a
  recording; clicking the embedded browser's native video accessibility control
  caused renderer crashes, so playback was verified through the page control.
- A stationary-person development run moved about 1.6 m before a missed detection
  latched a stop. A crossing run had 0 wrong-person observations in 23 scored samples
  and stopped on target loss. These do not establish uninterrupted tracking or
  general identity accuracy.
- The final detour trial (`run-20260917T124128-22d662e7`) executed 114 actual YOLO
  calls and 114 target-cue Flyvis observations, scored 113 selected observations
  without a wrong-person match, and had no contacts. It moved to approximately
  `(1.969, -0.253) m` but hit its 14-second attempt limit. **Zero detours completed.**
  The isolated depth planner passes its controlled fixture; reliable completion
  with learned guidance remains unresolved. Earlier detour failures are retained.
- Research recording (`run-20260917T123853-57697e97`) hit the 16 MiB limit and
  finalized at 16,659,487 bytes with 42 raw captures and a playable 13.8-second video
  prefix. It reports `budget-exhausted`, not complete research evidence. Its final
  simulation state had effectively zero velocity. The trial was already holding
  its follow distance, so this does not independently prove braking from motion.
- The connected raw-motion observe/brake trials generated actual neural output,
  but CPU processing exceeded the freshness/gap windows, repeatedly resetting
  warm-up. The final short observe/brake trials each allowed explicit selection,
  recorded three raw-motion observations, and had zero valid motion observations.
  No independent motion-braking benefit was demonstrated.
- The separate [frozen motion benchmark](motion/README.md) contains 156 actual
  observations, 96 scored. Held-out nominal vector RMSE was 23.6581 for Flyvis,
  0.2437 for Farneback and 22.1811 for a zero-motion baseline. Lower is better.
  No model/sign/lag/threshold was tuned against held-out outcomes.

All 14 development runs and their compact receipts are indexed in
[live_checks.json](live_checks.json). Each `live/<run>/` folder contains exact
summary/provenance copies and selected telemetry fields, without raw arrays or
videos. Original video/telemetry hashes are retained. The budget trial omits the
last rejected observation and subsequent braking telemetry; its final summary
records the endpoint. Other trials retain all observation-level samples.

The final combined native suite passed **787 tests in 75.443 seconds**, with
rendering enabled. Independent code review checked guidance-gap preservation
of detour bounds and failed holds. See [verification.json](verification.json)
for the check inventory and final source hashes. Software tests are distinct
from the development flight outcomes above.

Run the portable checks from the repository root:

```bash
python evidence/mantis_studio/verify_live.py
python evidence/mantis_studio/motion/verify.py
```

[video_decode.json](video_decode.json) records full FFmpeg decoding checks for
all saved videos, including the partial budget-limited prefix. Videos stay local
under ignored `results/studio/`, subject to the studio's storage budget.

These are stylized motor-physics simulations with ideal depth/state sensors.
No physical flight, real-time execution, general obstacle navigation or neural
superiority is established. See [the Studio guide](../../docs/MANTIS_STUDIO.md).
