# PX4 upgrade experiments — 2026-09-25 UTC

These are local, owned PX4 v1.16.2/SIH simulations on an 8 GB Apple M1 Mac.
MuJoCo supplies ideal RGB/depth rendering of stylized people, not aircraft
dynamics. No physical drone was connected. **The five requested upgrades are
implemented or investigated, but they are not all qualified by successful
flights.** Original failed outcomes are retained.

## Status

| Requested outcome | Evidence and remaining limit |
| --- | --- |
| Normal-noise hover | Experimental estimator/sensor-accuracy calibration improved behavior. One 20-second zero-velocity hold completed, but a later full-noise mission failed stable takeoff. Reliability remains unresolved. |
| Tracking through a turn and one depth gap | One reduced-noise 30-second development run passed. A stricter retrospective audit confirms the original anchor was actually reprojected after depth returned. A later final run aborted on clock freshness before reaching the gap. |
| Longer repeated missions | A frozen 18-case plan was launched. Two walking cases failed on source-clock freshness; the predeclared rule stopped the batch and marked 16 cases skipped. Crossings, occlusion, repeats and delayed frames were not reached in this batch. |
| PX4 in browser Studio | Backend launch, live frames and telemetry were observed in three browser trials. All aborted on clock freshness before an accepted operator selection. Start/select/follow/stop end-to-end acceptance remains incomplete. |
| Flyvis contribution | Neural and no-Flyvis direct guidance both ran in actual PX4. Their unequal, truncated windows are not a valid performance comparison. No advantage is established. |

The freshness limits, selected-person checks, 0.45 m/s speed cap and automatic
LAND cleanup were kept. Clock failures were not converted into passing flights
by increasing their deadlines. Source timestamps stopped advancing quickly
enough for the conservative host-clock bounds in the failing runs. The available
logs do not isolate the underlying PX4/host scheduling cause.

## Normal-noise investigation

`diagnosis/` preserves original summaries, provenance, event logs and compressed
traces for all eight attempts in this upgrade investigation. Its index also
records an unchanged repeat that automatic approval review rejected before
execution; that attempt has no flight measurements. Intermediate profiles
are retained for reproducibility, not offered as proven fixes.

Source inspection found SIH measurement-noise levels and reported GPS accuracy
that differed from the estimator's settings. Trials tested barometric height,
matched IMU covariance, GPS accuracy reporting, and a ten-second continuously
healthy settling interval. GPS calibration changes the reported uncertainty;
it does not reduce injected noise samples. The separately selected 1% diagnostic
does reduce simulated sensor noise and is always identified in provenance.

`px4_normal_velocity01` completed a 20-second zero-velocity hold and landed. It
still recorded 0.3779 m maximum estimated position error, 0.1512 m/s estimated
speed p95, 0.2362 m/s truth speed p95, and 0.394 m truth altitude range. This is
not stable position-hold qualification. `px4_recovery_normal01` subsequently
failed the unchanged takeoff gate. The current smoke command holds position;
that later code change has not been qualified in another normal-noise flight.

## Recovery evidence

`recovery_development/` retains the original passing summary, full runtime
snapshot, events and losslessly compressed traces. The run used 1% of stock
GPS/barometer/magnetometer/IMU noise, real YOLO and the frozen Flyvis guidance.

| Measurement | Value |
| --- | ---: |
| Following window | 30.0418 s |
| Observations | 107 |
| Correct-identity time coverage | 93.6517% |
| Wrong-person observations | 0 |
| Missing-depth sequence / restored sequence | 13 / 14 |
| Measured anchor-to-gap yaw change | 0.032072 rad |
| Gap-to-released recovery command | 0.549347 s |
| Original anchor age at restored capture | 0.496530 s |
| Gap ticks requesting zero forward speed | 9 |

`retrospective.json` is explicitly a later, stricter audit, not a rewritten
original score or a new flight. It pins its input bytes and audit-policy source.
The audit checks the same selected ID, a valid restored-depth reprojection of
the exact original anchor, its original age, no explicit reselection, and a
fresh independently depth-approved positive recovery command. Truth projections
are evaluator-only; they do not enter guidance. `audit-policy/` contains the
policy used for that audit. This establishes one controlled stationary-person
recovery, not general moving-person recovery or reliability.

`recovery_final/` is the later failed qualification: 3.5605 seconds, 8
observations, source-clock freshness abort, landed/disarmed. It did not reach
the missing-depth injection. Its original failure remains unchanged.

## Frozen comparison

`comparison/plan.json` was written before any case launched: three methods ×
three scenes × two repeats, each with a 60-second following window, speed
0.08 m/s, reduced-noise sensors and an actual 1-second inference delay at the
midpoint. Both attempted cases stopped before that midpoint. The batch's
predeclared two-consecutive-clock-failures rule stopped further attempts.

| Attempt | Observed window | Observations | Tracking loss time | Wrong-person observations | Flyvis calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| Walking, neural | 17.0386 s | 60 | 1.4774 s | 0 | 60 |
| Walking, direct | 4.5748 s | 22 | 0.4597 s | 0 | 0 |

These are failed-attempt diagnostics with **unequal exposure**, not comparable
loss rates or evidence that either controller is better. Both landed/disarmed;
recorded positive sends passed their evidence checks. The original progress
summary retains all 18 cases: 2 failed, 16 skipped, 0 passed. There are no complete
three-method pairs. The batch is incomplete, not an 18-flight benchmark.

## Browser trials

`studio/` includes each original flight summary, provenance, events, compressed
traces and Studio state/configuration; the third trial includes its full runtime
snapshot. The browser was launched on loopback port 8876. All three displayed
live rendered frames but ended before accepted selection. First-use JPEG
initialization and excess rendering were corrected between trials. The final
trial still had a clock-receipt drought despite short control-loop ticks.
No claim is made that browser automation caused the failure.

## Verify and reproduce

From the repository root, after installing test dependencies:

```bash
python -B scripts/verify_px4_missions.py evidence/px4_upgrade/comparison
python -B scripts/verify_px4_missions.py evidence/px4_upgrade/recovery_final
```

Verification means the original receipts, source hashes and failed outcomes
agree with the audit policy. It does not turn failure into success or replay
flight dynamics. The verifier never executes archived source snapshots.

The development recovery predates the stricter mission policy, so it is
separately audited by `retrospective.json`; it is not silently accepted as a
current-policy run. Original full snapshots for intermediate diagnosis/browser
attempts remain in local `results/`. Native binaries, dependencies, checkpoints,
neural arrays and videos are excluded from this compact export. Gzip traces
decompress to their exact original JSON bytes. `artifact-manifest.json` pins
the exported files; `verification.json` records final checks.

Reproduction commands and architecture are in [the PX4 guide](../../docs/PX4_SIH.md).
The earlier successful following/stall trial remains unchanged in
[the separate original bundle](../px4_sih/README.md). The next flight work is
to isolate the native source-clock/scheduling failure on a supported host,
then run a new declared batch; do not resume or relabel this aborted one.
