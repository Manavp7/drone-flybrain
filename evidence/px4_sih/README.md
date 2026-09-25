# PX4/SIH controlled integration evidence

`validation/` is the unchanged receipt for local run
`px4_sih_low_noise_validation04` on 2026-09-25 UTC. All eleven declared acceptance
gates passed. The executable was actual pinned PX4 v1.16.2 with SIH dynamics;
MuJoCo rendered camera geometry only. A separate worker used the existing
YOLOX/Flyvis neural-guidance pipeline. There were no hardware connections.

## Declared configuration and outcome

| Item | Recorded value |
| --- | --- |
| Simulated GPS/barometer/magnetometer/IMU noise | Explicitly reduced to 1% of stock |
| RGB/depth | Ideal rendering of stationary stylized people |
| Fresh selected-person neural results | 11, all selected Track 1 |
| Positive commands during following | 59 |
| Total positive sends, including before expiry during stall | 78 |
| Following displacement before stall | 0.3293 m |
| Selected-depth reduction at stall / final | 0.2249 m / 0.2903 m |
| Delayed inference result | 4.213 s old; rejected |
| Sends after frozen command expiry | 188, all requested zero |
| Maximum late-hold horizontal speed | 0.02111 m/s |
| End condition | PX4 landed, disarmed and exited |
| Recording | Two 10.6 s H.264 videos, 76 admitted samples, zero drops, 513,237 bytes total |

Source snapshots preserve all 81 runtime hashes. `artifact-manifest.json` pins
the exported data. `control.json`, `observations.json`, `events.json` and
`takeoff.json` preserve the underlying measurements. No checkpoint, native
binary, full neural array or raw video is included in this bundle. Local videos
remain in `results/px4_sih_low_noise_validation04/recording/`; they cover the
following/hold interval, with takeoff/landing evidenced by telemetry.

```bash
.venv-test/bin/python -B scripts/verify_px4_sih.py evidence/px4_sih/validation
```

The verifier checks hashes, recomputes gates and audits positive-command
deadlines. It does not independently replay physics or authenticate a physical
aircraft. The final source adds post-flight propagation of encoder error
receipts after this frozen flight; that error path has a focused regression
test. The successful recording's receipt has no error, so that correction does
not change its result.

## Preserved negative evidence

| Folder under `failed/` | Outcome |
| --- | --- |
| `validation01` | Nonlockstep source-clock freshness failure before stable takeoff; accelerometer timeouts; landed/disarmed |
| `validation02` | Stock-lockstep stable-hover failure; landed/disarmed |
| `hover_diagnostic02` | Repeated stock-noise hover failure with fresh source clocks; landed/disarmed |
| `ideal_smoke01` | Zero-noise magnetometer became constant/stale; never armed |
| `low_noise_validation01` | Passed takeoff, then freshness failure associated with synchronous recorder startup; landed/disarmed |
| `low_noise_validation02` | Stall/expiry/slowdown/landing passed; only 0.2415 m following and 0.1531 m depth reduction, so overall failed |
| `low_noise_validation03` | Selected Track 1 was lost and became Track 4; held and landed; no stall injected |

These runs keep their original outcomes and provenance; older snapshots are
retained in the local results. The first two reduced-noise failures have their
original protocol; validation03 and the passing run use a two-stage protocol:
complete unchanged following milestones, then start an active-command stall
between 4 and 18 seconds within the existing 24-second mission cap. This is a
protocol correction, not evidence of better controller performance.

## Limits

Stock-noise stability remains unresolved. The passing trial did **not** exercise
the new one-gap stationary association-retention branch (zero uses), so it
does not prove recovery from validation03's identity failure. That branch is
covered by focused tests and retains only old association geometry with its
original timestamp for at most 1.2 seconds; current missing-depth guidance
remains invalid. Shared Studio/base tracking is unchanged.

This is one controlled simulation, not a reliability distribution, onboard
real-time guarantee, physical obstacle-contact test, hardware-flight acceptance,
or demonstration that neural guidance outperforms conventional guidance.
