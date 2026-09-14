# Mantis 2.1: tracking recovery and independent braking

This revision addresses the failures recorded in [Mantis V2](MANTIS_V2.md).
The original `mantis_run02` and its 8/12 result remain unchanged.

## Repairs

**Bounded tracking memory.** The filtered walking failure followed a 1.585 s
processing gap. YOLO subsequently detected the person, but the track's 1.2 s
lifetime had expired. Mantis now retains prior observed descriptors and depth
anchors for at most 3 s. A fresh detection still has to pass the existing class,
overlap and clothing checks. Similar-looking people remain ambiguous; this is
temporary association, not persistent human identity.

A development probe exposed a second reset path: detector deadline rejection
cleared all descriptors even within that lifetime. The Mantis pipeline now
preserves only prior observed memory after verified, chronological inference or
processing timeouts in the same established stream. It rejects the stale
detections, retains old observation times, counts processing age toward memory
expiry and keeps IDs monotonic. Stream changes, reordered frames, invalid
clocks, malformed detector output and errors still clear memory.

Result freshness stays **0.65 s**, and commands expire **0.90 s** after capture.
Longer association memory cannot authorize motion from a stale observation.
The original perception pipeline and research sources remain unchanged.

**Independent obstacle braking.** V2's centered barrier obscured the person
before depth needed to brake. The revised fixture places the same box at
`[1.75, 0.40, 0.7]`, intersecting the aircraft corridor while leaving the actor
visible. Geometry and safety thresholds are unchanged.

The revised test requires a positive command derived from a fresh, post-event
person observation. Registered depth must override that command for at least
ten consecutive physics intervals (0.05 s). A synchronized counterfactual
depth capture, with only the barrier removed, must permit the identical request.
At least three fresh post-event person captures and 80% post-event coverage are
also required. A leftover pre-event command or a stop caused by target loss
cannot satisfy this check.

Counterfactual captures and their guardian outputs are evaluator evidence only.
The barrier is restored before physics advances; only the actual depth decision
reaches the autopilot. The test demonstrates stopping, not obstacle detours.

**Broader neural readout calibration.** The previous synthetic calibration used
box aspect ratios 0.40 and 0.47, poorly covering the animated actor's changing
silhouette. A new readout uses the same eight actual neural features and the
same ridge penalty. It was fitted on 739 generated rectangle/motion observations
and frozen before 544 disjoint held-out observations. Neither recorded flight
truth nor benchmark scores entered fitting.

Held-out horizontal reconstruction RMSE fell **20.32%**, vertical RMSE fell
**2.87%**, and height RMSE fell **72.30%**. Usable prediction coverage rose from
**476/544 to 544/544**. All predeclared selection gates passed. The worst abrupt
horizontal transition increased slightly (0.146 to 0.156 normalized units),
which remains in the complete scores. This is improved synthetic cue decoding;
it does not establish superiority over conventional controllers.

The new readout and its selection/model hashes are pinned in
[`assets/mantis_calibration`](../assets/mantis_calibration/README.md). The Flyvis
network remains frozen, and only neural features enter readout prediction.

## Evaluation protocol

Each of three controllers runs six scenarios: walking, turning, brief target
loss, an offset obstacle, a 1.6 s processing pause, and a 1.6 s detector pause.
The detector-pause test actually delays the detector before executing YOLO once;
its wall time includes the pause. Physics never counts that delay twice.

Both pause cases must reject the stale result, expire forward authority, stop,
recover the same temporary ID from fresh observations and restore tail tracking.
The detector case additionally requires evidence that a rejected, empty result
preserved prior memory without renewing timestamps or granting authority.

The nominal batch is a regression evaluation. A separate predeclared validation
batch offsets actor paths by `[-0.10, +0.12, 0]`. Each batch runs ten matched-input
bearing comparisons on its walking/turning neural camera paths: clean input,
three noise seeds and missing-observation gaps. The validation seeds are
38101–38103; baseline filter parameters remain unchanged.

All sources, readout/model hashes, scenario settings and criteria freeze before
each batch. No tuning occurs during either evaluation. Measured inference delays
affect the camera paths and observation counts, so independent flights are not
matched-image accuracy tests. The separate comparisons use identical inputs.

## Verification and results

The complete native-rendering software suite passed **686 tests**. Independent
calibration verification regenerated every cue and reproduced the fit with a
separate least-squares formulation, matching coefficients within 2.87e-16.
All held-out scores and selection gates matched.

The isolated development obstacle probe passed all three controllers, with
808/788/798 qualifying paired braking intervals (neural/direct/filtered).
An actual filtered-controller detector-pause probe passed stopping and recovery.
The earlier development probe is preserved as failed: long processing delays
reduced fresh observation coverage, and a detector timeout exposed the descriptor
reset bug described above.

The frozen nominal and shifted validation batches both completed with every
operational gate passing:

| Scenario | Nominal | Shifted validation |
|---|---:|---:|
| Walking | 3/3 | 3/3 |
| Turning | 3/3 | 3/3 |
| Brief target loss and recovery | 3/3 | 3/3 |
| Independent depth braking | 3/3 | 3/3 |
| Processing pause and recovery | 3/3 | 3/3 |
| Actual detector pause and recovery | 3/3 | 3/3 |
| **Total** | **18/18** | **18/18** |

These runs contain **360 simulated seconds**, **72,000 motor-physics steps**,
**3,166 actual YOLO calls and flight neural observations**, and another **1,920
neural observations** in twenty matched-input comparisons. Summed episode wall
time was **679.28 s**, excluding the separate comparisons. No aircraft contact
occurred. All 30 source/asset files matched the same freeze across both batches;
the original 104 Python files and four V1 calibration fixtures remain unchanged.

The six obstacle trials supplied 676–831 qualifying paired braking intervals
each. Fresh post-event person coverage ranged from 81.4% to 97.8%, with at least
0.313 m conservative planar clearance. In the twelve pause trials, speed at the
end of the pause was at most 0.0186 m/s; every controller recovered its selected
temporary ID from a fresh detection and passed the original tail-tracking gates.

**The neural estimator did not beat the stronger conventional baseline in any
of the twenty matched comparisons.** Its capture-time bearing RMSE ranged from
0.301° to 0.886° across both batches; alpha-beta ranged from 0.250° to 0.701°.
Those ranges span different inputs and are not paired improvement estimates;
the complete per-condition scores, shared coverage and compute costs are in
[`mantis-v2.1-summary.json`](../evidence/mantis-v2.1-summary.json).

The tracking and braking failures have been addressed in these controlled
scenarios. The neural decoder-domain mismatch improved on its separate holdout,
but superiority over conventional guidance remains unsupported. The neural path
remains an experimental comparison option; direct and filtered guidance remain
available with the same safety rules.

Separate saved-evidence verification passed for both batches. It checked 3,488
raw artifact hashes, reconstructed all 3,166 closed-loop neural feature records,
and replayed every one of the 72,000 physics/guardian intervals. Coverage includes
7,200 safety-depth rerenders, 108 primary RGB-D rerenders and 540 paired
barrier-free captures. Reconstructed numerical values stayed within the declared
tolerances; native physics and guardian records matched exactly.

All **36 H.264 replay videos** decoded completely: 3,600 frames and 360 seconds.
Separate media checks verified 3,636 causal timeline samples and twenty benchmark
displays. Actual browser checks selected all eighteen shifted-run flights and
all ten benchmark conditions, exercised playback and seeking, observed command
expiry followed by fresh same-ID recovery, and confirmed depth braking with the
person still observed. The 341 px layout had no horizontal overflow.

The compact [`verification receipt`](../evidence/mantis-v2.1-verification.json)
records exact counts and independence limits. Verification does not rerun YOLO
or recurrent neural dynamics. Native physics/rendering reuse the frozen runtime;
the verifier also authored the actor/depth/causal helper, so those checks are
explicitly implementation replay rather than independent implementations.

The complete recordings and exported report folders remain local and are not
bundled in this repository. The commands below generate a new validation report
and serve its replay lab at `http://127.0.0.1:8874/` on your own machine.

## Reproduce

Install the [existing runtime and weights](MODEL_SETUP.md), then use new output
directories for every run and export:

```bash
python -u -m experiments.mantis_flight --output results/mantis_regression
python -u -m experiments.mantis_flight --validation-fixture --output results/mantis_validation
python -m experiments.mantis_report --run results/mantis_validation
python -m experiments.mantis_report --serve results/mantis_validation/report --port 8874
```

These remain offline simulations with a stylized actor, ideal registered depth
and ideal aircraft state. Real cameras, physical flight, real-time operation,
general person identity, detours and a whole-fly motor brain are not established.
