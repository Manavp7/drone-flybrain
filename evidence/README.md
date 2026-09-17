# Public evidence subset

These records are a compact evidence subset covering the original V1 release and the Mantis V2 and 2.1 evaluations. The full raw runs are not bundled.

## Mantis 2.1

- [`mantis-v2.1-summary.json`](mantis-v2.1-summary.json): two frozen batches with 36 actual-model flights, 20 matched-input bearing comparisons, the broader synthetic readout calibration and its separate verification.
- [`mantis-v2.1-verification.json`](mantis-v2.1-verification.json): separate saved-evidence reconstruction, native physics/depth replay, media validation and actual browser checks, with each verifier's independence and reuse limits stated.

**All 36 operational trials passed.** This includes six visibility-preserving obstacle trials that independently demonstrated depth braking, plus twelve delay/recovery trials. The 686-test native software suite passed. Neural guidance still did not beat the stronger conventional baseline in any of the twenty matched comparisons; improved synthetic cue decoding is not a claim of conventional-controller superiority.

Full RGB-D captures, neural activity, traces, calibrated-cue recordings and 36 replay videos remain local. The original failed V2 run and first failed remediation probe are preserved. See [Mantis 2.1](../docs/MANTIS_V2_1.md) for precise mechanisms, test changes, reproduction and limitations.

## Mantis V2

- [`mantis-v2-summary.json`](mantis-v2-summary.json): twelve actual-model flights, ten matched-input bearing comparisons, counts and timing from the frozen `mantis_run02`.
- [`mantis-v2-verification.json`](mantis-v2-verification.json): separate reconstruction of neural readouts, controller math, benchmark scores and clocks, plus frozen-implementation replay of all 22,800 motor-physics steps and depth decisions. All 834 raw artifact hashes were checked. The verifier did not rerun YOLO or the recurrent neural dynamics; its other limits are recorded in the receipt.

**Eight of twelve** operational cases passed. One filtered walking run lost its target at the end. All three obstacle runs stopped, but target loss prevented them from demonstrating independent depth braking. Neural bearing estimates did not beat the stronger conventional baseline in any of the ten comparisons. Passing evidence verification does not turn these experimental failures into passes.

The complete RGB-D captures, neural arrays, motor traces, twelve replay videos and detailed verification artifacts remain local. A later narrow-screen CSS and waiting-text change is explicitly distinguished from the frozen runtime in the receipt. See [Mantis V2](../docs/MANTIS_V2.md) for reproduction and limitations, and [actor attribution](../assets/mantis_actor/ATTRIBUTION.md) for the CC BY 4.0 animated asset.

## Original motor-flight experiment (V1)

- `flight-summary.json`: all nine scenario scores and the predeclared acceptance criteria.
- `flight-metrics.json`: aggregate timing, inference, physics and observation counts.
- `independent-review.json`: separate reconstruction of saved detections, camera-motion association, neural features, retinal sampling and readout headings. The reviewer authored the report module, so this receipt does not independently certify its scoring code.
- `source-integrity.json`: hashes and publication provenance for the unchanged original Python files and calibration fixtures, generated during release preparation.
- `release-verification.json`: fresh dependency installation, all 515 native-rendering tests, exact model setup, a passing ten-second actual YOLO/Flyvis simulation and staged-payload checks before publication.

The original raw run contained 2,247 files. It included RGB-D frames, all neural activity arrays, camera poses, commands and complete physics traces. A coordinator replay verified all 16,000 motor-physics steps and safety decisions; the separate neural/tracker review is preserved here. Those large raw arrays, downloaded model weights and source videos are not published. Historical local source-lock hashes are not presented as a complete downloadable archive.

The demo video and screenshot are in [`assets/`](../assets/ATTRIBUTION.md), with their CC BY 3.0 attribution. Scores use approximate projected photograph bounds and do not constitute a general person-tracking benchmark.

## Interactive Studio development

[Initial Studio checks](mantis_studio/README.md) preserve bounded recordings,
selected-person checks, the initial failed integrated detours and the negative
raw-motion result. [Studio remediation](mantis_studio_remediation/README.md)
records the subsequent 3/3 controlled detours, both passing raw-motion modes,
bit-exact runtime optimization checks and 821 passing native software tests.
The earlier failures and lack of demonstrated neural superiority remain intact.
These development trials do not replace or regrade the earlier frozen suites.
