# Public evidence subset

These records were copied from the completed local motor-flight experiment. They are a compact publication subset, not the full raw run.

- `flight-summary.json`: all nine scenario scores and the predeclared acceptance criteria.
- `flight-metrics.json`: aggregate timing, inference, physics and observation counts.
- `independent-review.json`: separate reconstruction of saved detections, camera-motion association, neural features, retinal sampling and readout headings. The reviewer authored the report module, so this receipt does not independently certify its scoring code.
- `source-integrity.json`: hashes and publication provenance for the unchanged original Python files and calibration fixtures, generated during release preparation.
- `release-verification.json`: fresh dependency installation, all 515 native-rendering tests, exact model setup, a passing ten-second actual YOLO/Flyvis simulation and staged-payload checks before publication.

The original raw run contained 2,247 files. It included RGB-D frames, all neural activity arrays, camera poses, commands and complete physics traces. A coordinator replay verified all 16,000 motor-physics steps and safety decisions; the separate neural/tracker review is preserved here. Those large raw arrays, downloaded model weights and source videos are not published. Historical local source-lock hashes are not presented as a complete downloadable archive.

The demo video and screenshot are in [`assets/`](../assets/ATTRIBUTION.md), with their CC BY 3.0 attribution. Scores use approximate projected photograph bounds and do not constitute a general person-tracking benchmark.
