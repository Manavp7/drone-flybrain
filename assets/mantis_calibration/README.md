# Mantis neural readout calibration

`readout.json` contains a small ridge readout fitted to the existing eight neural
features. It does not contain a Flyvis checkpoint. `selection_frozen.json` binds
the fit, upstream model manifest and complete validation scores to exact hashes.
The Mantis loader also pins both files in source.

The fit used 739 generated rectangle observations, including multiple widths
and moving cues. The candidate was frozen before inference on 544 separate
held-out observations. Training and held-out geometry and rasterized cues are
disjoint. No recorded person-flight truth or benchmark scores were fitted.

On that synthetic holdout, horizontal reconstruction RMSE fell 20.32%; usable
prediction coverage rose from 476/544 to 544/544. All predeclared selection gates
passed. The largest abrupt-transition horizontal error increased slightly;
the full scores preserve it. This does not establish an advantage over direct
geometric bearing or a conventional temporal filter.

Reproduce a new calibration with:

```bash
python -m experiments.mantis_calibration --output results/new_calibration
```

The raw activation arrays, immutable definitions and independent numerical
verification from `mantis_calibration01` remain local. The original readout and
its calibration fixtures remain unchanged under `results/hybrid_flight_run01`.
Flyvis remains the upstream visual network, with its existing artifact terms;
see [third-party notices](../../THIRD_PARTY_NOTICES.md).
