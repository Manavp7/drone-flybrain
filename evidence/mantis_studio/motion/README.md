# Frozen raw-camera motion evidence

These are unchanged copies of the compact outputs from the actual CPU run on 2026-09-17, originally saved in `results/studio_motion_validation01`. The 12 fixed synthetic translation cases produced 156 neural/decoder observations and 96 post-warmup scored observations: 48 on training-definition textures and 48 on separate held-out textures. No model fitting, sign correction, threshold tuning or temporal lag search followed evaluation.

| Held-out estimator | Vector RMSE, nominal pixels/s |
| --- | ---: |
| Official Flyvis DecoderGAVP | 23.6580950491 |
| Farneback | 0.2437013551 |
| Zero motion | 22.1810730128 |

**The neural decoder did not beat either baseline.** Its moving held-out mean directional cosine was −0.1728039983. The experiment demonstrates software execution, not an improvement in drone control or performance on natural camera footage.

`definition.json` freezes the cases and seeds. `provenance.json` records checkpoint, manifest, implementation and definition digests, runtime versions, timing and unit assumptions. `observations.jsonl` contains compact per-observation responses and synthetic truth. `summary.json` records all per-case and split metrics, artifact digests and the negative conclusion. Neither model weights nor full image/neuron arrays are included.

Verify hashes, counts, causality and all 42 per-case/split vector RMSE values without loading models:

```sh
python evidence/mantis_studio/motion/verify.py
```

The official upstream model is [Flyvis, TuragaLab](https://github.com/TuragaLab/flyvis), by Janne K. Lappalainen, Fabian D. Tschopp, Mason McGill, Jakob H. Macke and Srinivas C. Turaga; see the repository's [upstream notice](../../../THIRD_PARTY_NOTICES.md#flyvis) and [retained MIT source notice](../../../LICENSES/Flyvis-MIT.txt).

Decoder axes are right/up. Nominal image velocity uses the existing Sintel transfer `436 × 24 / 169`, with the vertical sign changed to right/down. This is an uncalibrated unit transfer, not physical speed. Receptor pooling excludes incomplete kernels. Causal 0.02-second recurrence and a fixed 0.5-second warmup are declared in the receipts. [Method and limitations](../../../docs/MANTIS_MOTION.md) include the separate optional brake-only experiment; this benchmark itself has no control authority.
