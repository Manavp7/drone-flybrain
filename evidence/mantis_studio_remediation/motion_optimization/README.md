# Raw-motion startup and reset optimization

The recorded live failures were dominated by work that did not need to repeat during camera processing: the first flow call lazily constructed the official decoder, then large camera gaps repeatedly recomputed the fixed 50-step blank initial state. Those resets restarted the 0.5-second warmup. The optimization loads/evaluates the official frozen decoder during initialization and restores a separate, exact clone of the original blank node/edge state on each reset.

The neural dynamics, preprocessing, decoder weights, causal 0.02-second integration, 0.65-second gap limit and 0.5-second warmup are unchanged. It neither invents flow nor relaxes control freshness checks.

## Actual serialized CPU measurement

`profile_equivalence.json` is the unchanged receipt from one before/after run. Fifteen captures use fixed independent texture seed 9103, variable timestamps and one gap reset; four additional explicit resets test restoration. Every full dynamic state, all 2×721 flow values, initial-state tensors, clock/validity decisions and population summaries matched exactly: maximum differences were zero.

| Measurement | Before | After |
| --- | ---: | ---: |
| First timed camera processing | 2.151880 s | 0.045451 s |
| Camera processing with gap reset | 0.551139 s | 0.028698 s |
| Median explicit reset | 0.463353 s | 0.000074 s |
| Median ordinary camera processing | 0.175878 s | 0.144280 s |

Decoder initialization cost moved into loading; it did not disappear. Recorded construction times were 20.096 s before and 8.874 s after, with both tested sequentially in one process. Import/model caches and measurement order prevent interpreting those construction numbers, or the ordinary-call variation, as isolated algorithmic speedups. These measurements do not guarantee real-time behavior or complete studio latency.

**There is no accuracy gain in this change.** The original negative motion benchmark and its reported baseline comparison remain valid and untouched. Combined scheduling and flight behavior require separate evaluation.

## Reproduce

`mantis_motion_before.py` preserves the exact original module. `packaging.json` records its digest, the measured current implementation digest, supporting source digests, model digests and runtime versions. The adapted runner resolves paths relative to the repository and writes to a new output directory; it cannot overwrite this receipt.

From the repository root, with the documented Flyvis environment and acquired locked model:

```sh
python evidence/mantis_studio_remediation/motion_optimization/profile_equivalence.py \
  --manifest models/flyvis_0000_000.manifest.json \
  --output results/motion_optimization_recheck01
```

The new receipt records the then-current source hashes. Compare them with this bundle before attributing new timings to the same implementation. Running against changed supporting modules also changes the experiment. Model-free integrity checks can recompute the SHA-256 values listed in `packaging.json`.

The original upstream network and flow decoder are [Flyvis, TuragaLab](https://github.com/TuragaLab/flyvis). Existing [upstream notices](../../../THIRD_PARTY_NOTICES.md#flyvis) apply. No model weights or full neural/video recordings are distributed in this bundle.
