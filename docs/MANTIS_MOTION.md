# Raw camera motion experiment

`RawMotionExperiment` feeds camera luminance into a separate instance of the official pretrained Flyvis visual network and its frozen `DecoderGAVP` optical-flow decoder. It has its own recurrent state. It does not use the selected person's box, the YOLO salience mask, simulator truth, or a trained local replacement decoder.

This module reports a sensor diagnostic. It does not issue flight commands. The interactive studio can optionally evaluate a separate brake-only heuristic using this diagnostic; that experiment does not establish collision detection or better control. Depth safety checks remain the final gate.

## Timing and input

Camera images are RGB `uint8` arrays. The center square is converted to luminance and resized to 391×391 using antialiased bilinear interpolation before the 721-site `BoxEye` transform. Cropped peripheral pixels are declared in each result.

Integration uses 0.02-second steps. The previous image is held between captures; a new image is first presented at the first integration grid point at or after its capture time. Its output is available at `response_time_s`, after one further integration step. A caller must also account for measured computation latency before releasing the result into a running simulator.

Captures must increase by at least 0.02 seconds. Gaps over 0.65 seconds reset the state and restart the fixed 0.5-second warmup. Invalid, warming-up or stale results are not valid motion evidence. Wall execution times are reported separately; they do not establish a real-time deadline.

## Units

The decoder outputs two channels: right and up. The receipt reports the mean and median vector, mean magnitude and root-mean-square magnitude over receptors whose full 13×13 kernel lies inside the image.

`nominal_velocity_px_s` converts the median decoder value using `436 × 24 / 169` and reverses the vertical sign to image coordinates (right/down). This preserves the existing Sintel target convention as a declared transfer assumption. It is neither measured physical speed nor calibrated drone motion.

The conventional comparison runs Farneback on the same resized luminance images, divides its displacement by the actual frame interval, applies the existing target normalization and `BoxEye` sum, then uses the same interior receptor pooling. Mean T4/T5 population activities are diagnostic summaries, not hand-written substitute flow predictions.

## Frozen synthetic benchmark

From the repository root, using a compatible Python environment:

```sh
python -m experiments.mantis_motion \
  --manifest models/flyvis_0000_000.manifest.json \
  --output results/studio_motion_validation01
```

The output directory must be new. The definition is saved before model loading or inference. The suite contains stationary, left, right, up, down and diagonal translations, each on one training-definition texture seed and one separate held-out seed. No fitting, threshold selection, sign correction, temporal lag search or tuning uses either split. Each clip spans 1.2 seconds at 10 camera frames per second, with a fixed 0.5-second warmup.

Known texture velocities provide ground truth in the 391-square image. Every post-warmup vector is scored against the official neural decoder, Farneback and a zero-motion baseline. Direction coverage accompanies cosine scores so zero predictions cannot quietly disappear from the evaluation. Static scenes are included in vector error metrics.

Only compact definitions, provenance, per-observation summaries and scores are saved. Full images and neuron arrays are not accumulated. A failed or negative result remains a failed or negative result in the receipt; software execution alone is not evidence that neural motion improves flight. Synthetic global translations do not establish generalization to natural videos, independent moving people, looming obstacles, real cameras or physical aircraft.

## Measured local result, 2026-09-17

The frozen 12-case suite completed using the actual official pretrained network and decoder: 156 camera observations, 96 post-warmup scored observations and about 94.4 seconds of wall time including initial loading. The original compact artifacts occupy about 341 KiB in `results/studio_motion_validation01`. Byte-identical copies, source/artifact hashes and a model-free recomputation script are retained in [the motion evidence bundle](../evidence/mantis_studio/motion/README.md).

| Held-out estimator | Vector RMSE, nominal pixels/s |
| --- | ---: |
| Official Flyvis decoder | 23.6581 |
| Farneback | 0.2437 |
| Zero-motion baseline | 22.1811 |

The neural decoder did not beat either baseline. Its mean directional cosine on moving held-out rows was −0.1728; on the stationary held-out texture it still produced about 5.33 nominal pixels/s of vector error. No tuning or sign correction followed this result. The software pathway runs, but this experiment supplies no evidence that its raw-motion signal should improve flight control.

The pretrained network and decoder are upstream [Flyvis](https://github.com/TuragaLab/flyvis), rather than original Mantis model weights. Contributor and dependency notices are retained in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md#flyvis) and [LICENSES/Flyvis-MIT.txt](../LICENSES/Flyvis-MIT.txt).
