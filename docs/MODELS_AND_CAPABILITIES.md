# Model roles and capabilities

The current connected experiment uses YOLOX-tiny for recognition, the actual recurrent Flyvis visual network for neural cue processing, a learned readout for bearing, registered depth for standoff/stopping, and a conventional autopilot for stability. [Current results](MOTOR_FLIGHT_TEST.md) and [model setup](MODEL_SETUP.md) describe the tested configuration.

| Component | Output | Boundary |
|---|---|---|
| YOLOX | COCO classes, confidence and boxes | Confidence is not a calibrated probability of safety; unfamiliar aerial views can fail |
| Geometric/clothing tracker | Temporary target IDs | Camera-motion compensation does not establish persistent identity across occlusion |
| Flyvis | Recurrent modeled neural activity | Receives an engineered target rectangle; no independent semantic recognition or whole-fly motor control |
| Frozen ridge readout | Target bearing estimate from neural features | Rejects unsupported cue geometry; no proof of biological-feature superiority |
| Registered depth | Visible-surface range and observed stopping corridor | Requires calibrated alignment and pose; unseen space remains unknown |
| Conventional autopilot | Four bounded motor forces | Ideal-state simulation, not qualified aircraft software |

YOLOX uses its official ONNX preprocessing: float BGR input, top-left aspect-preserving resize, padding 114, raw grid/stride decoding and objectness times class score. Tiny uses 416-pixel input; S uses 640. The repository retains both interfaces but the current flight experiment pins Tiny and its exact model digest.

The camera optical axes are X right, Y down, Z forward. Depth is optical Z in metres. The new motor-flight guardian is separate from the older perception module's sampled forward advisory. Neither supplies global route planning.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for artifact/source distinctions and credits. Pretrained weights are acquired separately. Earlier unsuccessful experiments are summarized in [research history](RESEARCH_HISTORY.md).

Primary references: [YOLOX ONNX interface](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/demo/ONNXRuntime/README.md), [Flyvis custom stimuli](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/), [MuJoCo 3.2.7](https://mujoco.readthedocs.io/en/3.2.7/python.html).
