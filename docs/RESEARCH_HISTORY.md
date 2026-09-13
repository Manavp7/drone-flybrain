# Research history and negative results

This repository preserves multiple stages of an experimental drone-perception project. The current public demo is the connected MuJoCo motor-flight stage.

1. **YOLO on overhead footage:** the detector missed people and sometimes classified them as unrelated COCO objects. Cropped inference and clothing association improved specific cases, but dependable overhead person recognition was not achieved.
2. **Pure Flyvis tracking:** an initial neural-motion readout failed three controlled motion tests. A later spatial neural-template tracker passed six fresh synthetic cases, but a new real-person segment still passed only 1/9 overlap checks. A matched image-brightness template performed similarly on synthetic cases; no neural-feature superiority was established.
3. **YOLO plus Flyvis guidance:** YOLO supplied a target cue to the actual visual network, whose learned readout drove a point-mass simulation. Four primary approach/motion/loss cases passed; zero neural features prevented movement. This preceded motor-driven flight physics.
4. **Recorded YouTube input:** a ground-level street segment retained one person in 80/80 observations and passed 7/7 approximate annotated overlap checks. Eight seconds of media took 58.38 seconds to process. Its commands were diagnostic and never applied to aircraft. The footage is not redistributed in this source release.
5. **Motor-driven camera flight:** the current nine-case MuJoCo suite adds body-mounted RGB-D, a conventional autopilot, four motors, registered depth stopping and explicit measured-delay scheduling. [Current results](MOTOR_FLIGHT_TEST.md) describe its controlled scope.

The first motor-flight development run exposed a camera-motion association failure: YOLO still detected the person, but yaw/pitch changed the box enough to assign a new ID. The current compensation reprojects prior observations using registered depth and measured camera pose before association. It does not silently reselect another person or renew stale track timestamps.

The raw historical datasets and multi-gigabyte run archives remain outside the public repository. The [public evidence subset](../evidence/README.md) reports only what is included and does not claim that these earlier failures have been solved generally.
