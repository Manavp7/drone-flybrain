# PX4 and Gazebo camera companion

The new camera path is an observation-only companion: Gazebo image streams → ROS 2 → an atomic latest-frame file → the perception watcher. It has no arming, flight mode, motor, setpoint, or MAVLink interface. It can observe a camera attached to a simulated vehicle while another component operates the simulation. Connecting perception to navigation remains a separate milestone.

This release includes a runnable adapter and unit-tested decoding/synchronisation. This workspace does not have Gazebo, ROS 2, `rclpy`, `sensor_msgs`, or `rosgraph_msgs`; no live ROS camera capture or PX4 flight was executed here. That check was recorded in the private development archive; this public repository contains the adapter source and tests. The separate MuJoCo experiment does not establish PX4 integration.

## Version and model choice

Keep the existing PX4 pin: **v1.16.2**, commit `54f0455ffcd755534539a7cf33a09a20bf71d29d`. The stock camera vehicle is `gz_x500_depth`, autostart **4002**. PX4's versioned documentation describes its forward-facing camera and launch target. The old project's mission runner expects `gz_x500`, autostart **4001**; do not pass the depth vehicle into that runner or bypass its identity checks. Launch the camera vehicle separately. [PX4 v1.16 simulation targets](https://docs.px4.io/v1.16/en/sim_gazebo_gz/index), [PX4 v1.16 camera vehicle](https://docs.px4.io/v1.16/en/sim_gazebo_gz/vehicles#x500-quadrotor-with-depth-camera-front-facing).

The documented platform is Ubuntu 22.04, ROS 2 Humble and Gazebo Harmonic. Install the matching `ros-humble-ros-gzharmonic` packages rather than accidentally mixing the default Humble/Fortress bridge with Harmonic. For installation, use the versioned upstream guide; package availability and ABI compatibility must be checked on the actual host. This companion needs `sensor_msgs` and `rosgraph_msgs`, not the PX4 uXRCE-DDS agent, because it consumes Gazebo camera streams rather than PX4 telemetry. [PX4 ROS 2 setup and Gazebo bridge](https://docs.px4.io/v1.16/en/ros2/user_guide).

On a Linux simulation host, clone into a new directory and use the pinned source:

```bash
git clone --branch v1.16.2 --recursive https://github.com/PX4/PX4-Autopilot.git PX4-Autopilot-v1.16.2
cd PX4-Autopilot-v1.16.2
git rev-parse HEAD
git submodule status --recursive
bash Tools/setup/ubuntu.sh
```

Verify the commit equals the pin above before building. Install ROS 2 Humble following its official instructions and the matching Harmonic bridge:

```bash
sudo apt install ros-humble-ros-gzharmonic
source /opt/ros/humble/setup.bash
```

Use the ROS installation's Python interpreter for capture. If using a virtual environment, create it with `--system-site-packages` so the apt-installed `rclpy` bindings remain importable. Install the project's NumPy/perception requirements in that environment; do not use this workspace's Python 3.12 environment as the Humble Python 3.10 environment.

## Start the camera simulation

Terminal 1, inside the pinned PX4 checkout:

```bash
PX4_SIM_SPEED_FACTOR=1 make px4_sitl gz_x500_depth
```

The model is spawned; this command does not itself establish an autonomous inspection mission. Use simulation speed factor 1 for this initial camera timing test. The companion reports observations only; it does not fly or select objects to follow.

The default world is sparse. Depth can see its surfaces, but a COCO object detector may detect nothing. Boxes in the original hard course do not become semantic classes merely because they are visible. A later recognition benchmark needs representative visual assets, labelled views and adverse lighting, and a separate flight mission. The old 2,500 geometric trials are not a camera detection accuracy test.

## Discover topics and create the bridge

Terminal 2, with the ROS environment sourced:

```bash
gz topic -l
```

From that list select the vehicle's RGB image, depth image, camera info and simulator clock topics. Do not copy a guessed sensor path: model instance and world names vary. Inspect each selected topic's type using `gz topic -i -t /the/topic/from/the/list`. Expect `gz.msgs.Image`, `gz.msgs.CameraInfo` and `gz.msgs.Clock` respectively. If both `/clock` and a world clock exist, bridge exactly one simulator clock into ROS `/clock`.

In the project directory, set the following shell variables to the exact discovered topic strings. These are placeholders to fill, not literal topic names:

```bash
RGB_TOPIC='/paste/discovered/rgb/topic'
DEPTH_TOPIC='/paste/discovered/depth/topic'
INFO_TOPIC='/paste/discovered/camera_info/topic'
CLOCK_TOPIC='/paste/discovered/clock/topic'
python -m integrations.gazebo.camera_bridge bridge-config \
  --rgb-topic "$RGB_TOPIC" \
  --depth-topic "$DEPTH_TOPIC" \
  --info-topic "$INFO_TOPIC" \
  --clock-topic "$CLOCK_TOPIC" \
  --output results/camera_bridge.yaml
ros2 run ros_gz_bridge parameter_bridge --ros-args \
  -p config_file:=results/camera_bridge.yaml
```

The generated YAML maps these streams to `/inspection/rgb`, `/inspection/depth`, `/inspection/camera_info` and `/clock`, with one-entry queues and **GZ_TO_ROS** direction only. Gazebo documents these YAML keys and the unidirectional bridge format. [Gazebo Harmonic ROS integration](https://gazebosim.org/docs/harmonic/ros2_integration/).

`ros_gz_image image_bridge` is an alternative image-only transport, with `sensor_data` QoS available. If selecting it, bridge CameraInfo and clock separately and do not run a second publisher for the same ROS image topic. The supplied generated-config path uses `ros_gz_bridge` for all four streams to make names and direction explicit. [Upstream image bridge usage](https://github.com/gazebosim/ros_gz/blob/ros2/ros_gz_image/README.md).

Check the receiving topics before starting inference:

```bash
ros2 topic list -t
ros2 topic info /inspection/rgb --verbose
ros2 topic hz /inspection/rgb
ros2 topic echo /inspection/camera_info --once
ros2 topic echo /clock --once
```

Inspect RGB and depth together with an image viewer. Verify the chosen streams are pixel-aligned to the same optical camera, share acquisition timing and resolution, and use the expected intrinsics. The adapter does not compute an RGB-to-depth calibration or rectify real cameras.

## Capture and detect

Terminal 3, project directory:

```bash
python -m integrations.gazebo.camera_bridge doctor
python -m integrations.gazebo.camera_bridge capture \
  --spool results/camera_spool \
  --rgb-topic /inspection/rgb \
  --info-topic /inspection/camera_info \
  --ros-args -p use_sim_time:=true
```

This starts RGB-only capture. After verifying registered depth, use the following instead:

```bash
python -m integrations.gazebo.camera_bridge capture \
  --spool results/camera_spool \
  --rgb-topic /inspection/rgb \
  --depth-topic /inspection/depth \
  --info-topic /inspection/camera_info \
  --registered-depth \
  --ros-args -p use_sim_time:=true
```

`--registered-depth` is an explicit calibration assertion, not automatic calibration. Without it, the adapter never attaches depth to detections. With it, every paired frame must also satisfy dimensions, frame ID, calibration and timestamp checks. No depth match means an RGB-only output, not an assumed clear view.

Terminal 4, project directory, with the validated ONNX model and its expected SHA-256:

```bash
python -m perception watch \
  --spool results/camera_spool \
  --model models/yolox_s.onnx \
  --sha256 YOUR_VERIFIED_MODEL_SHA256 \
  --output results/live
```

See the perception README for model acquisition and the actual artifact filename. Run only one capture writer per spool. Capture and inference must run on the same host, because `received_monotonic_s` is a host-local timestamp. `latest.npz` deliberately overwrites unconsumed older frames; this is bounded live ingestion, not an archival recording of every source frame. The perception watcher logs the frames it actually processes.

## Frame and timing contract

The ROS Image specification defines acquisition timestamps, row byte stride and camera optical axes: x right, y down, z forward. CameraInfo supplies calibration; mismatched image and CameraInfo frame IDs are not interchangeable. This decoder honours `step` padding and `is_bigendian`. It accepts only `rgb8`, `bgr8`, `32FC1` and `16UC1`; it rejects compressed/Bayer/unknown encodings rather than guessing. [ROS Image definition](https://docs.ros.org/en/rolling/p/sensor_msgs/msg/Image.html), [CameraInfo definition](https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/msg/CameraInfo.msg).

Depth is **optical Z**, not Euclidean range or world position. `32FC1` is metres; `16UC1` is millimetres converted to float32 metres. Zero/nonfinite/negative depth becomes NaN and remains unknown. The bridge conservatively treats even infinite sensor returns as unknown rather than open space. [REP 118 depth-image representation](https://reps.openrobotics.org/rep-0118/).

Each file is an NPZ readable with `numpy.load(path, allow_pickle=False)`:

| Key | Shape/type | Meaning |
|---|---|---|
| `image_rgb` | H × W × 3 uint8 | Owned RGB pixels |
| `capture_time_s` | float scalar | RGB ROS acquisition time |
| `received_monotonic_s` | float scalar | RGB host arrival time |
| `capture_age_at_receive_s` | float scalar | Conservative source-age bound: nonnegative simulator clock minus capture time, plus elapsed host time since that clock sample |
| `sequence` | int64 scalar | Increasing emitted-frame number within stream |
| `stream_id` | Unicode scalar | New UUID at capture start and source clock reset |
| `frame_id` | Unicode scalar | Camera optical frame identifier |
| `clock_domain` | Unicode scalar | `ros` |
| `registration_verified` | bool scalar | Explicit registration opt-in plus current pairing checks |
| `camera_intrinsics` | Optional float64[4] | fx, fy, cx, cy; omitted if unavailable |
| `depth_m` | Optional H × W float32 | Registered optical-Z metres |
| `depth_time_s` | Optional float scalar | Attached depth acquisition time |

The adapter holds one latest depth, one calibration and one pending RGB. A matching pair emits immediately. Otherwise the pending RGB emits after at most 40 ms of pairing wait plus executor scheduling delay; newer RGB frames share that deadline, so repeated RGB arrivals cannot starve output during depth dropout. Inference runs in another process and cannot block the capture callback queue.

Default checks reject source age over 300 ms, a source timestamp over 30 ms ahead of `/clock`, clock silence over one wall second, duplicate/out-of-order frames and unsupported image dimensions. The source-age field includes elapsed host time since the last `/clock` observation, so a stale clock cannot make an old queued camera message look fresh. This uses the documented real-time factor of 1; pausing simulation can conservatively reject otherwise usable frames. The watcher adds elapsed host time after RGB arrival to this age. Clock silence below the one-second transport limit can still exceed the tighter 300 ms frame freshness budget and reject capture. `/clock` must be observed before capture; when it rewinds, cached depth/calibration and pending RGB clear and `stream_id` changes. A smaller timestamp in one image alone is dropped rather than treated as a clock reset.

Depth pairing requires timestamp difference ≤30 ms, equal optical frame IDs, equal pixel dimensions, calibration acquired within two source seconds, positive finite focal lengths, a canonical zero-skew K matrix, no nonzero distortion, and no unhandled ROI/binning. Consumers must continue treating missing depth, missing calibration, stale input and registration failures as unknown. Equal image shape alone is insufficient.

## Validation and next runtime gate

The ROS-free tests cover endian conversions, padded rows, dimensions and payload bounds, depth units/invalids, intrinsics validation, stale/future/out-of-order frames, clock reset, RGB fallback during high-rate depth dropout, registration failures, atomic writes and one-way topic configuration:

```bash
python -m unittest discover -s tests -p test_gazebo_v4.py -v
```

These checks verify adapter logic. The outstanding host gate is a real pinned PX4/Harmonic/Humble session with recorded source topics, visual registration checks, capture-to-inference timing under load, missing-frame tests and truth-labelled detections. It must run before reporting camera-in-flight validation. Neither 2D detections nor depth box statistics by themselves provide localisation, obstacle avoidance, map reconstruction, world coordinates or research-model flight control.
