"""ROS-free ingestion checks; these do not claim a live camera/SITL run."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as S
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from integrations.gazebo.camera_bridge import (
    AtomicFrameWriter, CameraError, LatestFrameSynchronizer,
    bridge_configuration, camera_intrinsics, decode_image, environment_status,
    stamp_seconds,
)


def header(t=1.0, frame="camera_optical"):
    ns = round(t * 1e9)
    return S(frame_id=frame, stamp=S(sec=ns // 10**9, nanosec=ns % 10**9))


def image(values, encoding="rgb8", t=1.0, frame="camera_optical", big=False, padding=0):
    values = np.asarray(values)
    dtype = np.dtype("u1" if encoding in ("rgb8", "bgr8") else (">" if big else "<") + ("f4" if encoding == "32FC1" else "u2"))
    values = values.astype(dtype)
    rows = [row.tobytes() + b"\xff" * padding for row in values]
    return S(header=header(t, frame), width=values.shape[1], height=values.shape[0],
             encoding=encoding, is_bigendian=int(big), step=len(rows[0]), data=b"".join(rows))


def calibration(t=1.0, frame="camera_optical", width=3, height=2):
    return S(header=header(t, frame), width=width, height=height,
             k=[10., 0., 1., 0., 11., .5, 0., 0., 1.], d=[0.] * 5,
             binning_x=0, binning_y=0,
             roi=S(x_offset=0, y_offset=0, width=0, height=0))


def rgb(t=1.0, frame="camera_optical"):
    return image(np.arange(18).reshape(2, 3, 3), t=t, frame=frame)


def depth(t=1.0, frame="camera_optical"):
    return image([[2., 3., 4.], [2., 3., 4.]], "32FC1", t=t, frame=frame)


def synchronizer(registered=True):
    sync = LatestFrameSynchronizer(registration_verified=registered)
    sync.ingest_clock(1., 10.)
    sync.ingest_info(calibration())
    return sync


class DecodeTests(unittest.TestCase):
    def test_rgb_padding_and_owned_memory(self):
        source = np.arange(18).reshape(2, 3, 3)
        output = decode_image(image(source, padding=7))
        np.testing.assert_array_equal(output, source)
        self.assertTrue(output.flags.owndata)

    def test_bgr_swizzle(self):
        output = decode_image(image([[[1, 2, 3], [4, 5, 6]]], "bgr8", padding=2))
        np.testing.assert_array_equal(output, [[[3, 2, 1], [6, 5, 4]]])

    def test_float_depth_endian_padding_invalids(self):
        for big in (False, True):
            output = decode_image(image([[1.25, float("nan"), 0.], [-1., float("inf"), 12.5]],
                                        "32FC1", big=big, padding=3))
            self.assertEqual(output.dtype, np.float32)
            np.testing.assert_allclose(output, [[1.25, np.nan, np.nan], [np.nan, np.nan, 12.5]], equal_nan=True)

    def test_uint16_mm_depth_both_endians(self):
        for big in (False, True):
            output = decode_image(image([[1000, 2500, 0], [1, 65535, 999]], "16UC1", big=big, padding=3))
            np.testing.assert_allclose(output, [[1., 2.5, np.nan], [.001, 65.535, .999]], rtol=1e-6, equal_nan=True)

    def test_bad_dimensions_stride_length_and_encoding(self):
        for name, value in (("width", 0), ("width", 999999), ("height", 3000),
                            ("step", 1), ("step", 100000000), ("data", b"bad"),
                            ("encoding", "jpeg"), ("is_bigendian", 2)):
            msg = rgb()
            setattr(msg, name, value)
            with self.subTest(field=name), self.assertRaises(CameraError):
                decode_image(msg)

    def test_timestamp_nanos_and_negative(self):
        self.assertEqual(stamp_seconds(S(sec=1, nanosec=500_000_000)), 1.5)
        for sec, ns in ((-1, 0), (0, -1), (0, 1_000_000_000)):
            with self.assertRaises(CameraError):
                stamp_seconds(S(sec=sec, nanosec=ns))

    def test_intrinsics_and_reject_uncalibrated_distorted_roi_binning(self):
        self.assertEqual(camera_intrinsics(calibration()).intrinsics, (10., 11., 1., .5))
        for field, value in (("k", [0.] * 9), ("k", [float("nan")] * 9),
                             ("d", [.1, 0., 0., 0., 0.]), ("binning_x", 2),
                             ("roi", S(x_offset=1, y_offset=0, width=1, height=2))):
            msg = calibration()
            setattr(msg, field, value)
            with self.subTest(field=field), self.assertRaises(CameraError):
                camera_intrinsics(msg)


class SynchronizerTests(unittest.TestCase):
    def test_matching_depth_rgb_and_contract(self):
        sync = synchronizer()
        self.assertIsNone(sync.ingest("depth", depth(), 10.01))
        packet = sync.ingest("rgb", rgb(), 10.02)
        self.assertTrue(packet["registration_verified"])
        self.assertEqual(float(packet["depth_time_s"]), 1.)
        self.assertAlmostEqual(float(packet["capture_age_at_receive_s"]), .02)
        self.assertEqual(float(packet["received_monotonic_s"]), 10.02)
        self.assertEqual(str(packet["clock_domain"]), "ros")
        self.assertEqual(packet["image_rgb"].shape, (2, 3, 3))
        np.testing.assert_allclose(packet["camera_intrinsics"], [10, 11, 1, .5])
        self.assertIsNone(sync.flush(10.03))

    def test_depth_after_rgb_pairs_once(self):
        sync = synchronizer()
        self.assertIsNone(sync.ingest("rgb", rgb(), 10.01))
        packet = sync.ingest("depth", depth(t=1.005), 10.02)
        self.assertTrue(packet["registration_verified"])
        self.assertIsNone(sync.flush(10.1))
        self.assertEqual(sync.counters["published"], 1)

    def test_missing_depth_emits_rgb_bounded(self):
        sync = synchronizer()
        self.assertIsNone(sync.ingest("rgb", rgb(), 10.01))
        self.assertIsNone(sync.flush(10.04))
        packet = sync.flush(10.051)
        self.assertFalse(packet["registration_verified"])
        self.assertNotIn("depth_m", packet)

    def test_depth_dropout_at_high_rgb_rate_does_not_starve(self):
        sync = synchronizer()
        packets = []
        for i in range(10):
            stamp, wall = 1. + i * .02, 10. + i * .02
            sync.ingest_clock(stamp, wall)
            packet = sync.ingest("rgb", rgb(t=stamp), wall)
            if packet:
                packets.append(packet)
        self.assertGreaterEqual(len(packets), 3)
        self.assertIsNotNone(sync.pending_rgb)
        self.assertGreater(sync.counters["replaced_rgb"], 0)
        self.assertTrue(all(not p["registration_verified"] for p in packets))

    def test_no_registration_inferred_from_equal_shape(self):
        sync = synchronizer(False)
        sync.ingest("depth", depth(), 10.01)
        packet = sync.ingest("rgb", rgb(), 10.02)
        self.assertFalse(packet["registration_verified"])
        self.assertNotIn("depth_m", packet)

    def test_mismatched_frames_shapes_time_and_info_no_depth(self):
        cases = [depth(frame="another_camera"),
                 image([[2., 3.]], "32FC1"), depth(t=.9)]
        for candidate in cases:
            sync = synchronizer()
            sync.ingest("depth", candidate, 10.01)
            sync.ingest("rgb", rgb(), 10.02)
            packet = sync.flush(10.07)
            self.assertFalse(packet["registration_verified"])
            self.assertNotIn("depth_m", packet)
        sync = synchronizer()
        sync.calibration = replace(sync.calibration, frame_id="wrong")
        sync.ingest("depth", depth(), 10.01)
        sync.ingest("rgb", rgb(), 10.02)
        packet = sync.flush(10.07)
        self.assertNotIn("camera_intrinsics", packet)
        self.assertNotIn("depth_m", packet)

    def test_unknown_info_emits_rgb_without_nan_intrinsics(self):
        sync = synchronizer()
        sync.calibration = None
        sync.ingest("rgb", rgb(), 10.01)
        packet = sync.flush(10.06)
        self.assertNotIn("camera_intrinsics", packet)

    def test_source_age_survives_receive(self):
        sync = synchronizer(False)
        sync.ingest_clock(1.2, 10.01)
        packet = sync.ingest("rgb", rgb(), 10.02)
        self.assertAlmostEqual(float(packet["capture_age_at_receive_s"]), .21)

    def test_aged_clock_cannot_refresh_buffered_camera_frame(self):
        sync = LatestFrameSynchronizer(registration_verified=False)
        sync.ingest_clock(10., 100.)
        # Clock silence is below its 1s transport limit, but the frame is still
        # older than the tighter 300ms freshness budget at real-time factor 1.
        with self.assertRaisesRegex(CameraError, "stale"):
            sync.ingest("rgb", rgb(10.), 100.9)
        self.assertEqual(sync.counters["published"], 0)

    def test_clock_observation_age_is_retained_for_downstream_deadline(self):
        sync = LatestFrameSynchronizer(registration_verified=False)
        sync.ingest_clock(10., 100.)
        packet = sync.ingest("rgb", rgb(9.95), 100.1)
        self.assertAlmostEqual(float(packet["capture_age_at_receive_s"]), .15)
        # Slightly future camera ordering cannot cancel elapsed host time.
        sync = LatestFrameSynchronizer(registration_verified=False)
        sync.ingest_clock(10., 100.)
        packet = sync.ingest("rgb", rgb(10.02), 100.1)
        self.assertAlmostEqual(float(packet["capture_age_at_receive_s"]), .1)

    def test_reject_stale_future_and_silent_clock(self):
        for source_stamp, wall in ((.5, 10.01), (1.2, 10.01), (1., 11.1)):
            sync = synchronizer()
            with self.assertRaises(CameraError):
                sync.ingest("rgb", rgb(source_stamp), wall)
        sync = LatestFrameSynchronizer()
        with self.assertRaises(CameraError):
            sync.ingest("rgb", rgb(), 10.)

    def test_pending_stale_dropped(self):
        sync = synchronizer()
        sync.ingest("rgb", rgb(), 10.01)
        self.assertIsNone(sync.flush(10.5))
        self.assertIsNone(sync.pending_rgb)

    def test_old_or_duplicate_image_does_not_reset_stream(self):
        sync = synchronizer(False)
        original_stream = sync.stream_id
        sync.ingest("rgb", rgb(), 10.01)
        for t in (1., .99):
            with self.assertRaises(CameraError):
                sync.ingest("rgb", rgb(t), 10.02)
        self.assertEqual(sync.stream_id, original_stream)
        self.assertEqual(sync.sequence, 1)

    def test_clock_reset_rotates_stream_and_clears_depth_info_pending(self):
        sync = synchronizer()
        sync.ingest("depth", depth(), 10.01)
        sync.ingest("rgb", rgb(t=1.01), 10.02)
        old_stream = sync.stream_id
        sync.ingest_clock(.1, 10.03)
        self.assertNotEqual(sync.stream_id, old_stream)
        self.assertIsNone(sync.depth)
        self.assertIsNone(sync.pending_rgb)
        self.assertIsNone(sync.calibration)
        self.assertEqual(sync.sequence, 0)
        with self.assertRaises(CameraError):
            sync.ingest("depth", depth(), 10.04)
        sync.ingest("rgb", rgb(.1), 10.04)
        packet = sync.flush(10.09)
        self.assertFalse(packet["registration_verified"])
        self.assertEqual(int(packet["sequence"]), 1)

    def test_local_clock_reverse_rejected(self):
        sync = synchronizer()
        sync.ingest("rgb", rgb(), 10.01)
        with self.assertRaises(CameraError):
            sync.flush(10.)
        with self.assertRaises(CameraError):
            sync.ingest_clock(1.1, 9.)

    def test_invalid_info_clears_old_calibration(self):
        sync = synchronizer()
        invalid = calibration()
        invalid.k = [0.] * 9
        with self.assertRaises(CameraError):
            sync.ingest_info(invalid)
        self.assertIsNone(sync.calibration)

    def test_config_limits(self):
        for kwargs in ({"pair_tolerance_s": -1}, {"max_pair_wait_s": 1},
                       {"max_frame_age_s": float("nan")}, {"max_frame_age_s": 0}):
            with self.assertRaises(CameraError):
                LatestFrameSynchronizer(**kwargs)


class WriterConfigTests(unittest.TestCase):
    def test_atomic_npz_roundtrip_without_pickle_and_single_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = AtomicFrameWriter(Path(directory))
            sync = synchronizer(False)
            first = sync.ingest("rgb", rgb(), 10.01)
            writer.write(first)
            with np.load(writer.path, allow_pickle=False) as data:
                np.testing.assert_array_equal(data["image_rgb"], first["image_rgb"])
                self.assertEqual(data["sequence"].item(), 1)
            second = sync.ingest("rgb", rgb(1.01), 10.02)
            writer.write(second)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["latest.npz"])
            with np.load(writer.path, allow_pickle=False) as data:
                self.assertEqual(data["sequence"].item(), 2)
            with self.assertRaises(CameraError):
                writer.write({"bad": np.asarray([{}], dtype=object)})

    def test_bridge_packet_loads_through_actual_perception_contract(self):
        from perception.__main__ import load_sample
        with tempfile.TemporaryDirectory() as directory:
            sync = synchronizer()
            sync.ingest("depth", depth(), 10.01)
            packet = sync.ingest("rgb", rgb(), 10.02)
            writer = AtomicFrameWriter(Path(directory))
            writer.write(packet)
            sample = load_sample(writer.path)
            self.assertTrue(sample.registration_verified)
            self.assertEqual(sample.sequence, 1)
            self.assertEqual(sample.clock_domain, "ros")
            self.assertEqual(sample.stream_id, sync.stream_id)
            np.testing.assert_allclose(sample.depth_m, [[2, 3, 4], [2, 3, 4]])

    def test_failed_atomic_replace_preserves_old_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = AtomicFrameWriter(Path(directory))
            writer.write({"sequence": np.asarray(1)})
            with patch("integrations.gazebo.camera_bridge.os.replace", side_effect=OSError("test disk error")):
                with self.assertRaises(OSError):
                    writer.write({"sequence": np.asarray(2)})
            with np.load(writer.path, allow_pickle=False) as data:
                self.assertEqual(data["sequence"].item(), 1)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ["latest.npz"])

    def test_bridge_config_is_read_only_and_maps_discovered_topics(self):
        config = bridge_configuration("/model/x/rgb", "/model/x/depth", "/model/x/info", "/world/test/clock")
        self.assertEqual(config.count('direction: "GZ_TO_ROS"'), 4)
        self.assertIn('gz_topic_name: "/world/test/clock"', config)
        self.assertIn('ros_topic_name: "/inspection/rgb"', config)
        self.assertNotIn("BIDIRECTIONAL", config)
        self.assertNotIn("/fmu", config)
        for topics in (("rgb", None, "/info"), ("/rgb", "/rgb", "/info"), ("/bad\ntopic", None, "/info")):
            with self.assertRaises(CameraError):
                bridge_configuration(*topics)

    def test_doctor_does_not_claim_actual_runtime(self):
        status = environment_status()
        self.assertFalse(status["actual_sitl_run"])
        self.assertFalse(status["actual_ros_camera_run"])
        self.assertTrue(status["observation_only"])
        self.assertEqual(status["autostart"], 4002)


if __name__ == "__main__":
    unittest.main()
