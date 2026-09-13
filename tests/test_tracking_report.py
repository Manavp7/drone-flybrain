"""Report contract checks use generated frames, not neural-accuracy fixtures."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from experiments import tracking_report as report


def row(sequence, timestamp, detections=()):
    return {"sequence": sequence, "capture_time_s": timestamp, "status": "ok",
            "inference_executed": True, "processing_ms": 20., "detections": list(detections)}


def detection(track_id=1):
    return {"track_id": track_id, "class_id": 0, "label": "person", "confidence": .8,
            "bbox_xyxy": [1.,1.,10.,10.]}


class SamplingTests(unittest.TestCase):
    def test_sampled_sequence_and_capture_time_determine_source_and_playback_fps(self):
        rows = [row(2,2/30),row(8,8/30),row(14,14/30)]
        result = report.sampling_metadata(rows,{"frames":3,"sample_every":6})
        self.assertAlmostEqual(result["nominal_source_fps"],30.)
        self.assertAlmostEqual(result["preview_fps"],5.)
        self.assertEqual(result["first_source_frame"],2)

    def test_irregular_reordered_and_mismatched_sampling_rejected(self):
        cases = [([row(0,0),row(3,.1),row(7,.2)],{}),
                 ([row(3,0),row(2,.1),row(1,.2)],{}),
                 ([row(0,0),row(3,.1),row(6,.2)],{"sample_every":2}),
                 ([row(0,0),row(3,0),row(6,.2)],{})]
        for rows, extra in cases:
            with self.subTest(rows=rows,extra=extra), self.assertRaises(ValueError):
                report.sampling_metadata(rows,{"frames":3,**extra})

    def test_single_frame_uses_recorded_or_decoder_fps(self):
        rows = [row(3,.1)]
        self.assertEqual(report.sampling_metadata(rows,{"frames":1,"sample_every":3,"nominal_source_fps":30})["preview_fps"],10.)
        self.assertEqual(report.sampling_metadata(rows,{"frames":1},24.)["preview_fps"],24.)
        with self.assertRaises(ValueError):
            report.sampling_metadata(rows,{"frames":1})

    def test_track_gaps_count_processed_frames_and_spans_use_recorded_time(self):
        rows = [row(0,0,[detection()]),row(6,.2,[detection()]),row(12,.4),
                row(18,.6,[detection()]),row(24,.8,[detection()])]
        track = report.summarize_tracks(rows)[0]
        self.assertEqual(track["observed_frames"],4)
        self.assertEqual(track["missing_frames_within_span"],1)
        self.assertEqual(track["gap_events"],1)
        self.assertEqual(track["longest_consecutive_frames"],2)
        self.assertAlmostEqual(track["longest_consecutive_span_s"],.2)
        self.assertEqual((track["first_frame"],track["last_frame"]),(0,24))
        self.assertEqual((track["first_processed_frame"],track["last_processed_frame"]),(0,4))

    def test_duplicate_track_and_class_changes_rejected(self):
        changed = {**detection(),"class_id":1}
        for rows in ([row(0,0,[detection(),detection()])],
                     [row(0,0,[detection()]),row(1,.1,[changed])]):
            with self.assertRaises(ValueError):
                report.summarize_tracks(rows)

    def test_decode_matches_exact_source_sequences_without_reusing_boxes(self):
        class Capture:
            def __init__(self): self.index = -1
            def read(self):
                self.index += 1
                return self.index < 8, self.index
        capture = Capture()
        rows = [row(1,.1),row(4,.4),row(7,.7)]
        actual = list(report.logged_source_frames(capture,rows))
        self.assertEqual([frame for _,frame in actual],[1,4,7])
        self.assertEqual([r["sequence"] for r,_ in actual],[1,4,7])
        with self.assertRaises(RuntimeError):
            list(report.logged_source_frames(Capture(),[row(9,.9)]))


class MetadataTests(unittest.TestCase):
    def test_new_source_requires_matching_provenance(self):
        with self.assertRaises(ValueError):
            report.load_provenance(None,"new-video")
        original = report.load_provenance(None,report.ORIGINAL_VIDEO_SHA256)
        self.assertEqual(original["author"],"Experienciausuario")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"provenance.json"
            path.write_text(json.dumps({**report.ORIGINAL_PROVENANCE,"sha256":"wrong"}))
            with self.assertRaises(ValueError):
                report.load_provenance(path,report.ORIGINAL_VIDEO_SHA256)

    def test_new_metadata_is_used_and_missing_values_remain_unknown(self):
        current = report.run_metadata({"model_filename":"models/yolox_m.onnx","confidence_threshold":.6,
                                       "detector_forward_passes":12,"max_age_s":30},"new","new")
        self.assertEqual(current["model_label"],"yolox_m")
        self.assertEqual(current["confidence_threshold"],.6)
        self.assertEqual(current["detector_forward_passes"],12)
        unknown = report.run_metadata({},"new","new")
        self.assertIsNone(unknown["confidence_threshold"])
        self.assertIn("unrecorded",unknown["model_label"])
        self.assertEqual(report.run_metadata({},"new","new","Custom model")["model_label"],"Custom model")

    def test_legacy_threshold_requires_exact_original_summary_and_log(self):
        legacy = report.run_metadata({},report.ORIGINAL_SUMMARY_SHA256,report.ORIGINAL_LOG_SHA256)
        self.assertEqual(legacy["confidence_threshold"],.35)
        self.assertEqual(legacy["model_label"],"YOLOX-S")
        for summary_hash, log_hash in ((report.ORIGINAL_SUMMARY_SHA256,"changed"),
                                       ("changed",report.ORIGINAL_LOG_SHA256)):
            self.assertIsNone(report.run_metadata({},summary_hash,log_hash)["confidence_threshold"])


class RenderTests(unittest.TestCase):
    def test_sampled_video_output_uses_metadata_and_matching_decoded_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, run, out = root/"fixture.mp4", root/"run", root/"report"
            run.mkdir()
            writer = cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*"mp4v"),30.,(64,36))
            self.assertTrue(writer.isOpened())
            for index in range(10):
                writer.write(np.full((36,64,3),(index*20,70,120),dtype=np.uint8))
            writer.release()
            rows = [row(1,1/30,[detection()]),row(4,4/30),row(7,7/30,[detection()])]
            (run/"detections.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            (run/"summary.json").write_text(json.dumps({"frames":3,"sample_every":3,"nominal_source_fps":30.,
                "input_sha256":report.sha(video),"model_sha256":"fixture-model","model_filename":"strong.onnx",
                "confidence_threshold":.6,"elapsed_wall_s":.6}))
            provenance = root/"provenance.json"
            provenance.write_text(json.dumps({"sha256":report.sha(video),"title":"Generated fixture",
                "author":"Report tests","license":"Generated test data","source_url":"https://example.invalid/fixture"}))
            with contextlib.redirect_stdout(io.StringIO()), patch.object(report,"put_line",wraps=report.put_line) as text_writer:
                report.main(["--run",str(run),"--video",str(video),"--output",str(out),"--provenance",str(provenance)])
            rendered = json.loads((out/"summary.json").read_text())
            self.assertEqual(rendered["model_label"],"strong")
            self.assertEqual(rendered["confidence_threshold"],.6)
            self.assertAlmostEqual(rendered["preview_fps"],10.)
            self.assertEqual(rendered["tracks"][0]["missing_frames_within_span"],1)
            self.assertEqual(rendered["source_provenance"]["author"],"Report tests")
            rendered_text = "\n".join(call.args[1] for call in text_writer.call_args_list)
            self.assertIn("strong / OBJECT TRACKING | confidence threshold 0.6",rendered_text)
            self.assertIn("Report tests | Generated test data",rendered_text)
            self.assertIn("https://example.invalid/fixture",rendered_text)
            capture = cv2.VideoCapture(str(out/"tracking_preview.mp4"))
            self.assertAlmostEqual(capture.get(cv2.CAP_PROP_FPS),10.,places=2)
            blues = []
            while True:
                ok, frame = capture.read()
                if not ok: break
                blues.append(int(frame[430,640,0]))
            capture.release()
            self.assertEqual(len(blues),3)
            np.testing.assert_allclose(blues,[20,80,140],atol=12)
            for name,digest in rendered["artifact_hashes"].items():
                self.assertEqual(report.sha(out/name),digest)


if __name__ == "__main__":
    unittest.main()
