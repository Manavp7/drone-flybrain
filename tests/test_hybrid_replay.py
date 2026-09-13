"""Causal saved-observation scoring; fixtures are not inference evidence."""
import tempfile
import unittest
from pathlib import Path

from experiments.hybrid_replay import evaluate_replay, require_hash, sha256, validate_rows


def detection(tid=296, box=(10.,10.,30.,50.)):
    return {"track_id":tid,"class_id":0,"label":"person","confidence":.9,"bbox_xyxy":list(box)}


def row(sequence,detections=None,**changes):
    r={"sequence":sequence,"capture_time_s":sequence*.1,"stream_id":"offline","clock_domain":"file",
       "frame_id":"optical","status":"ok","inference_executed":True,
       "detections":[detection()] if detections is None else detections}
    r.update(changes)
    return r


def annotations(sequences=(0,1,2),box=(10.,10.,30.,50.)):
    return {"initial_sequence":0,"frames":[{"sequence":s,"timestamp_s":s*.1,
            "bbox_xyxy":list(box),"scorable":True} for s in sequences]}


class ReplayTests(unittest.TestCase):
    def test_initialization_excluded_and_latest_prior_used_without_future(self):
        rows=[row(0),row(2,[detection(box=(100,100,150,200))])]
        checks,stats=evaluate_replay(rows,annotations(),[0.,.1,.2])
        self.assertEqual([r["observed_sequence"] for r in checks],[0,2])
        self.assertEqual(stats["scored_noninitial_checks"],2)
        self.assertEqual(stats["active_iou_ge_0_5_hits"],1)
        self.assertEqual(checks[0]["media_age_s"],.1)

    def test_missing_latest_row_never_falls_back_to_older_target(self):
        checks,stats=evaluate_replay([row(0),row(1,[]),row(2)],annotations(),[0.,.1,.2])
        self.assertFalse(checks[0]["valid"])
        self.assertEqual(checks[0]["reason"],"target_not_observed")
        self.assertEqual(stats["active_iou_ge_0_5_hits"],1)

    def test_failed_latest_inference_cannot_score_saved_box(self):
        checks,stats=evaluate_replay([row(0),row(1,status="rejected")],annotations((0,1)),[0.,.1])
        self.assertFalse(checks[0]["valid"])
        self.assertEqual(stats["active_iou_ge_0_5_hits"],0)

    def test_media_age_limit_prevents_indefinite_hold(self):
        checks,stats=evaluate_replay([row(0)],annotations((0,4)),[0.,.1,.2,.3,.4])
        self.assertEqual(checks[0]["reason"],"stale_target_frame")
        self.assertEqual(stats["valid_observation_checks"],0)
        self.assertIsNone(stats["mean_center_error_source_px_valid_only"])

    def test_other_id_cannot_replace_explicit_selection(self):
        checks,stats=evaluate_replay([row(0),row(1,[detection(297)])],annotations((0,1)),[0.,.1])
        self.assertFalse(checks[0]["valid"])
        self.assertEqual(stats["active_iou_ge_0_5_hits"],0)

    def test_intervening_stream_reset_is_seen_even_when_not_labelled(self):
        rows=[row(0),row(1,stream_id="other"),row(2)]
        checks,stats=evaluate_replay(rows,annotations((0,2)),[0.,.1,.2])
        self.assertEqual(checks[0]["reason"],"stream_changed")
        self.assertEqual(stats["active_iou_ge_0_5_hits"],0)

    def test_source_pts_is_used_and_original_logged_time_preserved(self):
        ann=annotations((0,1))
        ann["frames"][0]["timestamp_s"]=.003
        ann["frames"][1]["timestamp_s"]=.103
        checks,_=evaluate_replay([row(0)],ann,[.003,.103])
        self.assertEqual(checks[0]["observed_pts_s"],.003)
        self.assertEqual(checks[0]["original_logged_capture_time_s"],0.)
        self.assertAlmostEqual(checks[0]["media_age_s"],.1)

    def test_mismatched_annotation_pts_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_replay([row(0)],annotations((0,1)),[.003,.103])

    def test_duplicate_reordered_and_relabelled_rows_rejected(self):
        renamed=detection(); renamed["label"]="bird"
        for rows in ([row(0),row(0)],[row(1),row(0)],[row(0,[detection(),detection()])],[row(0,[renamed])]):
            with self.subTest(rows=rows),self.assertRaises(ValueError):validate_rows(rows,[0.,.1])

    def test_hash_mismatch_is_an_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"fixture"
            path.write_bytes(b"locked artifact")
            self.assertEqual(require_hash(path,sha256(path)),sha256(path))
            with self.assertRaises(ValueError):require_hash(path,"0"*64)


if __name__=="__main__":
    unittest.main()
