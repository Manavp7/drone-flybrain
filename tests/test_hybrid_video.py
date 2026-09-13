"""Offline sampling/selection contracts; tests perform no trained inference."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.hybrid_target import salience_mask
from experiments.hybrid_video import (
    FirstFrameSelector, causal_schedule, letterbox_metadata, load_frozen_readout, sha,
)


def detection(tid, box, confidence=.9):
    return dict(track_id=tid,bbox_xyxy=box,confidence=confidence,class_id=0,label='person')


def result(sequence, detections, timestamp=None):
    return dict(sequence=sequence,capture_time_s=sequence*.1 if timestamp is None else timestamp,
                stream_id='video',clock_domain='media',frame_id='rgb',status='ok',
                inference_executed=True,detections=detections)


class CausalSamplingTests(unittest.TestCase):
    def test_variable_pts_never_select_future_frame(self):
        pts=np.array([2.,2.031,2.099,2.103,2.198,2.25,2.311,2.401])
        schedule=causal_schedule(pts,0.,.4)
        np.testing.assert_array_equal(schedule['source_indices'],[0,2,4,5])
        self.assertTrue(np.all(schedule['source_pts_s'] <= schedule['grid_pts_s']+1e-10))
        np.testing.assert_allclose(schedule['capture_times_s'],[0.,.099,.198,.25])

    def test_start_normalizes_actual_selected_source_timestamp(self):
        schedule=causal_schedule(np.array([5.,5.09,5.19,5.29,5.39]),.15,.2)
        self.assertEqual(schedule['origin_pts_s'],5.09)
        np.testing.assert_allclose(schedule['grid_times_s'],[0.,.1])
        self.assertEqual(schedule['source_indices'][0],1)

    def test_gap_preserves_old_image_age_instead_of_refreshing_it(self):
        schedule=causal_schedule(np.array([0.,.033,.066,.45,.5]),0.,.5)
        np.testing.assert_array_equal(schedule['source_indices'],[0,2,2,2,2])
        self.assertGreater(schedule['grid_times_s'][-1]-schedule['capture_times_s'][-1],.3)

    def test_invalid_or_out_of_bounds_timestamps_fail(self):
        for pts in ([0.,0.],[-1.,0.],[0.,float('nan')],[.2,.1],[False,True]):
            with self.subTest(pts=pts), self.assertRaises(ValueError):
                causal_schedule(np.array(pts),0.,.1)
        for start,duration in ((0.,21.),(0.,.15),(1.,.1),(0.,.8)):
            with self.subTest(start=start,duration=duration), self.assertRaises(ValueError):
                causal_schedule(np.array([0.,.1,.2]),start,duration)


class FirstSelectionTests(unittest.TestCase):
    def test_original_pixel_point_maps_to_resized_image_once(self):
        selector=FirstFrameSelector([150.,100.])
        first=selector.update(result(0,[detection(1,[5.,5.,40.,90.]),
            detection(2,[60.,20.,95.,80.],.8)]),0.,[100,100],[200,200])
        self.assertTrue(first['valid']);self.assertEqual(first['track_id'],2)
        # Selected subject moves away from original point. Point is not reused.
        second=selector.update(result(1,[detection(1,[60.,20.,95.,80.]),
            detection(2,[10.,20.,45.,80.],.7)]),.1,[100,100],[200,200])
        self.assertTrue(second['valid']);self.assertEqual(second['track_id'],2)

    def test_first_point_miss_never_selects_a_later_person(self):
        selector=FirstFrameSelector([150.,100.])
        first=selector.update(result(0,[detection(1,[5.,5.,40.,90.])]),0.,[100,100],[200,200])
        self.assertFalse(first['valid'])
        later=selector.update(result(1,[detection(2,[60.,20.,95.,80.])]),.1,[100,100],[200,200])
        self.assertFalse(later['valid']);self.assertIsNone(later['track_id'])

    def test_no_point_selects_highest_confidence_and_keeps_id_during_loss(self):
        selector=FirstFrameSelector()
        first=selector.update(result(0,[detection(1,[5.,5.,40.,90.],.6),
            detection(2,[60.,20.,95.,80.],.9)]),0.,[100,100],[100,100])
        self.assertEqual(first['track_id'],2)
        lost=selector.update(result(1,[detection(1,[5.,5.,40.,90.],.99)]),.1,[100,100],[100,100])
        self.assertFalse(lost['valid']);self.assertEqual(lost['track_id'],2)

    def test_held_timestamp_is_not_fresh_target_evidence(self):
        selector=FirstFrameSelector()
        selector.update(result(0,[detection(1,[5.,5.,40.,90.])]),0.,[100,100],[100,100])
        held=selector.update(result(1,[detection(1,[5.,5.,40.,90.])],timestamp=0.),.1,[100,100],[100,100])
        self.assertFalse(held['valid'])
        self.assertEqual(held['reason'],'duplicate_or_reordered_frame')


class LetterboxAndReadoutTests(unittest.TestCase):
    def test_letterbox_metadata_matches_actual_cue_paint(self):
        metadata=letterbox_metadata([540,960],[720,1280])
        np.testing.assert_allclose(metadata['original_to_image_scale_xy'],[.75,.75])
        self.assertAlmostEqual(metadata['image_to_cue_scale'],391/960)
        np.testing.assert_allclose(metadata['image_to_cue_offset_xy'],[0.,85.53125])
        box=[100.,100.,200.,300.]
        cue=salience_mask(dict(valid=True,bbox_xyxy=box),[540,960])
        rows,cols=np.nonzero(cue<.5)
        scale=metadata['image_to_cue_scale'];x,y=metadata['image_to_cue_offset_xy']
        self.assertEqual(cols.min(),int(np.floor(x+box[0]*scale)))
        self.assertEqual(rows.min(),int(np.floor(y+box[1]*scale)))
        self.assertEqual(rows.max()+1,int(np.ceil(y+box[3]*scale)))
        self.assertFalse(metadata['metric_distance_available'])

    def test_frozen_selection_and_readout_are_both_hashed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)
            model=dict(mean=[0.]*8,scale=[1.]*8,coefficients=np.zeros((8,3)).tolist(),intercept=[0.,0.,.38])
            (folder/'readout.json').write_text(json.dumps(model))
            model_sha=sha(folder/'readout.json')
            (folder/'selection_frozen.json').write_text(json.dumps(dict(readout_sha256=model_sha,validation={'passed':True})))
            receipt_sha=sha(folder/'selection_frozen.json')
            loaded,_=load_frozen_readout(folder,expected_selection_sha=receipt_sha,expected_readout_sha=model_sha)
            np.testing.assert_array_equal(loaded.predict(np.zeros(8)),[0.,0.,.38])
            # Retaining the previous receipt does not permit a changed model.
            (folder/'readout.json').write_text(json.dumps(model)+' ')
            with self.assertRaises(ValueError):
                load_frozen_readout(folder,expected_selection_sha=receipt_sha,expected_readout_sha=model_sha)
            (folder/'selection_frozen.json').write_text('{}')
            with self.assertRaises(ValueError):
                load_frozen_readout(folder,expected_selection_sha=receipt_sha,expected_readout_sha=model_sha)


if __name__ == '__main__':
    unittest.main()
