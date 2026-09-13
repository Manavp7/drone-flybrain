"""Exact alignment and loss-aware scoring fixtures; no model execution."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.evaluate_hybrid_video import (
    EXPECTED_SOURCE_INDICES, INITIAL_SOURCE_INDEX, evaluate_records, write_report,
)


def fixture():
    indices=[INITIAL_SOURCE_INDEX,*EXPECTED_SOURCE_INDICES]
    pts=[i/30 for i in indices];origin=pts[0]
    captures=[t-origin for t in pts]
    grid=[float(i) for i in range(8)]
    definition=dict(video_sha256='a'*64,source_original_hw=[720,1280],
        letterbox=dict(image_hw=[540,960],original_hw=[720,1280],original_to_image_scale_xy=[.75,.75]),
        sampling=dict(source_indices=indices,source_pts_s=pts,origin_pts_s=origin,
                      capture_times_s=captures,grid_times_s=grid))
    labels=[];rows=[]
    for i,index in enumerate(indices):
        truth=[300.,100.,500.,600.]
        labels.append(dict(sequence=index,timestamp_s=pts[i],scorable=True,
            bbox_xyxy=truth,boundary_uncertainty_px=10 if i==0 else 8))
        rows.append(dict(sequence=i,source_index=index,source_pts_s=pts[i],
            capture_time_s=captures[i],grid_time_s=grid[i],neural_response_time_s=grid[i]+.1,
            inference_executed=True,perception_status='ok',
            target_observation=dict(valid=True,track_id=5,bbox_xyxy=(np.array(truth)*.75).tolist())))
    annotations=dict(source_sha256='a'*64,image_hw=[720,1280],initial_sequence=INITIAL_SOURCE_INDEX,
        frames=labels,annotation_method='Approximate agent fixture',limitations=['Not benchmark truth'])
    return definition,rows,annotations


class ExactTargetScoringTests(unittest.TestCase):
    def setUp(self):
        self.definition,self.rows,self.annotations=fixture()

    def score(self):
        return evaluate_records(self.definition,self.rows,self.annotations)

    def test_processed_scale_reconstructs_original_boxes(self):
        report=self.score()
        self.assertTrue(report['passed']);self.assertEqual(report['active_hits'],7)
        self.assertEqual(report['check_count'],7)
        np.testing.assert_allclose(report['checks'][0]['prediction_bbox_original_xyxy'],[300,100,500,600])
        self.assertEqual(report['checks'][0]['geometric_iou'],1.)
        self.assertEqual(report['checks'][0]['center_error_original_px'],0.)
        self.assertEqual(report['checks'][0]['boundary_uncertainty_px'],8.)
        self.assertNotIn(INITIAL_SOURCE_INDEX,[r['source_index'] for r in report['checks']])

    def test_lost_correct_retained_box_cannot_earn_hit(self):
        self.rows[1]['target_observation']['valid']=False
        report=self.score()
        self.assertEqual(report['active_hits'],6);self.assertEqual(report['check_count'],7)
        self.assertEqual(report['active_coverage_fraction'],6/7)
        self.assertEqual(report['checks'][0]['active_iou'],0.)
        self.assertIsNone(report['checks'][0]['center_error_original_px'])

    def test_missing_exact_frame_is_miss_without_nearest_substitution(self):
        del self.rows[1]
        report=self.score()
        self.assertEqual(report['active_hits'],6)
        self.assertEqual(report['checks'][0]['reason'],'missing_exact_source_observation')
        self.assertEqual(report['check_count'],7)

    def test_wrong_track_and_invalid_bbox_remain_denominator_misses(self):
        self.rows[1]['target_observation']['track_id']=6
        self.rows[2]['target_observation']['bbox_xyxy']=[10.,10.,10.,20.]
        report=self.score()
        self.assertFalse(report['passed']);self.assertEqual(report['active_hits'],5)
        self.assertEqual(report['check_count'],7)
        self.assertEqual(report['checks'][0]['reason'],'different_track_id')
        self.assertEqual(report['checks'][1]['reason'],'invalid_prediction_bbox')

    def test_perception_failure_is_inactive_even_with_good_retained_box(self):
        self.rows[1]['perception_status']='detector_error'
        report=self.score()
        self.assertFalse(report['checks'][0]['active'])
        self.assertEqual(report['checks'][0]['reason'],'perception_not_successful')

    def test_missing_initial_selection_does_not_choose_a_later_id(self):
        self.rows[0]['target_observation']['valid']=False
        report=self.score()
        self.assertEqual(report['active_hits'],0);self.assertEqual(report['check_count'],7)
        self.assertFalse(report['passed'])
        self.assertTrue(all(c['reason']=='initial_target_not_available' for c in report['checks']))

    def test_failed_initial_inference_cannot_establish_track_identity(self):
        self.rows[0]['inference_executed']=False
        report=self.score()
        self.assertIsNone(report['initial_track_id'])
        self.assertEqual(report['active_hits'],0)

    def test_empty_or_reduced_labels_are_rejected(self):
        for frames in ([],self.annotations['frames'][:-1]):
            altered=deepcopy(self.annotations);altered['frames']=frames
            with self.assertRaises(ValueError):
                evaluate_records(self.definition,self.rows,altered)

    def test_original_adjacent_annotations_are_not_interpolated(self):
        self.annotations['frames'][1]['sequence']=149
        with self.assertRaises(ValueError): self.score()

    def test_same_source_index_with_wrong_pts_is_rejected(self):
        self.rows[1]['source_pts_s']+=1/30
        with self.assertRaises(ValueError): self.score()

    def test_source_dimension_and_transform_mismatches_are_rejected(self):
        for definition,annotations in ((self.definition,{**self.annotations,'source_sha256':'b'*64}),
            (self.definition,{**self.annotations,'image_hw':[720,1920]})):
            with self.assertRaises(ValueError):
                evaluate_records(definition,self.rows,annotations)
        self.definition['letterbox']['original_to_image_scale_xy']=[1.,1.]
        with self.assertRaises(ValueError): self.score()

    def test_future_capture_is_rejected_and_grid_age_is_reported(self):
        report=self.score()
        self.assertAlmostEqual(report['checks'][0]['causal_grid_age_s'],1-(148-119)/30)
        self.definition['sampling']['capture_times_s'][1]=1.1
        self.definition['sampling']['source_pts_s'][1]=self.definition['sampling']['origin_pts_s']+1.1
        with self.assertRaises(ValueError): self.score()

    def test_duplicate_trace_sequence_is_rejected(self):
        self.rows.append(deepcopy(self.rows[1]))
        with self.assertRaises(ValueError): self.score()

    def test_all_missing_predictions_are_seven_misses(self):
        self.rows=self.rows[:1]
        report=self.score()
        self.assertEqual(report['check_count'],7);self.assertEqual(report['active_hits'],0)
        self.assertIsNone(report['mean_center_error_when_active_px'])

    def test_output_cannot_mutate_raw_run_or_overwrite_prior_score(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run=root/'raw';run.mkdir()
            with self.assertRaises(ValueError): write_report({},run/'score.json',run)
            output=root/'review'/'score.json'
            write_report({'checked':True},output,run)
            self.assertTrue(output.exists())
            with self.assertRaises(FileExistsError): write_report({},output,run)


if __name__=='__main__':
    unittest.main()
