"""Seven exact-frame approximate checks of one preselected video target.

This scorer reads saved observations only. It never runs a model, interpolates
predictions, changes a track ID, or drops a missing/lost target from the score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

INITIAL_SOURCE_INDEX = 119
EXPECTED_SOURCE_INDICES = (148, 178, 208, 238, 268, 298, 328)
IOU_THRESHOLD = .5
REQUIRED_ACTIVE_HIT_FRACTION = .75


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _index(value, label):
    if type(value) is not int or value < 0:
        raise ValueError(label+' must be a nonnegative integer')
    return value


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not np.isfinite(value):
        raise ValueError(label+' must be finite')
    return float(value)


def _dimensions(value, label):
    if (not isinstance(value,list) or len(value)!=2
            or any(type(v) is not int or v<2 for v in value)):
        raise ValueError(label+' must be [height,width] with positive dimensions')
    return value


def _box(value, image_hw):
    if not isinstance(value,(list,tuple)) or len(value)!=4:
        raise ValueError('A box must contain four finite coordinates')
    box=np.array([_finite(v,'bbox coordinate') for v in value],np.float64)
    h,w=image_hw
    if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        raise ValueError('Bounding box must have positive area within image bounds')
    return box


def box_metrics(prediction, truth):
    overlap=np.maximum(0.,np.minimum(prediction[2:],truth[2:])-np.maximum(prediction[:2],truth[:2]))
    intersection=float(np.prod(overlap))
    union=float(np.prod(prediction[2:]-prediction[:2])+np.prod(truth[2:]-truth[:2])-intersection)
    center_error=float(np.linalg.norm((prediction[:2]+prediction[2:]-truth[:2]-truth[2:])/2))
    return intersection/union,center_error


def evaluate_records(definition, rows, annotations):
    """Pure score computation; incomplete predictions count as misses."""
    source_sha=definition.get('video_sha256')
    if (not isinstance(source_sha,str) or not re.fullmatch('[0-9a-f]{64}',source_sha)
            or source_sha!=annotations.get('source_sha256')):
        raise ValueError('Run and annotations must identify the same source SHA256')
    original_hw=_dimensions(definition.get('source_original_hw'),'source dimensions')
    if _dimensions(annotations.get('image_hw'),'annotation dimensions')!=original_hw:
        raise ValueError('Annotation and source image dimensions disagree')
    transform=definition.get('letterbox',{})
    image_hw=_dimensions(transform.get('image_hw'),'processed image dimensions')
    if _dimensions(transform.get('original_hw'),'transform original dimensions')!=original_hw:
        raise ValueError('Stored transform identifies different original dimensions')
    scale=np.asarray(transform.get('original_to_image_scale_xy'),dtype=float)
    expected_scale=np.array([image_hw[1]/original_hw[1],image_hw[0]/original_hw[0]])
    if scale.shape!=(2,) or not np.isfinite(scale).all() or not np.allclose(scale,expected_scale,rtol=0,atol=1e-12):
        raise ValueError('Stored original-to-processed scaling is inconsistent')
    if annotations.get('initial_sequence')!=INITIAL_SOURCE_INDEX:
        raise ValueError('Expected source119 for initialization, excluded from scoring')
    labels=annotations.get('frames')
    if not isinstance(labels,list) or not labels:
        raise ValueError('Nonempty exact-frame annotations are required')
    label_by_source={}
    for label in labels:
        index=_index(label.get('sequence'),'annotation sequence')
        if index in label_by_source:
            raise ValueError('Duplicate annotation source frame')
        label_by_source[index]=label
    scored={index:label for index,label in label_by_source.items()
            if index!=INITIAL_SOURCE_INDEX and label.get('scorable') is True}
    if set(scored)!=set(EXPECTED_SOURCE_INDICES):
        raise ValueError('All seven predeclared noninitial exact-source checks are required')

    sampling=definition.get('sampling',{})
    indices=sampling.get('source_indices')
    pts=sampling.get('source_pts_s'); grid=sampling.get('grid_times_s')
    captures=sampling.get('capture_times_s'); origin=_finite(sampling.get('origin_pts_s'),'source time origin')
    if (not isinstance(indices,list) or not indices or any(not isinstance(v,list) or len(v)!=len(indices)
            for v in (pts,grid,captures))):
        raise ValueError('Complete frozen sampling metadata is required')
    first_sequence={}
    for sequence,index in enumerate(indices):
        _index(index,'sampled source index')
        source_time=_finite(pts[sequence],'source PTS')
        grid_time=_finite(grid[sequence],'grid time')
        capture=_finite(captures[sequence],'capture time')
        if (capture < -1e-8 or capture > grid_time+1e-8
                or abs(source_time-origin-capture)>1e-8):
            raise ValueError('Sampling includes future input or inconsistent source clocks')
        first_sequence.setdefault(index,sequence)
    if indices[0]!=INITIAL_SOURCE_INDEX or not set(EXPECTED_SOURCE_INDICES).issubset(first_sequence):
        raise ValueError('Annotations do not match the actual sampled source frames')
    if not isinstance(rows,list):
        raise ValueError('Trace must be a list of recorded rows')
    by_sequence={}
    for row in rows:
        sequence=_index(row.get('sequence'),'trace sequence')
        if sequence>=len(indices) or sequence in by_sequence:
            raise ValueError('Duplicate or out-of-range trace sequence')
        if row.get('source_index')!=indices[sequence]:
            raise ValueError('Trace source index contradicts frozen exact-frame sampling')
        for key,expected in (('source_pts_s',pts[sequence]),('capture_time_s',captures[sequence]),
                             ('grid_time_s',grid[sequence])):
            if abs(_finite(row.get(key),key)-expected)>1e-8:
                raise ValueError('Trace PTS/grid contradict frozen sampling: '+key)
        response=_finite(row.get('neural_response_time_s'),'neural response time')
        if response < row['grid_time_s']-1e-8:
            raise ValueError('Neural response predates its stimulus grid time')
        by_sequence[sequence]=row
    initial_row=by_sequence.get(0,{})
    initial=initial_row.get('target_observation',{})
    initial_id=initial.get('track_id') if (initial.get('valid') is True
        and initial_row.get('perception_status')=='ok'
        and initial_row.get('inference_executed') is True) else None
    if type(initial_id) is not int or initial_id<1:
        initial_id=None

    checks=[]
    for index in EXPECTED_SOURCE_INDICES:
        label=scored[index];truth=_box(label.get('bbox_xyxy'),original_hw)
        uncertainty=_finite(label.get('boundary_uncertainty_px'),'annotation boundary uncertainty')
        if uncertainty<0: raise ValueError('Boundary uncertainty cannot be negative')
        sequence=first_sequence[index]
        if abs(_finite(label.get('timestamp_s'),'annotation PTS')-pts[sequence])>1e-6:
            raise ValueError('Annotation PTS does not identify the exact sampled source frame')
        row=by_sequence.get(sequence)
        prediction=None;geometric_iou=None;error=None;reason='missing_exact_source_observation'
        active=False;tid=None
        if row is not None:
            observation=row.get('target_observation',{})
            tid=observation.get('track_id')
            if initial_id is None:
                reason='initial_target_not_available'
            elif observation.get('valid') is not True:
                reason='target_lost_or_invalid'
            elif tid!=initial_id or type(tid) is not int:
                reason='different_track_id'
            elif row.get('perception_status')!='ok' or row.get('inference_executed') is not True:
                reason='perception_not_successful'
            else:
                try:
                    processed=_box(observation.get('bbox_xyxy'),image_hw)
                except ValueError:
                    reason='invalid_prediction_bbox'
                else:
                    prediction=processed/np.tile(scale,2)
                    geometric_iou,error=box_metrics(prediction,truth)
                    active=True;reason='active_same_initial_track'
        checks.append(dict(source_index=index,trace_sequence=sequence,source_pts_s=float(pts[sequence]),
            annotation_bbox_original_xyxy=truth.tolist(),prediction_bbox_original_xyxy=None if prediction is None else prediction.tolist(),
            boundary_uncertainty_px=uncertainty,active=active,track_id=tid,reason=reason,
            geometric_iou=geometric_iou,active_iou=geometric_iou if active else 0.,
            center_error_original_px=error,hit=bool(active and geometric_iou>=IOU_THRESHOLD),
            causal_grid_age_s=float(grid[sequence]-captures[sequence]),
            neural_response_time_s=None if row is None else row['neural_response_time_s']))
    count=len(checks);hits=sum(r['hit'] for r in checks);active=sum(r['active'] for r in checks)
    errors=[r['center_error_original_px'] for r in checks if r['active']]
    return dict(kind='approximate_agent_exact_frame_target_check',source_sha256=source_sha,
        initial_source_index=INITIAL_SOURCE_INDEX,initialization_excluded=True,initial_track_id=initial_id,
        expected_source_indices=list(EXPECTED_SOURCE_INDICES),check_count=count,active_checks=active,
        active_coverage_fraction=active/count,active_hits=hits,active_hit_fraction=hits/count,
        iou_threshold=IOU_THRESHOLD,required_active_hit_fraction=REQUIRED_ACTIVE_HIT_FRACTION,
        passed=hits/count>=REQUIRED_ACTIVE_HIT_FRACTION,
        mean_active_iou_over_all_checks=float(np.mean([r['active_iou'] for r in checks])),
        mean_center_error_when_active_px=float(np.mean(errors)) if errors else None,
        missing_lost_or_invalid_checks=count-active,checks=checks,
        annotation_method=annotations.get('annotation_method','Approximate agent visual estimates'),
        annotation_limitations=annotations.get('limitations',[]),
        interpretation='Seven approximate agent labels in one scene, not benchmark truth or general tracking accuracy',
        uncertainty_note='Per-boundary annotation uncertainty retained verbatim; no IoU tolerance or box expansion applied',
        fixed_denominator='All seven checks retained, including missing, lost, invalid and wrong-ID predictions',
        not_measured=['metric3D navigation','collision avoidance','physical flight','general identity accuracy'])


def evaluate(run, annotations_path):
    run=Path(run);annotations_path=Path(annotations_path)
    definition=json.loads((run/'definition.json').read_text())
    provenance=json.loads((run/'provenance.json').read_text())
    if provenance.get('sha256')!=definition.get('video_sha256'):
        raise ValueError('Saved run provenance and definition identify different video bytes')
    frozen=json.loads((run/'frozen_inputs.json').read_text())
    if (frozen.get('definition_sha256')!=sha(run/'definition.json')
            or frozen.get('input_sha256')!=definition['video_sha256']):
        raise ValueError('Run definition does not match its frozen input receipt')
    rows=[json.loads(line) for line in (run/'trace.jsonl').read_text().splitlines()]
    annotations=json.loads(annotations_path.read_text())
    report=evaluate_records(definition,rows,annotations)
    report.update(run=str(run.resolve()),annotation_sha256=sha(annotations_path),
                  run_definition_sha256=sha(run/'definition.json'),run_trace_sha256=sha(run/'trace.jsonl'),
                  scoring_source_sha256=sha(Path(__file__)))
    return report


def write_report(report, output, run):
    output=Path(output).resolve();run=Path(run).resolve()
    if output==run or run in output.parents:
        raise ValueError('Score output must be outside the immutable raw run directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as handle:
        json.dump(report,handle,indent=2,allow_nan=False);handle.write('\n')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=evaluate(args.run,args.annotations)
    write_report(report,args.output,args.run)
    print(json.dumps({key:report[key] for key in ('passed','active_hits','check_count','active_coverage_fraction')}))


if __name__=='__main__':
    main()
