"""Count matches to an independently marked person subset, not whole-scene accuracy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def iou_matrix(annotations, detections):
    a = np.asarray(annotations, dtype=float).reshape(-1, 4)
    b = np.asarray(detections, dtype=float).reshape(-1, 4)
    for boxes in (a, b):
        if not np.isfinite(boxes).all() or np.any(boxes[:, 2:] <= boxes[:, :2]):
            raise ValueError('Boxes must have finite positive area')
    intersection = np.prod(np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])
                                     - np.maximum(a[:, None, :2], b[None, :, :2])), axis=2)
    union = np.prod(a[:, 2:]-a[:, :2], axis=1)[:, None] + np.prod(b[:, 2:]-b[:, :2], axis=1)[None, :] - intersection
    return intersection / union


def match_people(annotations, detections, threshold=.5):
    """Maximum-cardinality one-to-one IoU matching via augmenting paths.

    Prefer higher IoU among eligible edges; a prediction cannot count twice.
    No label fitting, confidence tuning, or temporal matching is involved.
    """
    if not np.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError('Invalid IoU threshold')
    ious = iou_matrix(annotations, detections)
    edges = [[int(p) for p in np.argsort(-row, kind='stable') if row[p] >= threshold] for row in ious]
    assignments = {}

    def augment(annotation, visited):
        for prediction in edges[annotation]:
            if prediction in visited:
                continue
            visited.add(prediction)
            if prediction not in assignments or augment(assignments[prediction], visited):
                assignments[prediction] = annotation
                return True
        return False

    for annotation in range(len(annotations)):
        augment(annotation, set())
    return [{'annotation_index':a, 'prediction_index':p, 'iou':float(ious[a, p])}
            for p, a in sorted(assignments.items(), key=lambda pair:pair[1])]


def evaluate(annotations, run, minimum_confidence=None):
    source = json.loads((run/'summary.json').read_text())
    if minimum_confidence is not None and (not np.isfinite(minimum_confidence)
        or not source['confidence_threshold'] <= minimum_confidence <= 1):
        raise ValueError('A replay may raise the recorded confidence threshold, never lower it')
    threshold = source['confidence_threshold'] if minimum_confidence is None else minimum_confidence
    if source['input_sha256'] != annotations['source_sha256']:
        raise ValueError('Run and annotations refer to different source bytes')
    rows = [json.loads(line) for line in (run/'detections.jsonl').read_text().splitlines()]
    by_sequence = {row['sequence']:row for row in rows}
    if len(by_sequence) != len(rows):
        raise ValueError('Duplicate source sequences')
    result = []
    for frame in annotations['frames']:
        row = by_sequence[frame['sequence']]
        if row['status'] != 'ok' or not row['inference_executed']:
            raise ValueError('Manual check requires successful real inference on every selected frame')
        boxes = [person['bbox_xyxy'] for person in frame['people']]
        predictions = [det for det in row['detections'] if det['class_id'] == 0 and det['label'] == 'person'
                       and det['confidence'] >= threshold]
        pairs = match_people(boxes, [det['bbox_xyxy'] for det in predictions])
        matched = {pair['annotation_index'] for pair in pairs}
        result.append({'sequence':frame['sequence'], 'annotated_people':len(boxes),
                       'matched_people':len(pairs), 'person_predictions':len(predictions),
                       'unmatched_predictions_not_scored_as_false_positives':len(predictions)-len(pairs),
                       'matched_pairs':pairs,
                       'missed_annotation_ids':[person['annotation_id'] for i,person in enumerate(frame['people']) if i not in matched]})
    total = sum(row['annotated_people'] for row in result)
    matched = sum(row['matched_people'] for row in result)
    return {'run':str(run), 'frames':result, 'annotated_people':total, 'matched_people':matched,
            'annotated_subset_recall':matched/total if total else None,
            'confidence_threshold':threshold, 'recorded_confidence_threshold':source['confidence_threshold'],
            'confidence_filter_replay':minimum_confidence is not None, 'tile_size':source['tile_size'],
            'run_summary_sha256':sha(run/'summary.json'), 'run_log_sha256':sha(run/'detections.jsonl')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-confidence', type=float,
                        help='Optionally raise the saved score threshold for a matched-threshold detection comparison. Does not re-run tracking.')
    args = parser.parse_args(argv)
    annotations = json.loads(args.annotations.read_text())
    report = {'kind':'manual_annotated_subset_check', 'iou_threshold':.5,
              'annotation_sha256':sha(args.annotations), 'source_sha256':annotations['source_sha256'],
              'results':[evaluate(annotations, run, args.minimum_confidence) for run in args.runs],
              'limitations':annotations['limitations'],
              'not_measured':['whole-scene recall', 'precision', 'identity accuracy', 'aerial accuracy']}
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({item['run']:{'matched':item['matched_people'],'annotated':item['annotated_people']}
                      for item in report['results']}, indent=2))


if __name__ == '__main__':
    main()
