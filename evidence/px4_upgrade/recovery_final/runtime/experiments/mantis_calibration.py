"""A bounded generic-rectangle calibration of the frozen neural feature readout.

Known synthetic box geometry supplies training/evaluation labels only. Inference
still receives only the existing eight neural features. No actor data, recorded
flight truth, YOLO geometry bypass, or benchmark fitting is used. This repairs a
readout's cue-shape domain; it cannot establish advantage over geometric bearing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import time

import numpy as np

from experiments.hybrid_control import NeuralReadout
from experiments.hybrid_flyvis import HybridFlyvis, FEATURE_NAMES
from experiments.hybrid_target import salience_mask

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT/'models/flyvis_0000_000.manifest.json'
ORIGINAL_READOUT = ROOT/'results/hybrid_flight_run01/readout.json'
TRAIN_PHASES = ('train_grid', 'train_motion')
HELDOUT_PHASES = ('heldout_grid', 'heldout_motion')
RIDGE = 1e-4
HOLD_S = .1
LIMITS = dict(minimum_horizontal_rmse_reduction_fraction=.10,
              maximum_other_axis_rmse_multiplier=1.25,
              maximum_other_axis_rmse_addition=.002,
              maximum_per_phase_axis_rmse_multiplier=1.25,
              maximum_per_phase_axis_rmse_addition=.003,
              minimum_new_coverage_relative_to_original=1.,
              minimum_neural_evidence_fraction=1.)
SOURCES = ('experiments/mantis_calibration.py', 'experiments/hybrid_control.py',
           'experiments/hybrid_flyvis.py', 'experiments/hybrid_target.py',
           'experiments/runtime.py', 'flybrain_sim/research_model.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def array_sha(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(str(value.shape).encode()+str(value.dtype).encode()+value.tobytes()).hexdigest()


def _box_row(x, y, height, aspect, block, hold_index):
    cx, cy = (x+1)*391/2, (y+1)*391/2
    h = height*391
    w = h*aspect
    box = [cx-w/2, cy-h/2, cx+w/2, cy+h/2]
    if not 0 <= box[0] < box[2] <= 391 or not 0 <= box[1] < box[3] <= 391:
        raise ValueError('Synthetic cue escaped the image')
    return dict(label=[float(x), float(y), float(height)], aspect=float(aspect),
                bbox_xyxy=box, block=int(block), hold_index=int(hold_index))


def definitions():
    """Return complete deterministic input lists, before any model execution."""
    phases = {}
    grids = [
        ('train_grid', 118701, [-.4, -.2, 0., .2, .4], [-.3, 0., .3],
         [.22, .37, .52], [.20, .35, .50, .65, .80], 3),
        ('heldout_grid', 118702, [-.3, -.1, .1, .3], [-.225, -.075, .075, .225],
         [.26, .34, .46], [.25, .40, .55, .70, .75], 2),
    ]
    for name, seed, xs, ys, hs, aspects, holds in grids:
        boxes = list(itertools.product(xs, ys, hs, aspects))
        np.random.default_rng(seed).shuffle(boxes)
        rows = [_box_row(*box, block, hold) for block, box in enumerate(boxes) for hold in range(holds)]
        phases[name] = dict(split='training' if name.startswith('train_') else 'heldout',
                            kind='shuffled_rectangles', seed=seed, holds_per_box=holds, rows=rows)
    for name, heldout in (('train_motion', False), ('heldout_motion', True)):
        rows = []
        for i in range(64):
            if heldout:
                x, y = .35*np.sin(.17*i+1.1), .20*np.sin(.13*i+.7)
                height, aspect = .365+.115*np.sin(.10*i+.4), .49+.24*np.sin(.12*i+1.5)
            else:
                x, y = .32*np.sin(.14*i+.2), .18*np.sin(.09*i+.5)
                height, aspect = .36+.10*np.sin(.07*i+.3), .48+.22*np.sin(.11*i+.8)
            rows.append(_box_row(x, y, height, aspect, i, 0))
        phases[name] = dict(split='heldout' if heldout else 'training',
                            kind='smooth_generated_motion', seed=None, rows=rows)
    ordered = {name: phases[name] for name in (*TRAIN_PHASES, *HELDOUT_PHASES)}
    return dict(version='mantis-generic-cue-calibration-v1', attempt=1, phases=ordered,
                observation_count=sum(len(phase['rows']) for phase in ordered.values()),
                prediction_inputs=list(FEATURE_NAMES), labels=['x', 'y', 'height'],
                original_calibration_aspect=dict(training=.40, validation=.47),
                neural_hold_s=HOLD_S, neural_dt_s=.02, image_hw=[391, 391], ridge=RIDGE,
                acceptance_limits=LIMITS,
                clock='One generated mask held for0.1modelseconds; no physical-flight timing claim',
                purpose='Reduce neural reconstruction bias across generic rectangle shapes',
                superiority_vs_direct_or_filtered_geometry_evaluated=False)


def validate_definition(definition):
    phases = definition.get('phases', {})
    if set(phases) != set(TRAIN_PHASES+HELDOUT_PHASES):
        raise ValueError('Expected explicit training and held-out phases')
    training = set()
    heldout = set()
    count = 0
    for name, phase in phases.items():
        expected = 'training' if name in TRAIN_PHASES else 'heldout'
        if phase.get('split') != expected or not phase.get('rows'):
            raise ValueError('Incorrect data split')
        for row in phase['rows']:
            label, box = np.asarray(row['label']), np.asarray(row['bbox_xyxy'])
            if label.shape != (3,) or box.shape != (4,) or not np.isfinite(np.r_[label, box]).all():
                raise ValueError('Malformed generated cue')
            target = training if expected == 'training' else heldout
            target.add(tuple(box))
            count += 1
    if training & heldout:
        raise ValueError('Training and held-out cues overlap')
    if count != definition.get('observation_count') or not 1000 <= count <= 1500:
        raise ValueError('Calibration must remain inside the predeclared observation budget')


def decode_features(readout, features):
    """The complete prediction interface: readout plus eight neural features."""
    array = np.asarray(features, dtype=float)
    if array.ndim not in (1, 2) or array.shape[-1] != 8 or not np.isfinite(array).all():
        raise ValueError('Prediction accepts only finite eight-feature neural records')
    return readout.predict(array)


def fit_training(training):
    """Reject held-out keys rather than relying on callers to avoid leakage."""
    if set(training) != set(TRAIN_PHASES):
        raise ValueError('Fit accepts exactly the two training phases, never held-out records')
    x = np.concatenate([training[name]['features'] for name in TRAIN_PHASES])
    labels = np.concatenate([training[name]['labels'] for name in TRAIN_PHASES])
    return NeuralReadout.fit(x, labels, ridge=RIDGE)


def collect_phase(brain, phase, folder):
    folder.mkdir()
    reset = brain.reset()
    started = time.perf_counter()
    features, labels, boxes, activities, retinas, validity, model_times, walls = ([] for _ in range(8))
    for index, row in enumerate(phase['rows']):
        mask = salience_mask(dict(valid=True, bbox_xyxy=row['bbox_xyxy']), (391, 391))
        response = brain.step(mask, HOLD_S)
        features.append(response['features'])
        labels.append(row['label'])
        boxes.append(row['bbox_xyxy'])
        activities.append(response['activity'])
        retinas.append(response['retina'])
        validity.append(response['valid'])
        model_times.append([response['stimulus_time_s'], response['response_time_s']])
        walls.append(response['elapsed_wall_s'])
        if abs(response['response_time_s']-(index+1)*HOLD_S) > 1e-7:
            raise RuntimeError('Unexpected neural integration clock')
        if (index+1)%100 == 0:
            print(json.dumps(dict(phase=folder.name, completed_observations=index+1,
                                  planned_observations=len(phase['rows']))), flush=True)
    x, y, valid = np.asarray(features), np.asarray(labels), np.asarray(validity, bool)
    np.savez_compressed(folder/'neural.npz', features=x, labels=y, boxes=boxes,
        activity=np.asarray(activities), retina=np.asarray(retinas), valid=valid,
        model_times=model_times, inference_wall_s=walls, baseline=brain.baseline_activity,
        cell_indices=brain.cell_indices, centers_rc=brain.centers_rc)
    dump(folder/'receipt.json', dict(observations=len(x), neural_steps=len(x)*5,
        valid_neural_observations=int(valid.sum()), whole_wall_s=time.perf_counter()-started,
        inference_wall_s=float(sum(walls)), reset=reset,
        features_sha256=array_sha(x), labels_sha256=array_sha(y), activity_file_sha256=sha(folder/'neural.npz')))
    return dict(features=x, labels=y, valid=valid, phase=phase)


def coverage(features, evidence_valid, decoded):
    features = np.asarray(features, dtype=float)
    decoded = np.asarray(decoded, dtype=float)
    return (np.asarray(evidence_valid, bool) & np.isfinite(decoded).all(1)
            & (features[:, 7] >= .05) & (features[:, 5] >= np.log1p(.05))
            & (abs(decoded[:, 0]) <= .5) & (abs(decoded[:, 1]) <= .4)
            & (decoded[:, 2] >= .15) & (decoded[:, 2] <= .60))


def errors(prediction, labels):
    residual = np.asarray(prediction)-np.asarray(labels)
    return dict(count=len(residual), rmse=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                mean_absolute=np.mean(abs(residual), axis=0).tolist(),
                maximum_absolute=np.max(abs(residual), axis=0).tolist(),
                bias=np.mean(residual, axis=0).tolist())


def score_phase(records, original, calibrated):
    x, y, valid = records['features'], records['labels'], records['valid']
    result = dict(observations=len(x), neural_evidence_count=int(valid.sum()), methods={})
    for name, readout in (('original', original), ('calibrated', calibrated)):
        predicted = decode_features(readout, x)
        available = coverage(x, valid, predicted)
        # Errors use all generated observations, including ones the flight gate
        # would reject. Lower coverage cannot improve this error denominator.
        result['methods'][name] = dict(**errors(predicted, y),
            available_count=int(available.sum()), coverage=float(available.mean()))
    original_rmse = np.asarray(result['methods']['original']['rmse'])
    calibrated_rmse = np.asarray(result['methods']['calibrated']['rmse'])
    result['rmse_change_fraction'] = (1-calibrated_rmse/np.maximum(original_rmse, 1e-12)).tolist()
    return result


def evaluate_heldout(heldout, original, calibrated):
    if set(heldout) != set(HELDOUT_PHASES):
        raise ValueError('Evaluation requires both complete held-out phases')
    results = {name: score_phase(heldout[name], original, calibrated) for name in HELDOUT_PHASES}
    combined = {key: np.concatenate([heldout[name][key] for name in HELDOUT_PHASES])
                for key in ('features', 'labels', 'valid')}
    results['combined'] = score_phase(combined, original, calibrated)
    old = np.array(results['combined']['methods']['original']['rmse'])
    new = np.array(results['combined']['methods']['calibrated']['rmse'])
    gates = dict(horizontal_rmse_improved=bool(new[0] <= old[0]*(1-LIMITS['minimum_horizontal_rmse_reduction_fraction'])),
        other_axes_not_materially_worse=bool(np.all(new[1:] <= old[1:]*LIMITS['maximum_other_axis_rmse_multiplier']
                                                  +LIMITS['maximum_other_axis_rmse_addition'])),
        coverage_preserved=all(result['methods']['calibrated']['available_count'] >= result['methods']['original']['available_count']
                               for result in results.values()),
        complete_neural_evidence=all(result['neural_evidence_count'] == result['observations'] for result in results.values()),
        no_phase_grossly_regresses=all(np.all(np.asarray(result['methods']['calibrated']['rmse'])
            <= np.asarray(result['methods']['original']['rmse'])*LIMITS['maximum_per_phase_axis_rmse_multiplier']
            +LIMITS['maximum_per_phase_axis_rmse_addition']) for name, result in results.items() if name != 'combined'))
    return dict(passed=all(gates.values()), gates=gates, results=results, thresholds=LIMITS,
                units='Normalized center coordinates x/y; height as a fraction of391pixels',
                error_denominator='Every predeclared held-out observation, no selective abstention',
                direct_geometry_is_exact_box_reconstruction_oracle=True,
                neural_superiority_over_geometry_or_filter_evaluated=False,
                physical_flight_evaluated=False)


def freeze_definition(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    definition = definitions()
    validate_definition(definition)
    definition.update(frozen_utc=datetime.now(timezone.utc).isoformat(),
        source_hashes={path: sha(ROOT/path) for path in SOURCES},
        manifest_sha256=sha(MANIFEST), original_readout_sha256=sha(ORIGINAL_READOUT))
    (output/'sources').mkdir()
    for path in SOURCES:
        shutil.copy2(ROOT/path, output/'sources'/path.replace('/', '__'))
    shutil.copy2(MANIFEST, output/'model_manifest.json')
    shutil.copy2(ORIGINAL_READOUT, output/'original_readout.json')
    dump(output/'definition.json', definition)
    return definition


def _unchanged(definition):
    return (all(sha(ROOT/path) == digest for path, digest in definition['source_hashes'].items())
            and sha(MANIFEST) == definition['manifest_sha256']
            and sha(ORIGINAL_READOUT) == definition['original_readout_sha256'])


def run(output):
    output = Path(output).resolve()
    definition = freeze_definition(output)
    failure = None
    try:
        # The complete ordered cues, seeds, model digest and acceptance criteria
        # are now on disk. Model inference starts only after that freeze.
        brain = HybridFlyvis(MANIFEST)
        dump(output/'runtime.json', dict(packages=brain.adapter.runtime,
            device=str(brain.adapter.device), model_manifest_sha256=sha(MANIFEST),
            feature_names=list(FEATURE_NAMES), backend=type(brain).__module__+'.'+type(brain).__name__))
        original = NeuralReadout(**json.loads(ORIGINAL_READOUT.read_text()))
        training = {name: collect_phase(brain, definition['phases'][name], output/name) for name in TRAIN_PHASES}
        calibrated = fit_training(training)
        dump(output/'candidate_readout.json', calibrated.to_dict())
        dump(output/'fit_frozen.json', dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
            trained_phase_names=list(TRAIN_PHASES), ridge=RIDGE,
            training_observations=sum(len(training[name]['features']) for name in TRAIN_PHASES),
            training_artifact_hashes={name: sha(output/name/'neural.npz') for name in TRAIN_PHASES},
            candidate_readout_sha256=sha(output/'candidate_readout.json'),
            heldout_inference_started=False, prediction_inputs=list(FEATURE_NAMES)))
        heldout = {name: collect_phase(brain, definition['phases'][name], output/name) for name in HELDOUT_PHASES}
        report = evaluate_heldout(heldout, original, calibrated)
        report.update(completed_utc=datetime.now(timezone.utc).isoformat(), attempt=1,
                      original_readout_sha256=sha(ORIGINAL_READOUT),
                      calibrated_readout_sha256=sha(output/'candidate_readout.json'),
                      observations=definition['observation_count'], neural_steps=definition['observation_count']*5)
        if not _unchanged(definition):
            raise RuntimeError('Frozen calibration sources/model/readout changed during inference')
        dump(output/'validation.json', report)
        print(json.dumps(dict(passed=report['passed'], gates=report['gates'],
                              combined=report['results']['combined'])), flush=True)
        return report
    except Exception as exc:
        failure = dict(type=type(exc).__name__, reason=str(exc))
        dump(output/'failure.json', failure)
        raise
    finally:
        dump(output/'artifact_hashes.json', {str(path.relative_to(output)): sha(path)
             for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'artifact_hashes.json'})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
