"""Matched, offline bearing estimators; target truth is accepted only by scoring.

Mantis Neural uses the supplied frozen Flyvis visual network and learned
readout. Direct YOLO and alpha-beta see the same transformed detector boxes,
calibration, timestamp and missing-input rules. This compares capture-time
estimation, not real-time throughput or closed-loop flight performance.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import math
from numbers import Integral, Real
import time

import numpy as np

from experiments.flight_contracts import wrap_angle
from experiments.flight_guidance import TRAIN_X, TRAIN_Y, TRAIN_HEIGHT, cue_geometry
from experiments.hybrid_target import salience_mask

METHODS = ('mantis_neural', 'direct_yolo', 'alpha_beta')
CONFIG = {
    'version': 'mantis-bearing-comparison-v2.1',
    'methods': list(METHODS),
    'display_names': {'mantis_neural': 'Mantis Neural', 'direct_yolo': 'Direct YOLO',
                      'alpha_beta': 'Alpha-beta'},
    'neural_component': 'Frozen Flyvis visual network plus validated shape-diverse neural readout',
    'recording_interval_s': None,
    'recording_clock': 'recorded_capture_timestamps',
    'neural_hold_s': .1,
    'neural_clock': 'One .1-second model hold per recorded observation; distinct from physical capture intervals',
    'minimum_response_interval_s': .1,
    'variants': ['clean', 'noise', 'gaps'],
    'noise_center_standard_deviation_px': 4.,
    'gap_sequence_ranges_inclusive': [[12, 15], [32, 35]],
    'alpha': .65,
    'beta': .10,
    'filter_reset_gap_s': .5,
    'development_seeds': [17011, 17012],
    'heldout_seeds': [27101, 27102, 27103, 27104, 27105, 27106, 38101, 38102, 38103],
    'eligibility_envelope': {'abs_x': TRAIN_X, 'abs_y': TRAIN_Y,
                             'height': list(TRAIN_HEIGHT)},
    'missing_rule': 'All methods suppress output on missing or unsupported current input',
    'noise_clipping': 'No clipping; an off-image perturbed box is shared unavailable input',
    'score_clock': 'capture_time_s',
    'primary_comparison': 'wrapped bearing RMSE on identical all-method matched observations',
    'automatic_winner_or_superiority_test': False,
    'timing_mode': 'offline per-observation comparison; no concurrency or backlog claim',
}


def _number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and math.isfinite(value)


@dataclass(frozen=True)
class BearingInput:
    """Sensor-only interface. No actor positions, labels or evaluator truth."""
    sequence: int
    capture_time_s: float
    image_hw: tuple
    bbox_xyxy: tuple | None
    intrinsics: tuple
    rotation_world_camera: tuple
    detector_wall_s: float = 0.

    def __post_init__(self):
        if (not isinstance(self.sequence, Integral) or isinstance(self.sequence, (bool, np.bool_))
                or self.sequence < 0 or not _number(self.capture_time_s) or self.capture_time_s < 0
                or not _number(self.detector_wall_s) or self.detector_wall_s < 0):
            raise ValueError('Invalid sensor sequence, capture time or detector duration')
        hw = tuple(self.image_hw)
        if (len(hw) != 2 or any(not isinstance(x, Integral) or isinstance(x, (bool, np.bool_))
                               or x < 2 for x in hw)):
            raise ValueError('image_hw requires two integer dimensions')
        k = tuple(self.intrinsics)
        if (len(k) != 4 or not all(_number(x) for x in k) or min(k[:2]) <= 0
                or not 0 <= k[2] < hw[1] or not 0 <= k[3] < hw[0]):
            raise ValueError('Invalid optical camera intrinsics')
        rotation = np.asarray(self.rotation_world_camera, dtype=float)
        if (rotation.shape != (3, 3) or not np.isfinite(rotation).all()
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6)
                or abs(np.linalg.det(rotation) - 1) > 1e-6):
            raise ValueError('Expected a proper optical-to-world rotation')
        box = self.bbox_xyxy
        if box is not None:
            box = tuple(box)
            if (len(box) != 4 or not all(_number(x) for x in box)
                    or not 0 <= box[0] < box[2] <= hw[1]
                    or not 0 <= box[1] < box[3] <= hw[0]):
                raise ValueError('Input detector box must lie within its source image')
        # Snapshot mutable caller arrays; later/future sensor mutation cannot
        # retroactively change an earlier input record.
        object.__setattr__(self, 'sequence', int(self.sequence))
        object.__setattr__(self, 'capture_time_s', float(self.capture_time_s))
        object.__setattr__(self, 'detector_wall_s', float(self.detector_wall_s))
        object.__setattr__(self, 'image_hw', tuple(int(x) for x in hw))
        object.__setattr__(self, 'bbox_xyxy', None if box is None else tuple(float(x) for x in box))
        object.__setattr__(self, 'intrinsics', tuple(float(x) for x in k))
        object.__setattr__(self, 'rotation_world_camera', tuple(tuple(float(x) for x in row) for row in rotation))


class AlphaBeta:
    """Causal angular filter, with no emitted prediction during measurement gaps."""
    def __init__(self):
        self.angle = None
        self.rate = 0.
        self.last_measurement_s = None
        self.last_step_s = None

    def step(self, capture_time_s, measurement):
        if (not _number(capture_time_s) or capture_time_s < 0
                or self.last_step_s is not None and capture_time_s <= self.last_step_s
                or measurement is not None and not _number(measurement)):
            raise ValueError('Filter requires finite, strictly increasing capture times and finite angles')
        self.last_step_s = float(capture_time_s)
        if measurement is None:
            return None
        measurement = wrap_angle(measurement)
        elapsed = None if self.last_measurement_s is None else capture_time_s - self.last_measurement_s
        if elapsed is None or elapsed > CONFIG['filter_reset_gap_s'] + 1e-9:
            self.angle, self.rate = measurement, 0.
        else:
            prediction = wrap_angle(self.angle + self.rate * elapsed)
            residual = wrap_angle(measurement - prediction)
            self.angle = wrap_angle(prediction + CONFIG['alpha'] * residual)
            self.rate += CONFIG['beta'] * residual / elapsed
        self.last_measurement_s = float(capture_time_s)
        return float(self.angle)

    def update(self, capture_time_s, measurement):
        """Public streaming alias; unlike run_comparison this retains state."""
        return self.step(capture_time_s, measurement)


def _pixel_bearing(uv, sensor):
    fx, fy, cx, cy = sensor.intrinsics
    u, v = uv
    ray = np.asarray(sensor.rotation_world_camera) @ np.array([(u-cx)/fx, (v-cy)/fy, 1.])
    if not np.isfinite(ray).all() or np.linalg.norm(ray[:2]) < .2:
        return None
    return float(np.arctan2(ray[1], ray[0]))


def direct_bearing(sensor):
    """Return capture-time YOLO bearing within the frozen shared envelope.

    This has no depth or actuator interface. A caller using it for flight must
    retain the existing registered-depth, freshness and motor-safety gates.
    """
    observation, shared = transformed_observation(sensor, 'clean', 0)
    if not shared['shared_eligible']:
        return None
    x0, y0, x1, y1 = observation['bbox_xyxy']
    return _pixel_bearing(((x0+x1)/2, (y0+y1)/2), sensor)


def transformed_observation(sensor, variant, seed):
    """Deterministic perturbation keyed by row, with no truth-dependent clipping."""
    if not isinstance(sensor, BearingInput):
        raise TypeError('Only BearingInput sensor records are accepted')
    if variant not in CONFIG['variants']:
        raise ValueError('Unknown comparison variant')
    if not isinstance(seed, Integral) or isinstance(seed, (bool, np.bool_)) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned 32-bit integer')
    box = None if sensor.bbox_xyxy is None else np.asarray(sensor.bbox_xyxy, dtype=float)
    reason = 'target_not_observed' if box is None else 'observed'
    noise = np.zeros(2)
    injected_gap = variant == 'gaps' and any(a <= sensor.sequence <= b
                                            for a, b in CONFIG['gap_sequence_ranges_inclusive'])
    if variant == 'noise':
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), sensor.sequence]))
        noise = rng.normal(0., CONFIG['noise_center_standard_deviation_px'], 2)
        if box is not None:
            box += np.tile(noise, 2)
    if injected_gap:
        box, reason = None, 'injected_measurement_gap'
    h, w = sensor.image_hw
    if box is not None and not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
        box, reason = None, 'perturbed_box_outside_image'
    observation = dict(valid=box is not None, bbox_xyxy=None if box is None else box.tolist(), reason=reason)
    cue = cue_geometry(observation, sensor.image_hw)
    eligible = cue is not None
    if eligible and (abs(cue[0]) > TRAIN_X + 1e-9 or abs(cue[1]) > TRAIN_Y + 1e-9
                     or not TRAIN_HEIGHT[0] <= cue[2] <= TRAIN_HEIGHT[1]):
        eligible, reason = False, 'outside_neural_training_envelope'
    return observation, dict(shared_eligible=bool(eligible), shared_reason=reason,
                             noise_xy_px=noise.tolist(), injected_gap=bool(injected_gap),
                             cue_input=None if cue is None else cue.tolist())


def _neural_bearing(response, readout, sensor, eligible, shared_reason):
    features = np.asarray(response['features'], dtype=float)
    if features.shape != (8,) or not np.isfinite(features).all():
        raise ValueError('Neural response must contain eight finite features')
    if not eligible:
        return None, shared_reason, None
    if (response.get('valid') is not True or features[7] < .05
            or features[5] < np.log1p(.05)):
        return None, 'neural_evidence_unavailable', None
    decoded = np.asarray(readout.predict(features), dtype=float)
    if (decoded.shape != (3,) or not np.isfinite(decoded).all()
            or abs(decoded[0]) > .5 or abs(decoded[1]) > .4
            or not .15 <= decoded[2] <= .60):
        return None, 'neural_readout_out_of_range', None
    h, w = sensor.image_hw
    scale = 391 / max(h, w)
    offset = np.array([(391-w*scale)/2, (391-h*scale)/2])
    uv = (391 * (decoded[:2] + 1) / 2 - offset) / scale
    bearing = _pixel_bearing(uv, sensor)
    return bearing, 'neural_bearing' if bearing is not None else 'invalid_level_bearing', decoded.tolist()


def run_comparison(inputs, brain, readout, *, variant='clean', seed=27101):
    """Run all methods on the same sensor stream; no evaluator truth argument.

    Supply an actual HybridFlyvis instance for learned inference. Tests may
    supply a deterministic stub; the concrete backend class is recorded. Each
    variant resets recurrence. The frozen readout is only queried via predict.
    """
    inputs = tuple(inputs)
    if not inputs:
        raise ValueError('A nonempty sensor stream is required')
    if any(not isinstance(row, BearingInput) for row in inputs):
        raise TypeError('Only BearingInput sensor records are accepted')
    for index, sensor in enumerate(inputs):
        if (sensor.sequence != index
                or index and sensor.capture_time_s <= inputs[index-1].capture_time_s):
            raise ValueError('Comparison requires sequence 0..N-1 with strictly increasing recorded capture times')
    # Validate variant/seed before resetting or executing the network.
    transformed_observation(inputs[0], variant, seed)
    reset_started = time.perf_counter()
    brain.reset()
    reset_wall = time.perf_counter() - reset_started
    smoother = AlphaBeta()
    rows = []
    for sensor in inputs:
        observation, shared = transformed_observation(sensor, variant, seed)
        eligible = shared['shared_eligible']
        started = time.perf_counter()
        direct = None
        if eligible:
            x0, y0, x1, y1 = observation['bbox_xyxy']
            direct = _pixel_bearing(((x0+x1)/2, (y0+y1)/2), sensor)
        direct_wall = time.perf_counter() - started
        started = time.perf_counter()
        filtered = smoother.step(sensor.capture_time_s, direct)
        filter_wall = time.perf_counter() - started
        started = time.perf_counter()
        response = brain.step(salience_mask(observation, sensor.image_hw), CONFIG['neural_hold_s'])
        neural, neural_reason, decoded = _neural_bearing(response, readout, sensor, eligible,
                                                        shared['shared_reason'])
        neural_wall = time.perf_counter() - started
        stimulus = float(response['stimulus_time_s'])
        response_time = float(response['response_time_s'])
        if (not np.isfinite([stimulus, response_time]).all()
                or abs(stimulus - sensor.sequence*.1) > 1e-7
                or abs(response_time - (sensor.sequence+1)*.1) > 1e-7):
            raise ValueError('Neural response clock does not match reset .1-second hold sequence')
        costs = {'mantis_neural': neural_wall, 'direct_yolo': direct_wall,
                 'alpha_beta': direct_wall + filter_wall}
        bearings = {'mantis_neural': neural, 'direct_yolo': direct, 'alpha_beta': filtered}
        shared_unavailable = shared['shared_reason'] if not eligible else 'invalid_level_bearing'
        reasons = {'mantis_neural': neural_reason,
                   'direct_yolo': 'direct_box_bearing' if direct is not None else shared_unavailable,
                   'alpha_beta': 'filtered_box_bearing' if filtered is not None else shared_unavailable}
        common_delay = max(CONFIG['minimum_response_interval_s'], sensor.detector_wall_s + max(costs.values()))
        methods = {name: dict(valid=bearings[name] is not None, heading_world_rad=bearings[name],
                              reason=reasons[name], wall_s=float(costs[name]),
                              standalone_available_time_s=float(sensor.capture_time_s + max(
                                  CONFIG['minimum_response_interval_s'], sensor.detector_wall_s + costs[name])))
                   for name in METHODS}
        rows.append(dict(sequence=sensor.sequence, capture_time_s=sensor.capture_time_s,
                         common_available_time_s=float(sensor.capture_time_s + common_delay),
                         detector_wall_s=sensor.detector_wall_s, bbox_xyxy=observation['bbox_xyxy'],
                         original_bbox_xyxy=None if sensor.bbox_xyxy is None else list(sensor.bbox_xyxy),
                         **shared, methods=methods,
                         neural=dict(features=np.asarray(response['features']).astype(float).tolist(),
                                     valid=bool(response['valid']), decoded=decoded,
                                     stimulus_time_s=stimulus, response_time_s=response_time,
                                     hold_s=CONFIG['neural_hold_s'])))
    return dict(config=copy.deepcopy(CONFIG), variant=variant, seed=int(seed), rows=rows,
                neural_backend=type(brain).__module__ + '.' + type(brain).__name__,
                timing=dict(neural_reset_wall_s=float(reset_wall),
                            per_method_wall_s={name: float(sum(row['methods'][name]['wall_s'] for row in rows))
                                               for name in METHODS},
                            detector_wall_s=float(sum(row.detector_wall_s for row in inputs)),
                            completion_times_are_offline_per_observation=True,
                            achieved_real_time=False))


def score_comparison(rows, truth_rows):
    """Evaluator-only truth join, with no truth returned to any estimator.

    Truth rows must cover the exact sequence IDs; heading_world_rad=None means
    truth is unavailable and is counted explicitly. RMSE compares the identical
    subset where truth and every method are valid. Full and shared-eligible
    coverage prevent presenting selective abstention as an accuracy gain.
    """
    rows, truth_rows = list(rows), list(truth_rows)
    if not rows or len(rows) != len(truth_rows):
        raise ValueError('Scoring requires complete equally sized prediction and truth streams')
    truth = {}
    for row in truth_rows:
        sequence = row.get('sequence')
        bearing = row.get('heading_world_rad')
        if (not isinstance(sequence, Integral) or isinstance(sequence, (bool, np.bool_))
                or sequence < 0 or sequence in truth or 'heading_world_rad' not in row
                or bearing is not None and not _number(bearing)):
            raise ValueError('Invalid or duplicate evaluator truth')
        truth[int(sequence)] = row
    if [row.get('sequence') for row in rows] != list(range(len(rows))) or set(truth) != set(range(len(rows))):
        raise ValueError('Predictions and evaluator truth must cover each sequence exactly once')
    eligible_count = truth_count = eligible_truth = matched_count = 0
    all_counts = {method: 0 for method in METHODS}
    eligible_counts = {method: 0 for method in METHODS}
    errors = {method: [] for method in METHODS}
    common_missing = {}
    for row in rows:
        sequence = row['sequence']
        capture, completion = row.get('capture_time_s'), row.get('common_available_time_s')
        if (not _number(capture) or capture < 0 or not _number(completion) or completion < capture
                or sequence and capture <= rows[sequence-1]['capture_time_s']):
            raise ValueError('Invalid prediction clocks')
        ground = truth[sequence]
        if 'capture_time_s' in ground and (not _number(ground['capture_time_s'])
                                           or abs(ground['capture_time_s']-capture) > 1e-8):
            raise ValueError('Evaluator truth does not match capture time')
        eligible = row.get('shared_eligible')
        if type(eligible) is not bool or set(row.get('methods', {})) != set(METHODS):
            raise ValueError('Invalid shared eligibility or method collection')
        if not eligible:
            reason = row.get('shared_reason', 'unspecified')
            common_missing[reason] = common_missing.get(reason, 0) + 1
        eligible_count += int(eligible)
        has_truth = ground['heading_world_rad'] is not None
        truth_count += int(has_truth)
        eligible_truth += int(eligible and has_truth)
        valid = {}
        for method in METHODS:
            result = row['methods'][method]
            angle, available = result.get('heading_world_rad'), result.get('valid')
            if (type(available) is not bool or available and (not eligible or not _number(angle))
                    or not available and angle is not None):
                raise ValueError('Invalid method estimate or output during shared missing input')
            valid[method] = available
            all_counts[method] += int(available and has_truth)
            eligible_counts[method] += int(available and has_truth and eligible)
        if has_truth and all(valid.values()):
            matched_count += 1
            for method in METHODS:
                errors[method].append(wrap_angle(row['methods'][method]['heading_world_rad'] - ground['heading_world_rad']))
    scores = {}
    for method in METHODS:
        residuals = np.asarray(errors[method])
        scores[method] = dict(matched_capture_rmse_rad=float(np.sqrt(np.mean(residuals**2))) if matched_count else None,
                              matched_capture_mean_abs_rad=float(np.mean(np.abs(residuals))) if matched_count else None,
                              matched_capture_max_abs_rad=float(np.max(np.abs(residuals))) if matched_count else None,
                              truth_prediction_count=all_counts[method],
                              full_truth_coverage=all_counts[method]/truth_count if truth_count else None,
                              eligible_truth_coverage=eligible_counts[method]/eligible_truth if eligible_truth else None)
    baseline = min((scores[name]['matched_capture_rmse_rad'] for name in ('direct_yolo', 'alpha_beta')),
                   default=None) if matched_count else None
    improvement = None if baseline is None or baseline <= 1e-12 else 1 - scores['mantis_neural']['matched_capture_rmse_rad']/baseline
    return dict(recorded_count=len(rows), truth_available_count=truth_count,
                truth_unavailable_count=len(rows)-truth_count, shared_eligible_count=eligible_count,
                shared_eligible_truth_count=eligible_truth, matched_count=matched_count,
                unmatched_truth_count=truth_count-matched_count, shared_missing_reasons=common_missing,
                methods=scores, neural_improvement_over_stronger_baseline_fraction=improvement,
                score_clock='capture_time_s', completion_time_accuracy_evaluated=False,
                neural_coverage_at_least_baselines=all_counts['mantis_neural'] >= max(all_counts.values()),
                automatic_winner_or_superiority_test=False)
