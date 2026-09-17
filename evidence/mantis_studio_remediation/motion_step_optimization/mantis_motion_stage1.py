"""Observational raw-camera Flyvis motion experiment with a frozen benchmark.

The official pretrained DecoderGAVP supplies optic-flow estimates. A separately
owned recurrent state receives camera luminance, never YOLO masks, boxes, truth
or flight commands. This sensor diagnostic has no control authority. Nominal
pixel/s values use the existing, explicitly uncalibrated Sintel scale transfer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from experiments.compare_flow import FARNEBACK, decoder_unit_fields
from experiments.fly_tracking_core import IMAGE_SIDE, raw_flow_mapping, supported_indices
from experiments.runtime import VideoFlyvisAdapter
from flybrain_sim.research_model import center_crop_square, file_sha256

DT = .02
MAX_GAP_S = .65
WARMUP_S = .5
POPULATIONS = ('T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d')
TRANSFER = 'interior BoxEye sum: [right,up] * (436*24/169), invert y to right/down; not calibrated physical speed'


def _time(value):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('capture_time_s must be finite and nonnegative')
    return float(value)


def _image(rgb):
    if (not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3
            or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 2 or rgb.nbytes > 16*1024**2):
        raise ValueError('Camera input must be uint8 HWC RGB, at most 16 MiB')
    return rgb


def summarize_flow(field, centers_rc):
    """Pool only complete retinal kernels, retaining declared decoder units."""
    values = np.asarray(field)
    centers = np.asarray(centers_rc)
    if values.shape != (2, 721) or centers.shape != (721, 2) or not np.isfinite(values).all():
        raise ValueError('Motion fields must contain finite [2,721] decoder outputs')
    indices = supported_indices(centers, [0, 0, IMAGE_SIDE, IMAGE_SIDE])
    if len(indices) < 9:
        raise ValueError('Insufficient interior retinal support')
    field = values[:, indices].astype(np.float64)
    median = np.median(field, axis=1)
    return dict(mean_decoder_xy=field.mean(axis=1).tolist(), median_decoder_xy=median.tolist(),
                mean_decoder_magnitude=float(np.linalg.norm(field, axis=0).mean()),
                rms_decoder_magnitude=float(np.sqrt(np.mean(np.sum(field**2, axis=0)))),
                nominal_velocity_px_s=(np.r_[median, 1.] @ raw_flow_mapping()).tolist(),
                supported_receptors=len(indices))


def calculate_pixel_flow(previous, current, delta_s):
    import cv2
    if (not isinstance(previous, np.ndarray) or previous.shape != (IMAGE_SIDE, IMAGE_SIDE)
            or previous.dtype != np.uint8 or not isinstance(current, np.ndarray)
            or current.shape != previous.shape or current.dtype != np.uint8):
        raise ValueError('Farneback comparison needs matching uint8 391-square luminance')
    _time(delta_s)
    if delta_s <= 0:
        raise ValueError('Frame interval must be positive')
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, **FARNEBACK)
    return decoder_unit_fields(flow, delta_s)


class _MotionBackend:
    def __init__(self, manifest):
        self.adapter = VideoFlyvisAdapter(manifest)
        self._blank_state = None
        self._decoder_warmed = False
        types = self.adapter.network.connectome.nodes.type[:].astype(str)
        self.population_indices = {name: np.flatnonzero(types == name) for name in POPULATIONS}
        if any(len(indices) == 0 for indices in self.population_indices.values()):
            raise ValueError('Pretrained connectome lacks a required T4/T5 population')
        self.centers_rc = self.adapter.eye.receptor_centers.detach().cpu().numpy()+195
        self.provenance = dict(manifest_sha256=self.adapter.locked.manifest_sha256,
            checkpoint_sha256=self.adapter.locked.manifest['files'][self.adapter.locked.manifest['checkpoint']],
            decoder='official pretrained DecoderGAVP flow; strict checkpoint load; eval',
            neural_input='camera luminance; center-square crop; 391-square bilinear antialiased resize',
            integration_dt_s=DT, warmup_s=WARMUP_S, maximum_frame_gap_s=MAX_GAP_S,
            temporal_resampling='causal previous-image zero-order hold; first new stimulus at or after capture',
            unit_transfer_assumption=TRANSFER, decoder_axes=['right', 'up'],
            nominal_velocity_axes=['right', 'down'], control_authority=False,
            fitting=False, device=str(self.adapter.device), packages=self.adapter.runtime,
            runtime_optimizations=['eager_official_decoder_warmup', 'exact_cloned_blank_state_reset'])

    def _clone_state(self, state):
        """Copy dynamic tensors and rebuild their official source/target views.

        Copying the entire state object would retain or duplicate cached edge
        gathers. Rebuilding via the pinned network's own state API ensures each
        restored episode references its own cloned node and edge tensors.
        """
        copied = type(state)(
            nodes=type(state.nodes)(**{key: state.nodes[key].clone() for key in state.nodes}),
            edges=type(state.edges)(**{key: state.edges[key].clone() for key in state.edges}))
        return self.adapter.network._state_api(copied)

    def reset(self):
        adapter = self.adapter
        with adapter.torch.inference_mode():
            if self._blank_state is None:
                blank = adapter.torch.full((1, 1, IMAGE_SIDE, IMAGE_SIDE), .5,
                                           dtype=adapter.torch.float32, device=adapter.device)
                retina = adapter.eye(blank)
                initial = adapter.network.fade_in_state(1., DT, retina[:, 0])
                self._blank_state = self._clone_state(initial)
            # Frozen weights and a fixed blank fade produce the same initial
            # state for every episode. Restore its values; never retain a prior
            # camera's recurrence or repeat 50 integration steps on a gap.
            self.state = self._clone_state(self._blank_state)
            if not self._decoder_warmed:
                activity = self.state.nodes.activity.detach().cpu().numpy()[0].copy()
                # VideoFlyvisAdapter constructs DecoderGAVP lazily. Perform that
                # loading and its first evaluation before accepting camera time,
                # without advancing the recurrent visual model or saving output.
                adapter.decode_flow(activity[None])
                self._decoder_warmed = True

    def prepare(self, rgb):
        import cv2
        adapter = self.adapter
        native = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)/255.
        square, transform = center_crop_square(native[None])
        with adapter.torch.inference_mode():
            tensor = adapter.torch.as_tensor(square[None].copy(), device=adapter.device)
            tensor = adapter.torch.nn.functional.interpolate(tensor, size=(IMAGE_SIDE, IMAGE_SIDE),
                mode='bilinear', align_corners=False, antialias=True)
            retina = adapter.eye(tensor)
            gray = np.rint(np.clip(tensor.cpu().numpy()[0, 0], 0, 1)*255).astype(np.uint8)
        return gray, retina, transform

    def advance(self, retina, steps):
        adapter = self.adapter
        # Bound temporary network states; never accumulate all session activity.
        with adapter.torch.inference_mode():
            while steps:
                count = min(steps, 5)
                states = adapter.network.simulate(retina.repeat(1, count, 1, 1), DT,
                                                   initial_state=self.state, as_states=True)
                self.state = states[-1]
                steps -= count
            activity = self.state.nodes.activity.detach().cpu().numpy()[0].copy()
        if activity.shape != (45669,) or not np.isfinite(activity).all():
            raise ValueError('Flyvis returned invalid neural activity')
        return activity.astype(np.float32, copy=False)

    def decode(self, activity):
        return self.adapter.decode_flow(activity[None])[0]

    def conventional(self, previous, current, delta_s):
        fields = calculate_pixel_flow(previous, current, delta_s)
        adapter = self.adapter
        with adapter.torch.inference_mode():
            filtered = adapter.eye(adapter.torch.as_tensor(fields[None], device=adapter.device), ftype='sum')
        return filtered.detach().cpu().numpy()[0, :, 0]

    def populations(self, activity):
        return {name: dict(mean=float(activity[indices].mean()), std=float(activity[indices].std()),
                           count=len(indices)) for name, indices in self.population_indices.items()}


class RawMotionExperiment:
    """Streaming optic-flow diagnostic; independent from target-mask recurrence.

    Camera captures must be strictly increasing and at most 50 Hz. A gap over
    .65 s resets the recurrence and its .5 s warmup, without bridging the gap.
    ``response_time_s`` is when the model response becomes available on the
    simulation clock; callers must not use it before that time. Wall execution
    time is reported separately and is not a real-time performance guarantee.
    """
    def __init__(self, manifest):
        self.backend = _MotionBackend(manifest)
        self.provenance = self.backend.provenance
        self.reset()

    def reset(self):
        started = time.perf_counter()
        self.backend.reset()
        self.previous_time = self.origin = None
        self.previous_retina = self.previous_gray = self.image_shape = None
        self.next_step = 0
        self.reset_wall_s = time.perf_counter()-started
        return dict(status='reset', fade_in_s=1., reset_wall_s=self.reset_wall_s, control_authority=False)

    def step(self, rgb, capture_time_s):
        started = time.perf_counter()
        rgb = _image(rgb)
        timestamp = _time(capture_time_s)
        gap = None if self.previous_time is None else timestamp-self.previous_time
        if gap is not None and gap < DT-1e-9:
            raise ValueError('Captures must strictly increase by at least .02 seconds (maximum 50 Hz)')
        if self.image_shape is not None and rgb.shape != self.image_shape:
            raise ValueError('Camera dimensions cannot change within a motion episode')
        gap_reset = gap is not None and gap > MAX_GAP_S+1e-9
        if gap_reset:
            self.reset()
            gap = None
        neural_started = time.perf_counter()
        gray, retina, transform = self.backend.prepare(rgb)
        if self.origin is None:
            self.origin = timestamp
        # An image arriving at .25 cannot be used for .02.. .24 integration.
        wanted = math.ceil((timestamp-self.origin)/DT-1e-9)
        held_steps = wanted-self.next_step
        if held_steps > 0:
            if self.previous_retina is None:
                raise RuntimeError('Missing prior camera image for causal integration')
            self.backend.advance(self.previous_retina, held_steps)
            self.next_step += held_steps
        stimulus_time = self.origin+self.next_step*DT
        activity = self.backend.advance(retina, 1)
        field = self.backend.decode(activity)
        neural = summarize_flow(field, self.backend.centers_rc)
        populations = self.backend.populations(activity)
        self.next_step += 1
        response_time = self.origin+self.next_step*DT
        neural_wall_s = time.perf_counter()-neural_started
        conventional = None
        conventional_started = time.perf_counter()
        if self.previous_gray is not None:
            conventional = summarize_flow(self.backend.conventional(self.previous_gray, gray, gap),
                                          self.backend.centers_rc)
        conventional_wall_s = time.perf_counter()-conventional_started
        elapsed = timestamp-self.origin
        valid = bool(elapsed >= WARMUP_S-1e-9 and conventional is not None)
        self.previous_time, self.previous_gray, self.previous_retina = timestamp, gray, retina
        self.image_shape = rgb.shape
        result = dict(status='gap-reset' if gap_reset else 'ok' if valid else 'warming-up', valid=valid,
            capture_time_s=timestamp, stimulus_time_s=stimulus_time, response_time_s=response_time,
            elapsed_since_reset_s=elapsed, warmup_s=WARMUP_S, gap_reset=gap_reset,
            previous_frame_interval_s=gap, neural=neural, conventional=conventional, populations=populations,
            neural_wall_s=neural_wall_s, conventional_wall_s=conventional_wall_s,
            elapsed_wall_s=time.perf_counter()-started, reset_wall_s=self.reset_wall_s,
            control_authority=False, spatial_transform=transform,
            unit_transfer_assumption=TRANSFER, decoder_axes=['right', 'up'],
            nominal_velocity_axes=['right', 'down'])
        # No activity arrays are retained or smuggled into compact telemetry.
        json.dumps(result, allow_nan=False)
        return result


def benchmark_definition():
    """Frozen before inference; training textures are never used to fit a model."""
    motions = [('stationary', [0., 0.]), ('left', [-24., 0.]), ('right', [24., 0.]),
               ('up', [0., -24.]), ('down', [0., 24.]), ('diagonal', [18., -18.])]
    return dict(schema='mantis-raw-motion-benchmark-v1', duration_s=1.2, capture_dt_s=.1,
        integration_dt_s=DT, warmup_s=WARMUP_S, fitting=False, parameter_search=False,
        train_seeds=[1701], heldout_seeds=[2909], velocity_axes=['right', 'down'],
        reference='known constant whole-texture translation, 391-square pixel/s',
        metrics='post-warmup vector RMSE and directional cosine; no lag optimization',
        cases=[dict(name=f'{split}_{name}_{seed}', split=split, seed=seed, motion=name,
                    velocity_px_s=velocity) for split, seed in [('train', 1701), ('heldout', 2909)]
               for name, velocity in motions])


def translated_texture(seed, velocity_px_s, capture_time_s):
    """Translate a larger deterministic texture into the window without wrapping."""
    import cv2
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('seed must be an unsigned integer')
    velocity = np.asarray(velocity_px_s, dtype=float)
    timestamp = _time(capture_time_s)
    if velocity.shape != (2,) or not np.isfinite(velocity).all() or np.any(abs(velocity*timestamp) > 60):
        raise ValueError('Translation must remain inside the fixed texture margin')
    rng = np.random.default_rng(seed)
    source = rng.integers(0, 256, (IMAGE_SIDE+128, IMAGE_SIDE+128), dtype=np.uint8)
    source = cv2.GaussianBlur(source, (0, 0), .8)
    yy, xx = np.mgrid[:IMAGE_SIDE, :IMAGE_SIDE].astype(np.float32)
    sampled = cv2.remap(source, xx+64-velocity[0]*timestamp, yy+64-velocity[1]*timestamp,
                        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return np.repeat(sampled[:, :, None], 3, axis=2)


def score_vectors(prediction, truth):
    prediction, truth = np.asarray(prediction, float), np.asarray(truth, float)
    if (prediction.ndim != 2 or prediction.shape[1] != 2 or truth.shape != prediction.shape
            or not len(prediction) or not np.isfinite(prediction).all() or not np.isfinite(truth).all()):
        raise ValueError('Scoring needs matching nonempty finite [N,2] vectors')
    pred_norm, true_norm = np.linalg.norm(prediction, axis=1), np.linalg.norm(truth, axis=1)
    moving = true_norm > 1e-9
    directional = moving & (pred_norm > 1e-9)
    cosine = np.sum(prediction[directional]*truth[directional], axis=1)/(pred_norm[directional]*true_norm[directional])
    return dict(rows=len(prediction), vector_rmse_px_s=float(np.sqrt(np.mean(np.sum((prediction-truth)**2, axis=1)))),
        mean_vector_error_px_s=float(np.linalg.norm(prediction-truth, axis=1).mean()),
        mean_predicted_magnitude_px_s=float(pred_norm.mean()), moving_truth_rows=int(moving.sum()),
        direction_coverage=float(directional.sum()/moving.sum()) if moving.any() else None,
        mean_direction_cosine=float(np.clip(cosine, -1, 1).mean()) if directional.any() else None)


def run_benchmark(manifest, output):
    folder = Path(output).expanduser().absolute()
    if '..' in folder.parts or any(path.is_symlink() for path in (folder, *folder.parents)):
        raise ValueError('Benchmark output may not use traversal or symlinks')
    folder.mkdir(parents=True, exist_ok=False)
    definition = benchmark_definition()
    (folder/'definition.json').write_text(json.dumps(definition, indent=2, allow_nan=False)+'\n')
    started = time.perf_counter()
    rows, cases = [], []
    try:
        experiment = RawMotionExperiment(manifest)
        provenance = dict(experiment.provenance, definition_sha256=file_sha256(folder/'definition.json'),
            implementation_sha256=file_sha256(Path(__file__)),
            limitation='Synthetic global translation; no natural-video, obstacle avoidance or flight benefit established')
        (folder/'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False)+'\n')
        with (folder/'observations.jsonl').open('x') as telemetry:
            for case in definition['cases']:
                experiment.reset()
                scored = []
                for index in range(round(definition['duration_s']/definition['capture_dt_s'])+1):
                    timestamp = index*definition['capture_dt_s']
                    rgb = translated_texture(case['seed'], case['velocity_px_s'], timestamp)
                    response = experiment.step(rgb, timestamp)
                    row = dict(case=case['name'], split=case['split'], motion=case['motion'],
                               truth_velocity_px_s=case['velocity_px_s'], response=response)
                    telemetry.write(json.dumps(row, allow_nan=False, separators=(',', ':'))+'\n')
                    if response['valid']:
                        scored.append(row)
                        rows.append(row)
                truth = [case['velocity_px_s']]*len(scored)
                metrics = {name: score_vectors([row['response'][name]['nominal_velocity_px_s'] for row in scored], truth)
                           for name in ('neural', 'conventional')}
                metrics['zero'] = score_vectors(np.zeros((len(scored), 2)), truth)
                cases.append(dict(**case, metrics=metrics))
                print(json.dumps(dict(case=case['name'], status='measured', scored_rows=len(scored))), flush=True)
        grouped = {}
        for split in ('train', 'heldout'):
            subset = [row for row in rows if row['split'] == split]
            truth = [row['truth_velocity_px_s'] for row in subset]
            grouped[split] = {name: score_vectors([row['response'][name]['nominal_velocity_px_s'] for row in subset], truth)
                              for name in ('neural', 'conventional')}
            grouped[split]['zero'] = score_vectors(np.zeros((len(subset), 2)), truth)
        heldout = grouped['heldout']
        neural_wins = bool(heldout['neural']['vector_rmse_px_s'] < min(
            heldout['conventional']['vector_rmse_px_s'], heldout['zero']['vector_rmse_px_s']))
        summary = dict(schema='mantis-raw-motion-result-v1', status='completed', inference_executed=True,
            created_utc=datetime.now(timezone.utc).isoformat(), control_authority=False,
            fitting=False, parameter_search=False, raw_neural_arrays_saved=False,
            elapsed_wall_s=time.perf_counter()-started, cases=cases, splits=grouped,
            neural_outperformed_both_heldout_baselines=neural_wins,
            conclusion=('Neural decoder beat both baselines on this frozen synthetic subset only.' if neural_wins
                else 'Neural decoder did not outperform both frozen held-out baselines; no control improvement established.'),
            timing='Prediction after current image integrates for .02 s; constant translation truth; no fitted lag',
            artifacts={path.name: file_sha256(path) for path in folder.iterdir() if path.is_file()})
        (folder/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
        return summary
    except BaseException as error:
        (folder/'failure.json').write_text(json.dumps(dict(status='failed', error=str(error),
            error_type=type(error).__name__, control_authority=False), allow_nan=False)+'\n')
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True, help='New folder for compact frozen benchmark receipts')
    args = parser.parse_args(argv)
    result = run_benchmark(args.manifest, args.output)
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
